package cn.sibat.ksh

import java.io.File

import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions

import scala.collection.mutable.ArrayBuffer

/**
  * Created by kong at 2020/5/21
  */
class TestA2C(maze: Maze) {
  private val load_model = false
  private val state_size = maze.reset().length
  private val action_size = maze.possible_actions.length
  private val value_size = 1
  private val discount_factor = 0.99
  private val actor_lr = 0.001
  private val critic_lr = 0.005
  private var actor: ComputationGraph = _
  private var critic: ComputationGraph = _

  def build_model(): Unit = {
    if (load_model) {
      actor = ComputationGraph.load(new File("actor"), false)
      critic = ComputationGraph.load(new File("critic"), false)
    } else {
      val actor_conf = new NeuralNetConfiguration.Builder()
        .updater(new Adam(actor_lr))
        .graphBuilder()
        .addInputs("input")
        .addLayer("d1", new DenseLayer.Builder().nIn(state_size).nOut(24).activation(Activation.LEAKYRELU).build(), "input")
        .addLayer("d2", new DenseLayer.Builder().nIn(24).nOut(24).activation(Activation.LEAKYRELU).build(), "d1")
        .addLayer("output", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).nOut(action_size).nIn(24).activation(Activation.SOFTMAX).build(), "d2")
        .setOutputs("output")
        .build()
      val critic_conf = new NeuralNetConfiguration.Builder()
        .updater(new Adam(critic_lr))
        .graphBuilder()
        .addInputs("input")
        .addLayer("d1", new DenseLayer.Builder().nIn(state_size).nOut(24).activation(Activation.LEAKYRELU).build(), "input")
        .addLayer("d2", new DenseLayer.Builder().nIn(24).nOut(24).activation(Activation.LEAKYRELU).build(), "d1")
        .addLayer("output", new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nOut(value_size).nIn(24).activation(Activation.IDENTITY).build(), "d2")
        .setOutputs("output")
        .build()
      actor = new ComputationGraph(actor_conf)
      critic = new ComputationGraph(critic_conf)
    }
    actor.init()
    critic.init()
  }

  def get_action(state: (Int, Int)): Int = {
    val arr = maze.get_state(state)
    val input = Nd4j.create(arr).reshape(1, arr.length)
    val policy = actor.output(false, input).head
    val actions = Nd4j.create(maze.possible_actions.map(_.toFloat))
    Nd4j.choice(actions, policy, 1).getNumber().floatValue().toInt
  }

  def train_model(state: (Int, Int), action: Int, reward: Double, next_state: (Int, Int), done: Boolean): Unit = {
    val target = Nd4j.zeros(1L, value_size.toLong)
    val advantages = Nd4j.zeros(1L, action_size.toLong)
    val arr = maze.get_state(state)
    val vec = Nd4j.create(arr).reshape(1, arr.length)
    val value = critic.output(vec).head
    val arr2 = maze.get_state(next_state)
    val next_vec = Nd4j.create(arr2).reshape(1, arr2.length)
    val next_value = critic.output(next_vec).head

    if (done) {
      advantages.put(0, action, reward - value.getNumber().doubleValue())
      target.put(0, 0, reward)
    } else {
      advantages.put(0, action, reward + discount_factor * next_value.getDouble() - value.getDouble())
      target.put(0, 0, reward + discount_factor * next_value.getDouble())
    }

    actor.fit(Array(vec), Array(advantages))
    critic.fit(Array(vec), Array(target))
  }

  /**
    * 执行策略
    *
    */
  def move_by_policy(log: Boolean = false): Unit = {
    var start = (0, 0)
    var count = 0
    val moves = new ArrayBuffer[String]()
    var reward = 0.0
    var done = true
    while (done) {
      val action = get_action(start)
      val next_state = maze.state_after_action(start, action)
      count += 1 //up,right,down,left
      val out = action match {
        case 0 => "up"
        case 1 => "right"
        case 2 => "down"
        case 3 => "left"
        case _ => "null"
      }
      if (log)
        moves += out
      reward = maze.get_reward(start)
      //val next_action = get_action(next_state)
      if (reward > 0)
        done = false
      train_model(start, action, reward, next_state, !done)
      start = next_state
    }
    if (log)
      println(s"policy:move-$count,moves:${moves.mkString(",")},reward:$reward")
  }
}

object TestA2C {
  def main(args: Array[String]): Unit = {
    val maze = new Maze()
    val deepSarsa = new TestA2C(maze)
    deepSarsa.build_model()
    println("before improve:")
    deepSarsa.move_by_policy(true)
    for (i <- 0 to 10000) {
      deepSarsa.move_by_policy()
      print(s"\rexec:$i")
    }
    println("after improve:")
    deepSarsa.move_by_policy(true)
  }
}