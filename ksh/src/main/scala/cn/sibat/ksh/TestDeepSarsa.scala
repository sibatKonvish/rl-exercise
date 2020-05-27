package cn.sibat.ksh

import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/**
  * Created by kong at 2020/5/9
  */
class TestDeepSarsa(maze: Maze) {
  private val load_model = false
  private val action_size = maze.possible_actions.length
  private val state_size = maze.reset().length
  private val discount_factor = 0.99
  private val learning_rate = 0.001
  private var epsilon = 1.0
  private val epsilon_decay = 0.9999
  private val epsilon_min = 0.01
  private var model: ComputationGraph = _

  def predict(input: INDArray): Array[INDArray] = {
    model.output(false, input)
  }

  def build_model(): Unit = {
    val conf = new NeuralNetConfiguration.Builder()
      .updater(new Adam(learning_rate))
      .weightInit(WeightInit.RELU)
      .graphBuilder()
      .addInputs("input")
      .addLayer("d1", new DenseLayer.Builder().nOut(30).nIn(state_size).activation(Activation.RELU).build(), "input")
      .addLayer("d2", new DenseLayer.Builder().nOut(30).nIn(30).activation(Activation.RELU).build(), "d1")
      .addLayer("d3", new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nOut(action_size).nIn(30).build(), "d2")
      .setOutputs("d3")
      .build()
    model = new ComputationGraph(conf)
    model.init()
  }

  def get_action(state: (Int, Int)): Int = {
    var action = 0
    if (Random.nextDouble() < epsilon) {
      action = maze.possible_actions(Random.nextInt(action_size))
    } else {
      val arr = maze.get_state(state)
      val input = Nd4j.create(arr).reshape(1, arr.length)
      val q_values = model.output(false, input)
      action = q_values.head.argMax(1).getNumber().intValue()
    }
    action
  }

  def train_model(state: (Int, Int), action: Int, reward: Double, next_state: (Int, Int), next_action: Int, done: Boolean): Unit = {
    if (epsilon > epsilon_min)
      epsilon *= epsilon_decay

    val state_vec = maze.get_state(state)
    val input = Nd4j.create(state_vec).reshape(1, state_vec.length)
    val target = model.output(input).head
    if (done)
      target.putScalar(action, reward)
    else {
      val next_state_arr = maze.get_state(next_state)
      val next_state_vec = Nd4j.create(next_state_arr).reshape(1, next_state_arr.length)
      val value = model.output(next_state_vec).head.getDouble(next_action.toLong)
      target.putScalar(action, reward + discount_factor * value)
    }

    model.fit(Array(input), Array(target))
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
      val next_action = get_action(next_state)
      if (reward > 0)
        done = false
      train_model(start, action, reward, next_state, next_action, !done)
      start = next_state
    }
    if (log)
      println(s"policy:move-$count,moves:${moves.mkString(",")},reward:$reward")
  }
}

object TestDeepSarsa {
  def main(args: Array[String]): Unit = {
    val maze = new Maze()
    val deepSarsa = new TestDeepSarsa(maze)
    deepSarsa.build_model()
    println("before improve:")
    deepSarsa.move_by_policy(true)
    for (_ <- 0 to 10000) {
      deepSarsa.move_by_policy()
    }
    println("after improve:")
    deepSarsa.move_by_policy(true)
  }
}