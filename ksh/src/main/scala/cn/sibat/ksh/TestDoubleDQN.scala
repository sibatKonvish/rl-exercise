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

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/**
  * Created by kong at 2020/5/21
  */
class TestDoubleDQN(maze: Maze) {
  private val load_model = false
  private val state_size = maze.reset().length
  private val action_size = maze.possible_actions.length

  private val discount_factor = 0.99
  private val learning_rate = 0.001
  private var epsilon = 1.0
  private val epsilon_decay = 0.999
  private val epsilon_min = 0.01
  private val batch_size = 64
  private val train_start = 100
  private val memory = new ArrayBuffer[((Int, Int), Int, Double, (Int, Int), Boolean)]()
  private var model: ComputationGraph = _
  private var target_model: ComputationGraph = _

  def build_model(): Unit = {
    val conf = new NeuralNetConfiguration.Builder()
      .updater(new Adam(learning_rate))
      .weightInit(WeightInit.XAVIER)
      .graphBuilder()
      .addInputs("input")
      .addLayer("d1", new DenseLayer.Builder().nIn(state_size).nOut(24).activation(Activation.RELU).build(), "input")
      .addLayer("d2", new DenseLayer.Builder().nIn(24).nOut(24).activation(Activation.RELU).build(), "d1")
      .addLayer("d3", new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nOut(action_size).nIn(24).build(), "d2")
      .setOutputs("d3")
      .build()
    model = new ComputationGraph(conf)
    model.init()
    target_model = model.clone()
  }

  def update_target_model(): Unit = {
    target_model = model.clone()
  }

  def get_action(state: (Int, Int)): Int = {
    if (Random.nextDouble() <= epsilon)
      Random.nextInt(action_size)
    else {
      val arr = maze.get_state(state)
      val input = Nd4j.create(arr).reshape(1, arr.length)
      val q_value = model.output(false, input)
      q_value.head.argMax(1).getNumber().intValue()
    }
  }

  def append_sample(state: (Int, Int), action: Int, reward: Double, next_state: (Int, Int), done: Boolean): Unit = {
    memory += ((state, action, reward, next_state, done))
    if (epsilon > epsilon_min)
      epsilon *= epsilon_decay
    if (memory.size > 2000) {
      memory.remove(0, memory.length - 2000)
    }
  }

  def train_model(): Unit = {
    if (memory.length > train_start) {
      val batch_size = math.min(this.batch_size, memory.length)
      val mini_batch = sample(batch_size)
      val update_input = Nd4j.zeros(batch_size.toLong, state_size.toLong)
      val update_target = Nd4j.zeros(batch_size.toLong, state_size.toLong)
      val actions = new Array[Int](batch_size)
      val rewards = new Array[Double](batch_size)
      val dones = new Array[Boolean](batch_size)
      for (i <- mini_batch.indices) {
        val state_vec = maze.get_state(mini_batch(i)._1)
        val input = Nd4j.create(state_vec)
        update_input.putRow(i, input)
        actions(i) = mini_batch(i)._2
        rewards(i) = mini_batch(i)._3
        val next_state_vec = maze.get_state(mini_batch(i)._4)
        val input_next = Nd4j.create(next_state_vec)
        update_target.putRow(i, input_next)
        dones(i) = mini_batch(i)._5
      }
      val target = model.output(false, update_input)
      val target_next = model.output(false, update_target)
      val target_val = target_model.output(false, update_target)

      for (i <- 0 until batch_size) {
        if (dones(i))
          target.head.put(i, actions(i), rewards(i))
        else {
          val a = target_next.head.getRow(i).argMax(0).getNumber().longValue()
          target.head.put(i, actions(i), rewards(i) + discount_factor * target_val.head.getRow(i).getDouble(a))
        }
      }
      model.fit(Array(update_input), target)
    }
  }

  def sample(batch_size: Int): Array[((Int, Int), Int, Double, (Int, Int), Boolean)] = {
    val arr = new mutable.HashSet[Int]()
    while (arr.size < batch_size) {
      arr += Random.nextInt(memory.length)
    }
    arr.map(memory).toArray
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
      append_sample(start, action, reward, next_state, !done)
      train_model()
      start = next_state
    }
    update_target_model() //结束更新模型
    if (log)
      println(s"policy:move-$count,moves:${moves.mkString(",")},reward:$reward")
  }
}

object TestDoubleDQN {
  def main(args: Array[String]): Unit = {
    val maze = new Maze()
    val deepSarsa = new TestDoubleDQN(maze)
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