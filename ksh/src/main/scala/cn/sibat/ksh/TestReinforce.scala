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
  * Created by kong at 2020/5/20
  */
class TestReinforce(maze: Maze) {
  private val load_model = false
  private val action_size = maze.possible_actions.length
  private val state_size = maze.reset().length
  private val discount_factor = 0.99
  private val learning_rate = 0.001
  private var model: ComputationGraph = _
  private val states = new ArrayBuffer[(Int, Int)]()
  private val actions = new ArrayBuffer[Int]()
  private val rewards = new ArrayBuffer[Double]()

  def predict(input: INDArray): Array[INDArray] = {
    model.output(false, input)
  }

  def build_model(): Unit = {
    val conf = new NeuralNetConfiguration.Builder()
      .weightInit(WeightInit.RELU)
      .graphBuilder()
      .addInputs("input")
      .addLayer("d1", new DenseLayer.Builder().nOut(24).nIn(state_size).activation(Activation.RELU).build(), "input")
      .addLayer("d2", new DenseLayer.Builder().nOut(24).nIn(24).activation(Activation.RELU).build(), "d1")
      .addLayer("d3", new DenseLayer.Builder().nOut(action_size).nIn(24).activation(Activation.SOFTMAX).build(), "d2")
      .setOutputs("d3")
      .build()
    model = new ComputationGraph(conf)
    model.init()
  }

  def optimizer(): Unit = {
    val action = Nd4j.create(1)
    val discounted_rewards = Nd4j.create(1)
    val action_prob = Nd4j.sum(action.mul(model.output().head), 1)
    val cross_entropy = action_prob.logEntropy(0).mul(discounted_rewards)
    val loss = -Nd4j.sum(cross_entropy).varNumber().doubleValue()
    val op = new Adam(learning_rate)
  }

  def get_action(state: (Int, Int)): Int = {
    val arr = maze.get_state(state)
    val input = Nd4j.create(arr).reshape(1, arr.length)
    val q_values = model.output(false, input)
    Nd4j.choice(Nd4j.create(maze.possible_actions.map(_.toDouble)), q_values.head, 1).getNumber().intValue()
  }

  def train_model(state: (Int, Int), action: Int, reward: Double, next_state: (Int, Int), next_action: Int, done: Boolean): Unit = {
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
