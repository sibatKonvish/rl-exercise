package cn.sibat.ksh.rl

import java.io.File

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
  * Created by kong at 2020/5/27
  */
class DeepSarsa(env: Environment) extends RL {
  private val load_model = false
  private val action_size = env.getAction.size
  private val state_size = env.getInitState.getDimension
  private val discount_factor = 0.99
  private val learning_rate = 0.001
  private var epsilon = 1.0
  private val epsilon_decay = 0.9999
  private val epsilon_min = 0.01
  private var model: ComputationGraph = _

  /**
    * 预测action的概率
    *
    * @param input 环境输入
    * @return
    */
  def predict(input: INDArray): Array[INDArray] = {
    model.output(false, input)
  }

  /**
    * 建立与初始化模型
    * 如果加载训练好的模型则输入model_path,否则不需要
    *
    * @param model_path 模型地址
    */
  def build_model(model_path: String = "null"): Unit = {
    if (load_model) {
      model = ComputationGraph.load(new File(model_path), false)
    } else {
      val conf = new NeuralNetConfiguration.Builder()
        .updater(new Adam(learning_rate))
        .weightInit(WeightInit.XAVIER)
        .graphBuilder()
        .addInputs("input")
        .addLayer("d1", new DenseLayer.Builder().nOut(30).nIn(state_size).activation(Activation.RELU).build(), "input")
        .addLayer("d2", new DenseLayer.Builder().nOut(30).nIn(30).activation(Activation.RELU).build(), "d1")
        .addLayer("d3", new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nOut(action_size).nIn(30).build(), "d2")
        .setOutputs("d3")
        .build()
      model = new ComputationGraph(conf)
    }
    model.init()
  }

  /**
    * 训练模型
    *
    * @param state       当前状态
    * @param action      当前动作
    * @param reward      反馈
    * @param next_state  下一状态
    * @param next_action 下一动作
    * @param done        是否结束
    */
  def train_model(state: State, action: Int, reward: Double, next_state: State, next_action: Int, done: Boolean): Unit = {
    if (epsilon > epsilon_min)
      epsilon *= epsilon_decay

    val input = Nd4j.create(state.currentState).reshape(1, state.currentState.length)
    val target = model.output(input).head
    if (done)
      target.putScalar(action, reward)
    else {
      val next_state_vec = Nd4j.create(next_state.currentState).reshape(1, next_state.currentState.length)
      val value = model.output(next_state_vec).head.getDouble(next_action.toLong)
      target.putScalar(action, reward + discount_factor * value)
    }

    model.fit(Array(input), Array(target))
  }

  /**
    * 获取当前状态的最大价值的动作
    *
    * @param state 状态
    * @return
    */
  override def get_action(state: State): Int = {
    var action = 0
    if (Random.nextDouble() < epsilon) {
      action = Random.nextInt(action_size)
    } else {
      val input = Nd4j.create(state.currentState).reshape(1, state.currentState.length)
      val q_values = model.output(false, input)
      action = q_values.head.argMax(1).getNumber().intValue()
    }
    action
  }

  /**
    * 执行策略
    *
    */
  override def move_by_policy(log: Boolean): Unit = {
    var start = env.getInitState
    var count = 0
    val moves = new ArrayBuffer[String]()
    var reward = 0.0
    var done = true
    while (done) {
      val action = get_action(start)
      val next_state = env.state_after_action(start, action)
      count += 1 //up,right,down,left
      val out = env.getAction.action_name(action)
      if (log)
        moves += out
      reward = env.get_reward(start)
      val next_action = get_action(next_state)
      if (reward > 0)
        done = false
      train_model(start, action, reward, next_state, next_action, !done)
      start = next_state
    }
    if (log)
      println(s"policy:move-$count,moves:${moves.mkString(",")},reward:$reward")
  }

  def move_by_policy2(log: Boolean): Unit = {
    var start = env.getInitState
    var count = 0
    val moves = new ArrayBuffer[String]()
    val states = new ArrayBuffer[State]()
    var reward = 0.0
    while (env.asInstanceOf[MultiOD].ods.nonEmpty) {
      val action = get_action(start)
      val next_state = env.state_after_action(start, action)
      count += 1 //up,right,down,left
      val out = env.getAction.action_name(action)
      if (log) {
        moves += out
        states += start
      }
      reward = env.get_action_reward(start, action)
      val next_action = get_action(next_state)
      val done = env.asInstanceOf[MultiOD].ods.isEmpty
      train_model(start, action, reward, next_state, next_action, done)
      start = next_state
    }
    if (log) {
      println(s"policy:move-$count,moves:${moves.mkString(",")},reward:$reward")
      println(states.mkString(";"))
    }
    moves.clear()
  }
}

object DeepSarsa {
  def main(args: Array[String]): Unit = {
    val env = new MultiODWithDetectAndDirectEnv(5, 5, 1)
    val ods = Array((0, 1, 1, 3, 2), (1, 2, 2, 1, 3), (2, 4, 0, 0, 4))
    env.setODS(ods)
    val sarsa = new DeepSarsa(env)
    sarsa.build_model()
    println("before improve:")
    sarsa.move_by_policy2(true)
    for (i <- 0 to 10000) {
      env.reset(ods)
      sarsa.move_by_policy2(false)
      print(s"\rexec:$i")
    }
    val model = new File("ksh/src/main/resources/deepsarsa")
    sarsa.model.save(model, true)
    println("after improve:")
    env.reset(ods)
    sarsa.move_by_policy2(true)
  }
}
