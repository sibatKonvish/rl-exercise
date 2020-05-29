package cn.sibat.ksh.rl

import java.io.File

import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/**
  * Created by kong at 2020/5/28
  */
class DoubleDQN(env: Environment) extends RL {
  private val load_model = false
  private val state_size = env.getInitState.currentState.length
  private val action_size = env.getAction.size

  private val discount_factor = 0.99
  private val learning_rate = 0.001
  private var epsilon = 1.0
  private val epsilon_decay = 0.999
  private val epsilon_min = 0.01
  private val batch_size = 64
  private val train_start = 100
  private val memory = new ArrayBuffer[(State, Int, Double, State, Boolean)]()
  private var model: ComputationGraph = _
  private var target_model: ComputationGraph = _

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
        .addLayer("d1", new DenseLayer.Builder().nIn(state_size).nOut(24).activation(Activation.RELU).build(), "input")
        .addLayer("d2", new DenseLayer.Builder().nIn(24).nOut(24).activation(Activation.RELU).build(), "d1")
        .addLayer("d3", new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nOut(action_size).nIn(24).build(), "d2")
        .setOutputs("d3")
        .build()
      model = new ComputationGraph(conf)
    }
    model.init()
    target_model = model.clone()
  }

  /**
    * 更新原始阶段性模型
    */
  def update_target_model(): Unit = {
    target_model = model.clone()
  }

  /**
    * 添加示例样本
    *
    * @param state      状态
    * @param action     动作
    * @param reward     反馈
    * @param next_state 下一状态
    * @param done       是否结束
    */
  def append_sample(state: State, action: Int, reward: Double, next_state: State, done: Boolean): Unit = {
    memory += ((state, action, reward, next_state, done))
    if (epsilon > epsilon_min)
      epsilon *= epsilon_decay
    if (memory.size > 2000) {
      memory.remove(0, memory.length - 2000)
    }
  }

  /**
    * 训练模型
    */
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
        val state_vec = mini_batch(i)._1.currentState
        val input = Nd4j.create(state_vec)
        update_input.putRow(i, input)
        actions(i) = mini_batch(i)._2
        rewards(i) = mini_batch(i)._3
        val next_state_vec = mini_batch(i)._4.currentState
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

  /**
    * 抽样，抽取batch_size个样本
    *
    * @param batch_size 抽样数
    * @return
    */
  def sample(batch_size: Int): Array[(State, Int, Double, State, Boolean)] = {
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
  override def move_by_policy(log: Boolean = false): Unit = {
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

  /**
    * 获取当前状态的最大价值的动作
    *
    * @param state 状态
    * @return
    */
  override def get_action(state: State): Int = {
    if (Random.nextDouble() <= epsilon)
      Random.nextInt(action_size)
    else {
      val input = Nd4j.create(state.currentState).reshape(1, state.currentState.length)
      val q_value = model.output(false, input)
      q_value.head.argMax(1).getNumber().intValue()
    }
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
      val done = env.asInstanceOf[MultiOD].ods.isEmpty
      append_sample(start, action, reward, next_state, done)
      train_model()
      start = next_state
    }
    if (log) {
      println(s"policy:move-$count,moves:${moves.mkString(",")},reward:$reward")
      println(states.mkString(";"))
    }
    moves.clear()
  }
}

object DoubleDQN {
  def main(args: Array[String]): Unit = {
    val env = new MultiODWithDetectAndDirectEnv(5, 5, 1)
    val ods = Array((0, 1, 1, 3, 2), (1, 2, 2, 1, 3), (2, 4, 0, 0, 4))
    env.setODS(ods)
    val sarsa = new DoubleDQN(env)
    sarsa.build_model()
    println("before improve:")
    sarsa.move_by_policy2(true)
    for (i <- 0 to 10000) {
      env.reset(ods)
      sarsa.move_by_policy2(false)
      if (i % 5 == 0)
        sarsa.update_target_model()
      print(s"\rexec:$i")
    }
    val model = new File("ksh/src/main/resources/ddqn")
    sarsa.model.save(model, true)
    println("after improve:")
    env.reset(ods)
    sarsa.move_by_policy2(true)
  }
}
