package cn.sibat.ksh.rl

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/**
  * Created by kong at 2020/5/19
  */
class QLearning(env: Environment) extends RL {
  private val learning_rate = 0.01
  private val discount_factor = 0.9
  private val epsilon = 0.1
  private val q_table = new mutable.HashMap[State, Array[Double]]()

  /**
    * 学习q_table表
    *
    * @param state      当前状态
    * @param action     动作
    * @param reward     反馈
    * @param next_state 下一状态
    */
  def learn(state: State, action: Int, reward: Double, next_state: State): Unit = {
    val current_q = q_table.getOrElseUpdate(state, new Array[Double](env.getAction.size))(action)
    val next_q = q_table.getOrElseUpdate(state, new Array[Double](env.getAction.size))
    val new_q = reward + discount_factor * next_q.max
    q_table(state)(action) += learning_rate * (new_q - current_q)
  }

  /**
    * 获取当前状态的最大价值的动作
    *
    * @param state 状态
    * @return
    */
  override def get_action(state: State): Int = {
    var action = 0
    if (Random.nextDouble() < epsilon)
      action = env.getAction.getActions(Random.nextInt(env.getAction.size))
    else {
      val state_action = q_table.getOrElse(state, new Array[Double](env.getAction.size))
      action = env.getAction.getActions(arg_max(state_action))
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
    while (reward == 0.0) {
      val action = get_action(start)
      val next_state = env.state_after_action(start, action)
      count += 1 //up,right,down,left
      val out = env.getAction.action_name(action)
      if (log)
        moves += out
      reward = env.get_action_reward(start, action)
      learn(start, action, reward, next_state)
      start = next_state
    }
    if (log)
      println(s"policy:move-$count,moves:${moves.mkString(",")},reward:$reward")
    moves.clear()
  }

  def move_by_policy2(log: Boolean): Unit = {
    var start = env.getInitState
    var count = 0
    val moves = new ArrayBuffer[String]()
    val states = new ArrayBuffer[State]()
    var reward = 0.0
    while (env.asInstanceOf[MultiODEnv].ods.nonEmpty) {
      val action = get_action(start)
      val next_state = env.state_after_action(start, action)
      count += 1 //up,right,down,left
      val out = env.getAction.action_name(action)
      if (log) {
        moves += out
        states += start
      }
      reward = env.get_action_reward(start, action)
      learn(start, action, reward, next_state)
      start = next_state
    }
    if (log) {
      println(s"policy:move-$count,moves:${moves.mkString(",")},reward:$reward")
      println(states.mkString(";"))
    }
    moves.clear()
  }
}

object QLearning {
  def main(args: Array[String]): Unit = {
    val env = new MultiODEnv(6, 6)
    val ods = Array((0, 1, 1, 3, 2), (1, 2, 2, 1, 3), (2, 3, 2, 0, 2))
    env.setODS(ods)
    val sarsa = new QLearning(env)
    println("before improve:")
    sarsa.move_by_policy2(true)
    for (i <- 0 to 10000) {
      env.reset(ods)
      sarsa.move_by_policy2(false)
      print(s"\rexec:$i")
    }
    println("after improve:")
    env.reset(ods)
    sarsa.move_by_policy2(true)
  }
}