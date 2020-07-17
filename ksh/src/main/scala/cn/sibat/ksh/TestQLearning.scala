package cn.sibat.ksh

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/**
  * Created by kong at 2020/5/9
  */
class TestQLearning(maze: Maze) {
  private val learning_rate = 0.01
  private val discount_factor = 0.9
  private val epsilon = 0.1
  private val q_table = new mutable.HashMap[(Int, Int), Array[Double]]()

  /**
    * 学习q_table表
    *
    * @param state      当前状态
    * @param action     动作
    * @param reward     反馈
    * @param next_state 下一状态
    */
  def learn(state: (Int, Int), action: Int, reward: Double, next_state: (Int, Int)): Unit = {
    val current_q = q_table.getOrElseUpdate(state, new Array[Double](maze.possible_actions.length))(action)
    val next_q = q_table.getOrElseUpdate(state, new Array[Double](maze.possible_actions.length))
    val new_q = reward + discount_factor * next_q.max
    q_table(state)(action) += learning_rate * (new_q - current_q)
  }

  /**
    * 获取当前状态的最大价值的动作
    *
    * @param state 状态
    * @return
    */
  def get_action(state: (Int, Int)): Int = {
    var action = 0
    if (Random.nextDouble() < epsilon)
      action = maze.possible_actions(Random.nextInt(maze.possible_actions.length))
    else {
      val state_action = q_table.getOrElse(state, new Array[Double](maze.possible_actions.length))
      action = arg_max(state_action)
    }
    action
  }

  /**
    * 选取最大价值的动作
    *
    * @param state_action 动作对应的价值
    * @return
    */
  def arg_max(state_action: Array[Double]): Int = {
    val max_index_list = new ArrayBuffer[Int]()
    var max_value = state_action.head
    for (index <- state_action.indices) {
      if (state_action(index) > max_value) {
        max_index_list.clear()
        max_value = state_action(index)
        max_index_list += index
      } else if (state_action(index) == max_value)
        max_index_list += index
    }
    max_index_list(Random.nextInt(max_index_list.length))
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
    while (reward == 0.0) {
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
      reward = maze.get_action_reward(start, action)
      learn(start, action, reward, next_state)
      start = next_state
    }
    if (log)
      println(s"policy:move-$count,moves:${moves.mkString(",")},reward:$reward")
  }
}

object TestQLearning {
  def main(args: Array[String]): Unit = {
    val maze = new Maze()
    val sarsa = new TestQLearning(maze)
    println("before improve:")
    sarsa.move_by_policy(true)
    for (_ <- 0 to 10000) {
      sarsa.move_by_policy()
    }
    println("after improve:")
    sarsa.move_by_policy(true)
  }
}