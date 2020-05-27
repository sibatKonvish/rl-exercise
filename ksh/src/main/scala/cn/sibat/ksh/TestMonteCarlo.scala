package cn.sibat.ksh

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/**
  * Created by kong at 2020/5/9
  */
class MonteCarlo(maze: Maze) {
  private val learning_rate = 0.01
  private val discount_factor = 0.9
  private val epsilon = 0.1
  private val samples = new ArrayBuffer[Sample]()
  private val value_table = new mutable.HashMap[(Int, Int), Double]()

  /**
    * 保存状态及反馈
    *
    * @param state  状态
    * @param reward 反馈
    */
  def save_sample(state: (Int, Int), reward: Double): Unit = {
    samples += Sample(state, reward)
  }

  /**
    * 更新价值
    *
    */
  def update(): Unit = {
    var g_t = 0.0
    val visit_state = new ArrayBuffer[(Int, Int)]()
    for (i <- samples.length - 1 to 0 by -1) {
      val state = samples(i).state
      if (!visit_state.contains(state)) {
        visit_state += state
        g_t = discount_factor * (samples(i).reward + g_t)
        val value = value_table.getOrElse(state, 0.0)
        value_table.put(state, value + learning_rate * (g_t - value))
      }
    }
  }

  /**
    * 获取动作
    *
    * @param state 状态
    * @return
    */
  def get_action(state: (Int, Int)): Int = {
    var action = 0
    if (Random.nextDouble() < epsilon) {
      action = maze.possible_actions(Random.nextInt(maze.possible_actions.length))
    } else {
      val next_state = possible_next_state(state)
      action = arg_max(next_state)
    }
    action
  }

  /**
    * 获取最大价值的动作
    *
    * @param next_state 下一动作的价值
    * @return
    */
  def arg_max(next_state: Array[Double]): Int = {
    val max_index_list = new ArrayBuffer[Int]()
    var max_value = next_state.head
    for (index <- next_state.indices) {
      if (next_state(index) > max_value) {
        max_index_list.clear()
        max_value = next_state(index)
        max_index_list += index
      } else if (next_state(index) == max_value)
        max_index_list += index
    }
    max_index_list(Random.nextInt(max_index_list.length))
  }

  /**
    * 当前状态下的各个动作的价值
    *
    * @param state 状态
    * @return
    */
  def possible_next_state(state: (Int, Int)): Array[Double] = {
    val next_state = new Array[Double](maze.possible_actions.length)
    //up,right,down,left
    //Array((-1, 0), (0, 1), (1, 0), (0, -1))
    if (state._1 != 0)
      next_state(0) = value_table.getOrElse((state._1 - 1, state._2), 0.0)
    else
      next_state(0) = value_table.getOrElse(state, 0.0)
    if (state._2 != maze.height - 1)
      next_state(1) = value_table.getOrElse((state._1, state._2 + 1), 0.0)
    else
      next_state(1) = value_table.getOrElse(state, 0.0)
    if (state._1 != maze.width - 1)
      next_state(2) = value_table.getOrElse((state._1 + 1, state._2), 0.0)
    else
      next_state(2) = value_table.getOrElse(state, 0.0)
    if (state._2 != 0)
      next_state(3) = value_table.getOrElse((state._1, state._2 - 1), 0.0)
    else
      next_state(3) = value_table.getOrElse(state, 0.0)
    next_state
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
      start = maze.state_after_action(start, action)
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
      save_sample(start, reward)
    }
    if (log)
      println(s"policy:move-$count,moves:${moves.mkString(",")},reward:$reward")
    update()
    samples.clear()
  }
}

case class Sample(state: (Int, Int), reward: Double)

object MonteCarlo {
  def main(args: Array[String]): Unit = {
    val maze = new Maze()
    val mc = new MonteCarlo(maze)
    println("before improve:")
    mc.move_by_policy(true)
    for (_ <- 0 to 10000) {
      mc.move_by_policy()
    }
    println("after improve:")
    mc.move_by_policy(true)
  }
}