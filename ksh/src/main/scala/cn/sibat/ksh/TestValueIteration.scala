package cn.sibat.ksh

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/**
  * Created by kong at 2020/5/8
  */
class TestValueIteration(maze: Maze) {
  private var value_table = Array.ofDim[Double](maze.width, maze.height)
  private val discount_factor = 0.9

  /**
    * 基于价值迭代
    *
    */
  def value_iteration(): Unit = {
    val next_value_table = Array.ofDim[Double](maze.width, maze.height)

    for (state <- maze.get_all_states) {
      val value_list = new ArrayBuffer[Double]()

      for (action <- maze.possible_actions) {
        val next_state = maze.state_after_action(state, action)
        val reward = maze.get_action_reward(state, action)
        val next_value = get_value(next_state)
        value_list += reward + discount_factor * next_value
      }
      next_value_table(state._1)(state._2) = value_list.max
      if (state == (2, 2))
        next_value_table(2)(2) = 0.0
    }
    value_table = next_value_table
  }

  /**
    * 获取最大价值的动作列表，因为最大价值有可能几个动作
    *
    * @param state 状态
    * @return
    */
  def get_action(state: (Int, Int)): Array[Int] = {
    if (state == (2, 2))
      return Array()
    val action_list = new ArrayBuffer[Int]()
    var max_value = -99999.0
    for (action <- maze.possible_actions) {
      val next_state = maze.state_after_action(state, action)
      val reward = maze.get_action_reward(state, action)
      val next_value = get_value(next_state)
      val value = reward + discount_factor * next_value

      if (value > max_value) {
        action_list.clear()
        action_list.append(action)
        max_value = value
      } else if (value == max_value)
        action_list += action
    }
    action_list.toArray
  }

  /**
    * 当前状态的价值
    *
    * @param state 状态
    * @return
    */
  def get_value(state: (Int, Int)): Double = value_table(state._1)(state._2)

  /**
    * 执行策略
    *
    */
  def move_by_policy(): Unit = {
    var start = (0, 0)
    var count = 0
    val moves = new ArrayBuffer[String]()
    var reward = 0.0
    while (reward == 0.0) {
      val actions = get_action(start)
      val action = actions(Random.nextInt(actions.length))
      start = maze.state_after_action(start, action)
      count += 1 //left,up,right,down
      val out = action match {
        case 0 => "up"
        case 1 => "right"
        case 2 => "down"
        case 3 => "left"
        case _ => "null"
      }
      moves += out
      reward = maze.get_reward(start)
    }
    println(s"policy:move-$count,moves:${moves.mkString(",")},reward:$reward")
  }

}

object TestValueIteration {
  def main(args: Array[String]): Unit = {
    val maze = new Maze()
    val valueIteration = new TestValueIteration(maze)
    println("before improve:")
    valueIteration.move_by_policy()
    for (i <- 0 to 10) {
      valueIteration.value_iteration()
    }
    println("after improve:")
    valueIteration.move_by_policy()
  }
}
