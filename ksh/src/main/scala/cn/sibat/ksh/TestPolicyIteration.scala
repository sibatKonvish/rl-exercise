package cn.sibat.ksh

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/**
  * Created by kong at 2020/5/8
  */
class TestPolicyIteration(env: Maze) {
  private var value_table = Array.ofDim[Double](env.width, env.height)
  private var policy_table = Array.ofDim[Double](env.width, env.height, env.possible_actions.length).map(_.map(_.map(_ => 0.25)))
  private val discount_factor = 0.9
  this.policy_table(2)(2) = Array() //target point

  /**
    * 策略的评估，计算当前策略的价值
    *
    */
  def policy_evaluation(): Unit = {
    val next_value_table = Array.ofDim[Double](env.width, env.height)
    for (state <- env.get_all_states) {
      var value = 0.0
      for (action <- env.possible_actions) {
        val next_state = env.state_after_action(state, action)
        val reward = env.get_action_reward(state, action)
        val next_value = get_value(next_state)
        val prob = if (get_policy(state).nonEmpty) get_policy(state)(action) else 0.0
        value += prob * (reward + discount_factor * next_value)
      }
      next_value_table(state._1)(state._2) = value
      if (state == (2, 2))
        next_value_table(2)(2) = 0.0
    }
    value_table = next_value_table
  }

  /**
    * 提升策略
    *
    */
  def policy_improvement(): Unit = {
    val next_policy = policy_table
    for (state <- env.get_all_states if state != (2, 2)) {
      var value = -99999.0
      val max_index = new ArrayBuffer[Int]()
      val result = Array(0.0, 0.0, 0.0, 0.0)

      for (index <- env.possible_actions.indices) {
        val action = env.possible_actions(index)
        val next_state = env.state_after_action(state, action)
        val reward = env.get_action_reward(state, action)
        val next_value = get_value(next_state)
        val temp = reward + discount_factor * next_value

        if (temp == value)
          max_index += index
        else if (temp > value) {
          value = temp
          max_index.clear()
          max_index += index
        }
      }

      val prob = 1.0 / max_index.length
      max_index.foreach(i => result(i) = prob)

      next_policy(state._1)(state._2) = result
    }
    policy_table = next_policy
  }

  /**
    * 随机挑选动作
    *
    * @param state 状态
    * @return
    */
  def get_action(state: (Int, Int)): Int = {
    val random_pick = Random.nextInt(100) / 100.0

    val policy = get_policy(state)
    var policy_sum = 0.0
    var count = -1
    while (policy_sum <= random_pick && count < policy.length) {
      count += 1
      policy_sum += policy(count)
    }
    count
  }

  /**
    * 获取当前状态的策略
    *
    * @param state 状态
    * @return
    */
  def get_policy(state: (Int, Int)): Array[Double] = {
    if (state == (2, 2))
      Array()
    else policy_table(state._1)(state._2)
  }

  /**
    * 获取当前状态的价值
    *
    * @param state 状态
    * @return
    */
  def get_value(state: (Int, Int)): Double = {
    value_table(state._1)(state._2)
  }

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
      val action = get_action(start)
      start = env.state_after_action(start, action)
      count += 1 //left,up,right,down
      val out = action match {
        case 0 => "up"
        case 1 => "right"
        case 2 => "down"
        case 3 => "left"
        case _ => "null"
      }
      moves += out
      reward = env.get_reward(start)
    }
    println(s"policy:move-$count,moves:${moves.mkString(",")},reward:$reward")
  }

}

object TestPolicyIteration {
  def main(args: Array[String]): Unit = {
    val env = new Maze()
    val policy = new TestPolicyIteration(env)
    println("before improve:")
    policy.move_by_policy()
    for (i <- 0 to 10) {
      policy.policy_improvement()
      policy.policy_evaluation()
    }
    println("after improve:")
    policy.move_by_policy()
  }
}