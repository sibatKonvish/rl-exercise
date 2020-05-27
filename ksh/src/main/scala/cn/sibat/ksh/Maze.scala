package cn.sibat.ksh

import scala.collection.mutable.ArrayBuffer

/**
  * Created by kong at 2020/5/8
  */
class Maze {
  private var transition_probability = 1.0
  var width = 4
  var height = 4
  private val reward = Array.ofDim[Double](width, height)
  val possible_actions = Array(0, 1, 2, 3) //up,right,down,left
  private val actions = Array((-1, 0), (0, 1), (1, 0), (0, -1))
  reward(2)(2) = 1.0
  reward(1)(2) = -1.0
  reward(2)(1) = -1.0
  private val all_state = reward.indices.flatMap(i => reward.head.indices.map(i2 => (i, i2))).toArray

  def reset(): Array[Double] = {
    Array(1.0, 2.0, -1, -1, 2.0, 1.0, -1, -1, 2.0, 2.0, 1)
  }

  def get_state(state: (Int, Int)): Array[Double] = {
    val states = new ArrayBuffer[Double]()
    for (i <- reward.indices; j <- reward(i).indices) {
      if (reward(i)(j) > 0) {
        states.append(i - state._1)
        states.append(j - state._2)
        states.append(1.0)
      } else if (reward(i)(j) < 0) {
        states.append(i - state._1)
        states.append(j - state._2)
        states.append(-1.0)
        states.append(-1.0)
      }
    }
    states.toArray
  }

  /**
    * 获得当前状态下执行action的反馈
    *
    * @param state  当前状态
    * @param action 动作
    * @return
    */
  def get_action_reward(state: (Int, Int), action: Int): Double = {
    val next_state = state_after_action(state, action)
    reward(next_state._1)(next_state._2)
  }

  /**
    * 获得当前状态下的反馈
    *
    * @param state 当前状态
    * @return
    */
  def get_reward(state: (Int, Int)): Double = {
    reward(state._1)(state._2)
  }

  /**
    * 当前状态执行完action后的下一个状态
    *
    * @param tuple 当前状态
    * @param index 动作
    * @return
    */
  def state_after_action(tuple: (Int, Int), index: Int): (Int, Int) = {
    val action = actions(index)
    check_boundary((tuple._1 + action._1, tuple._2 + action._2))
  }

  /**
    * 检查当前状态的合理性，并转换为合理状态
    *
    * @param state 当前状态
    * @return
    */
  def check_boundary(state: (Int, Int)): (Int, Int) = {
    val v1 = math.min(math.max(state._1, 0), width - 1)
    val v2 = math.min(math.max(state._2, 0), height - 1)
    (v1, v2)
  }

  def get_transition_prob: Double = transition_probability

  def get_all_states: Array[(Int, Int)] = all_state
}