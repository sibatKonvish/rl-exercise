package cn.sibat.ksh.rl

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/**
  * Created by kong at 2020/5/11
  */
abstract class RL {

  /**
    * 获取当前状态的最大价值的动作
    *
    * @param state 状态
    * @return
    */
  def get_action(state: State): Int

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
      } else if (max_value == state_action(index))
        max_index_list += index
    }
    max_index_list(Random.nextInt(max_index_list.length))
  }

  /**
    * 执行策略
    *
    */
  def move_by_policy(log: Boolean = false): Unit
}
