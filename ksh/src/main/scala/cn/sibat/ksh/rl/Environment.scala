package cn.sibat.ksh.rl

import scala.collection.mutable

/**
  * Created by kong at 2020/5/11
  */
abstract class Environment {
  protected val reward = new mutable.HashMap[State, Double]()
  private var init_state: State = new State(1)
  private var action = new Action(Array("l", "r"))

  /**
    * 设置初始化的动作
    *
    * @param action 动作空间
    */
  def setAction(action: Action): Unit = {
    this.action = action
  }

  /**
    * 获取初始化的状态
    */
  def getAction: Action = this.action

  /**
    * 设置初始化的状态
    *
    * @param start 初始化状态
    */
  def setInitState(start: State): Unit = {
    this.init_state = start
  }

  /**
    * 获取初始化的状态
    */
  def getInitState: State = this.init_state

  /**
    * 获得当前状态下执行action的反馈
    *
    * @param state  当前状态
    * @param action 动作
    * @return
    */
  def get_action_reward(state: State, action: Int): Double = {
    val next_state = state_after_action(state, action)
    reward.getOrElse(next_state, 0.0)
  }

  /**
    * 获得当前状态下的反馈
    *
    * @param state 当前状态
    * @return
    */
  def get_reward(state: State): Double= reward.getOrElse(state, 0.0)

  /**
    * 当前状态执行完action后的下一个状态
    *
    * @param state  当前状态
    * @param action 动作
    * @return
    */
  def state_after_action(state: State, action: Int): State

  /**
    * 检查当前状态的合理性，并转换为合理状态
    *
    * @param vec 当前状态
    * @return
    */
  def check_boundary(vec: Array[Double]): State
}
