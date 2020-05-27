package cn.sibat.ksh.rl

/**
  * Created by kong at 2020/5/11
  *
  * @param actions 动作空间名称
  */
class Action(actions: Array[String]) {
  def size: Int = actions.length

  /**
    * 把动作编码成index
    *
    * @return
    */
  def getActions: Array[Int] = actions.indices.toArray

  /**
    * 动作index的名称
    *
    * @param index 动作索引
    * @return
    */
  def action_name(index: Int): String = actions(index)
}
