package cn.sibat.ksh.rl

/**
  * Created by kong at 2020/5/11
  */
class TwoEnv(width: Int, height: Int) extends Environment {
  private val actions = Array(Array(0, -1), Array(0, 1), Array(-1, 0), Array(1, 0))
  reward.put(new State(Array(2.0, 2.0)), 1.0)
  reward.put(new State(Array(1.0, 2.0)), -1.0)
  reward.put(new State(Array(2.0, 1.0)), -1.0)
  setInitState(new State(2))
  setAction(new Action(Array("l", "r", "u", "d")))

  /**
    * 当前状态执行完action后的下一个状态
    *
    * @param state  当前状态
    * @param action 动作
    * @return
    */
  override def state_after_action(state: State, action: Int): State = {
    val s = actions(action)
    val vec = state.currentState.clone()
    for (i <- s.indices) {
      vec(i) += s(i)
    }
    check_boundary(vec)
  }

  /**
    * 检查当前状态的合理性，并转换为合理状态
    *
    * @param vec 当前状态
    * @return
    */
  override def check_boundary(vec: Array[Double]): State = {
    val v1 = math.min(math.max(vec.head, 0), width - 1)
    val v2 = math.min(math.max(vec.last, 0), height - 1)
    new State(Array(v1, v2))
  }
}
