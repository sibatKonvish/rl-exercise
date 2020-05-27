package cn.sibat.ksh.rl

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

trait MultiOD {
  val ods = new ArrayBuffer[(Int, Int, Int, Int, Int)]()
}

/**
  * Created by kong at 2020/5/18
  */
class ODEnv(width: Int, height: Int) extends Environment {
  private val actions = Array(Array(0, -1), Array(0, 1), Array(-1, 0), Array(1, 0))
  reward.put(new State(Array(1.0, 1.0, 3.0, 4.0)), 1.0)
  setInitState(new State(4))
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
    if (vec(0) == 1.0 && vec(1) == 1.0) {
      vec(2) += s.head
      vec(3) += s.last
    } else {
      vec(0) += s.head
      vec(1) += s.last
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
    for (i <- 0 until vec.length / 2) {
      vec(i * 2) = math.min(math.max(vec(i * 2), 0), width - 1)
      vec(i * 2 + 1) = math.min(math.max(vec(i * 2 + 1), 0), height - 1)
    }
    new State(vec)
  }
}

/**
  * Created by kong at 2020/5/18
  */
class MultiODEnv(width: Int, height: Int) extends Environment with MultiOD {
  private val actions = Array(Array(0, -1), Array(0, 1), Array(-1, 0), Array(1, 0))
  private val pick = new ArrayBuffer[Int]()
  private val drop = new ArrayBuffer[Int]()
  setInitState(new State(2))
  setAction(new Action(Array("l", "r", "u", "d")))

  /**
    * 重新开始
    *
    * @param arr Array
    */
  def reset(arr: Array[(Int, Int, Int, Int, Int)]): Unit = {
    ods.clear()
    ods ++= arr
    pick.clear()
    drop.clear()
  }

  /**
    * 设置OD集
    *
    * @param arr od集
    */
  def setODS(arr: Array[(Int, Int, Int, Int, Int)]): Unit = {
    ods ++= arr
  }

  /**
    * 获得当前状态下的反馈
    *
    * @param state 当前状态
    * @return
    */
  override def get_reward(state: State): Double = {
    var reward = 0.0
    for (od <- ods) {
      if (pick.isEmpty || !pick.contains(od._1)) {
        if (state.currentState.head == od._2 && state.currentState.last == od._3) {
          pick += od._1
          //          reward += 0.5
        }
      } else if (pick.contains(od._1) && state.currentState.head == od._4 && state.currentState.last == od._5) {
        reward += 1.0
        pick.remove(pick.indexOf(od._1)) //到站
        drop += od._1
      }
    }
    drop.foreach(t => {
      val index = ods.indexWhere(_._1 == t)
      if (index != -1)
        ods.remove(index)
    })
    reward
  }


  /**
    * 获得当前状态下执行action的反馈
    *
    * @param state  当前状态
    * @param action 动作
    * @return
    */
  override def get_action_reward(state: State, action: Int): Double = {
    val next_state = state_after_action(state, action)
    get_reward(next_state)
  }

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
    for (i <- 0 until vec.length / 2) {
      vec(i * 2) = math.min(math.max(vec(i * 2), 0), width - 1)
      vec(i * 2 + 1) = math.min(math.max(vec(i * 2 + 1), 0), height - 1)
    }
    new State(vec)
  }
}

/**
  * Created by kong at 2020/5/18
  */
class MultiODWithCapacityEnv(width: Int, height: Int) extends Environment with MultiOD {
  private val actions = Array(Array(0, -1), Array(0, 1), Array(-1, 0), Array(1, 0))
  private val pick = new ArrayBuffer[Int]()
  private val drop = new ArrayBuffer[Int]()
  setInitState(new State(2))
  setAction(new Action(Array("l", "r", "u", "d")))

  /**
    * 重新开始
    *
    * @param arr Array
    */
  def reset(arr: Array[(Int, Int, Int, Int, Int)]): Unit = {
    ods.clear()
    ods ++= arr
    pick.clear()
    drop.clear()
  }

  /**
    * 设置OD集
    *
    * @param arr od集
    */
  def setODS(arr: Array[(Int, Int, Int, Int, Int)]): Unit = {
    ods ++= arr
  }

  /**
    * 获得当前状态下的反馈
    *
    * @param state 当前状态
    * @return
    */
  override def get_reward(state: State): Double = {
    var reward = 0.0
    for (od <- ods) {
      if (pick.isEmpty || !pick.contains(od._1)) {
        if (state.currentState.head == od._2 && state.currentState.last == od._3) {
          pick += od._1
        }
      } else if (pick.contains(od._1) && state.currentState.head == od._4 && state.currentState.last == od._5) {
        reward += 1.0
        pick.remove(pick.indexOf(od._1)) //到站
        drop += od._1
      }
    }
    drop.foreach(t => {
      val index = ods.indexWhere(_._1 == t)
      if (index != -1)
        ods.remove(index)
    })
    reward
  }


  /**
    * 获得当前状态下执行action的反馈
    *
    * @param state  当前状态
    * @param action 动作
    * @return
    */
  override def get_action_reward(state: State, action: Int): Double = {
    val next_state = state_after_action(state, action)
    get_reward(next_state)
  }

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
    for (i <- 0 until vec.length / 2) {
      vec(i * 2) = math.min(math.max(vec(i * 2), 0), width - 1)
      vec(i * 2 + 1) = math.min(math.max(vec(i * 2 + 1), 0), height - 1)
    }
    new State(vec)
  }
}

/**
  * 以state为中心向周边做+1的detect，形成状态
  *
  * Created by kong at 2020/5/18
  */
class MultiODWithDetectEnv(width: Int, height: Int, detect: Int) extends Environment with MultiOD {
  private val actions = Array(Array(0, -1), Array(0, 1), Array(-1, 0), Array(1, 0))
  private val pick = new ArrayBuffer[Int]()
  private val drop = new ArrayBuffer[Int]()
  setInitState(new State(1 + (1 + 2 * detect) * (1 + 2 * detect)))
  setAction(new Action(Array("l", "r", "u", "d")))

  /**
    * 重新开始
    *
    * @param arr Array
    */
  def reset(arr: Array[(Int, Int, Int, Int, Int)]): Unit = {
    ods.clear()
    ods ++= arr
    pick.clear()
    drop.clear()
  }

  /**
    * 设置OD集
    *
    * @param arr od集
    */
  def setODS(arr: Array[(Int, Int, Int, Int, Int)]): Unit = {
    ods ++= arr
  }

  def get_total_reward: Double = drop.length * 1.0

  /**
    * 获得当前状态下的反馈
    *
    * @param state 当前状态
    * @return
    */
  override def get_reward(state: State): Double = {
    var reward = 0.0
    for (od <- ods) {
      if (pick.isEmpty || !pick.contains(od._1)) {
        if (state.currentState.head == od._2 && state.currentState(1) == od._3) {
          pick += od._1
          //          reward += 0.5
        }
      } else if (pick.contains(od._1) && state.currentState.head == od._4 && state.currentState(1) == od._5) {
        reward += 1.0
        pick.remove(pick.indexOf(od._1)) //到站
        drop += od._1
      }
    }
    drop.foreach(t => {
      val index = ods.indexWhere(_._1 == t)
      if (index != -1)
        ods.remove(index)
    })
    reward
  }


  /**
    * 获得当前状态下执行action的反馈
    *
    * @param state  当前状态
    * @param action 动作
    * @return
    */
  override def get_action_reward(state: State, action: Int): Double = {
    val next_state = state_after_action(state, action)
    get_reward(next_state)
  }

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
    val newState = check_boundary(vec)
    val newVec = newState.currentState
    val center = (newVec(0), newVec(1))
    val coords = (-detect to detect).flatMap(t => (-detect to detect).map(s => (t, s))).toArray.filter(_ != (0, 0))
    val neighbor = coords.map(t => {
      val head = center._1 + t._1
      val last = center._2 + t._2
      if (head < 0 || last < 0) //超出边框
        -10
      else if (ods.exists(s => s._2 == head && s._3 == last)) { //o点
        val id = ods.find(s => s._2 == head && s._3 == last).get
        if (pick.contains(id._1))
          0
        else
          -1
      } else if (ods.exists(s => s._4 == head && s._5 == last)) //d点
        +1
      else //空白点
        0
    })
    for (i <- s.length until newVec.length) {
      newVec(i) = neighbor(i - s.length)
    }
    newState
  }

  /**
    * 检查当前状态的合理性，并转换为合理状态
    *
    * @param vec 当前状态
    * @return
    */
  override def check_boundary(vec: Array[Double]): State = {
    vec(0) = math.min(math.max(vec(0), 0), width - 1)
    vec(1) = math.min(math.max(vec(1), 0), height - 1)
    new State(vec)
  }
}

/**
  * 以state为中心向周边做+1的detect，形成状态
  * 增加导航功能，一共4个方向，每个方向有5中状态
  * 编码：当前坐标（2）,侦测范围（（1+detect*2）*（1+detect*2）-1）,导航（4）
  * 即编码长度：5+（1+detect*2）*（1+detect*2）
  * (-1.0, -0.5, 0.0, 0.5, 1.0)全O,有OD偏O,无OD,有OD偏D,全D
  * Created by kong at 2020/5/18
  */
class MultiODWithDetectAndDirectEnv(width: Int, height: Int, detect: Int) extends Environment with MultiOD {
  private val actions = Array(Array(0, -1), Array(0, 1), Array(-1, 0), Array(1, 0))
  private val pick = new ArrayBuffer[Int]()
  private val drop = new ArrayBuffer[Int]()
  setInitState(new State(5 + (1 + 2 * detect) * (1 + 2 * detect))) //编码：当前坐标,侦测范围,导航
  setAction(new Action(Array("l", "r", "u", "d")))

  /**
    * 重新开始
    *
    * @param arr Array
    */
  def reset(arr: Array[(Int, Int, Int, Int, Int)]): Unit = {
    ods.clear()
    ods ++= arr
    pick.clear()
    drop.clear()
  }

  /**
    * 设置OD集
    *
    * @param arr od集
    */
  def setODS(arr: Array[(Int, Int, Int, Int, Int)]): Unit = {
    ods ++= arr
  }

  def get_total_reward: Double = drop.length * 1.0

  /**
    * 获得当前状态下的反馈
    *
    * @param state 当前状态
    * @return
    */
  override def get_reward(state: State): Double = {
    var reward = 0.0
    for (od <- ods) {
      if (pick.isEmpty || !pick.contains(od._1)) {
        if (state.currentState.head == od._2 && state.currentState(1) == od._3) {
          pick += od._1
          //          reward += 0.5
        }
      } else if (pick.contains(od._1) && state.currentState.head == od._4 && state.currentState(1) == od._5) {
        reward += 1.0
        pick.remove(pick.indexOf(od._1)) //到站
        drop += od._1
      }
    }
    drop.foreach(t => {
      val index = ods.indexWhere(_._1 == t)
      if (index != -1)
        ods.remove(index)
    })
    reward
  }


  /**
    * 获得当前状态下执行action的反馈
    *
    * @param state  当前状态
    * @param action 动作
    * @return
    */
  override def get_action_reward(state: State, action: Int): Double = {
    val next_state = state_after_action(state, action)
    get_reward(next_state)
  }

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
    val newState = check_boundary(vec)
    val newVec = newState.currentState
    val center = (newVec(0), newVec(1))
    val coords = (-detect to detect).flatMap(t => (-detect to detect).map(s => (t, s))).toArray.filter(_ != (0, 0))
    val neighbor = coords.map(t => {
      val head = center._1 + t._1
      val last = center._2 + t._2
      if (head < 0 || last < 0) //超出边框
        -10
      else if (ods.exists(s => s._2 == head && s._3 == last)) { //o点
        val id = ods.find(s => s._2 == head && s._3 == last).get
        if (pick.contains(id._1)) //已经过
          0
        else
          -1
      } else if (ods.exists(s => s._4 == head && s._5 == last)) //d点
        +1
      else //空白点
        0
    })
    //侦测信息
    for (i <- s.length until newVec.length) {
      newVec(i) = neighbor(i - s.length)
    }
    //方向信息
    val map = new mutable.HashMap[String, Int]()
    for (od <- ods) {
      val o = (od._2 - center._1, od._3 - center._2)
      val d = (od._4 - center._1, od._5 - center._2)
      if (!pick.contains(od._1))
        if (o._1 <= 0 && o._2 < 0) {
          map.put(s"o-${newVec.length - 4}", map.getOrElse(s"o-${newVec.length - 4}", 0) + 1)
        } else if (o._1 > 0 && o._2 <= 0) {
          map.put(s"o-${newVec.length - 3}", map.getOrElse(s"o-${newVec.length - 3}", 0) + 1)
        } else if (o._1 < 0 && o._2 >= 0) {
          map.put(s"o-${newVec.length - 2}", map.getOrElse(s"o-${newVec.length - 2}", 0) + 1)
        } else if (o._1 >= 0 && o._2 > 0) {
          map.put(s"o-${newVec.length - 1}", map.getOrElse(s"o-${newVec.length - 1}", 0) + 1)
        }

      if (d._1 <= 0 && d._2 < 0) {
        map.put(s"d-${newVec.length - 4}", map.getOrElse(s"d-${newVec.length - 4}", 0) + 1)
      } else if (d._1 > 0 && d._2 <= 0) {
        map.put(s"d-${newVec.length - 3}", map.getOrElse(s"d-${newVec.length - 3}", 0) + 1)
      } else if (d._1 < 0 && d._2 >= 0) {
        map.put(s"d-${newVec.length - 2}", map.getOrElse(s"d-${newVec.length - 2}", 0) + 1)
      } else if (d._1 >= 0 && d._2 > 0) {
        map.put(s"d-${newVec.length - 1}", map.getOrElse(s"d-${newVec.length - 1}", 0) + 1)
      }
    }
    for (i <- newVec.length - 4 until newVec.length) {
      if (map.getOrElse(s"o-$i", 0) > 0 && map.getOrElse(s"d-$i", 0) > 0) {
        newVec(i) = if (map(s"o-$i").toFloat / map(s"d-$i") > 1.0) -0.5 else 0.5
      } else if (map.getOrElse(s"o-$i", 0) > 0 && map.getOrElse(s"d-$i", 0) == 0) {
        newVec(i) = -1.0
      } else if (map.getOrElse(s"o-$i", 0) == 0 && map.getOrElse(s"d-$i", 0) > 0) {
        newVec(i) = 1.0
      } else
        newVec(i) = 0.0
    }

    newState
  }

  /**
    * 检查当前状态的合理性，并转换为合理状态
    *
    * @param vec 当前状态
    * @return
    */
  override def check_boundary(vec: Array[Double]): State = {
    vec(0) = math.min(math.max(vec(0), 0), width - 1)
    vec(1) = math.min(math.max(vec(1), 0), height - 1)
    new State(vec)
  }
}