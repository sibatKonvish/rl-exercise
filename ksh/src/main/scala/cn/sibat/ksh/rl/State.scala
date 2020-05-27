package cn.sibat.ksh.rl

import scala.collection.mutable

/**
  * 状态
  *
  * Created by kong at 2020/5/11
  */
class State(dim: Int) {
  private val vector = new Array[Double](dim)

  def this(vec: Array[Double]) {
    this(vec.length)
    for (i <- vec.indices) {
      vector(i) = vec(i)
    }
  }

  def getDimension: Int = dim

  def currentState: Array[Double] = vector

  def clone(vec: Array[Double]): State = {
    new State(vec)
  }


  override def toString: String = {
    s"$dim-${vector.mkString(",")}"
  }


  override def hashCode(): Int = {
    toString.hashCode
  }

  override def equals(obj: Any): Boolean = {
    val other = obj.asInstanceOf[State]
    var t = true
    for (i <- vector.indices) {
      t = t && vector(i) == other.vector(i)
    }
    if (other.getDimension == this.getDimension && t)
      true
    else
      false
  }
}