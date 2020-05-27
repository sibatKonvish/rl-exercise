package cn.sibat.ksh

import org.nd4j.linalg.factory.Nd4j

/**
  *
  *
  * Created by kong at 2020/5/21
  */
class SumTree(capacity: Int) {
  private var write = 0
  private val tree = Nd4j.zeros(2 * capacity - 1)
  private val data = new Array[Any](capacity)

  def propagate(index: Int, change: Double): Unit = {
    val parent = (index - 1) / 2
    tree.putScalar(parent, tree.getDouble(parent.toLong) + change)
    if (parent != 0)
      propagate(parent, change)
  }

  def retrieve(index: Int, s: Double): Int = {
    val left = 2 * index + 1
    val right = left + 1
    if (left >= tree.length())
      return index
    if (s <= tree.getDouble(left.toLong))
      retrieve(left, s)
    else
      retrieve(right, s - tree.getDouble(left.toLong))
  }

  def total(): Double = tree.getDouble(0L)

  def add(p: Double, data: Any): Unit = {
    val index = write + capacity - 1
    this.data(write) = data
    update(index, p)
    write += 1
    if (write >= capacity) {
      write = 0
    }
  }

  def update(index: Int, p: Double): Unit = {
    val change = p - tree.getDouble(index.toLong)
    tree.putScalar(index, p)
    propagate(index, change)
  }

  def get(s: Double): Unit = {
    val index = retrieve(0, s)
    val dataIndex = index - capacity + 1
    (index, tree.getDouble(index.toLong), data(dataIndex))
  }
}


