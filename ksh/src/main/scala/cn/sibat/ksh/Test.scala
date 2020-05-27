package cn.sibat.ksh

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import scala.collection.mutable.ArrayBuffer

object Test {
  def main(args: Array[String]): Unit = {
    val o = Nd4j.zeros(4L, 3L)
    val ones = Nd4j.ones(3L)
    o.putRow(0, ones)
    println(o)
    println(ones)
    o.put(2, 2, 2.0)
    println(o)
  }
}
