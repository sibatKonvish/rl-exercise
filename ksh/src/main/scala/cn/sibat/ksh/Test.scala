package cn.sibat.ksh

import org.nd4j.linalg.factory.Nd4j

object Test {
  def main(args: Array[String]): Unit = {
    val out = Nd4j.create(1, 2 + 7, 13, 13)
    println(out)
  }
}
