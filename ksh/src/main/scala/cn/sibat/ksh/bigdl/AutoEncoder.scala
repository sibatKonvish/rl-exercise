package cn.sibat.ksh.bigdl

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.{Linear, ReLU, Reshape, Sequential, Sigmoid}
import com.intel.analytics.bigdl.numeric.NumericFloat

/**
  * Created by kong at 2020/7/20
  */
object AutoEncoder {
  val rowN = 28
  val colN = 28
  val featureSize = rowN * colN

  def apply(classNum: Int): Module[Float] = {
    val model = Sequential[Float]()
      .add(Reshape[Float](Array(featureSize)))
      .add(Linear(featureSize, classNum))
      .add(ReLU[Float]())
      .add(Linear(classNum, featureSize))
      .add(Sigmoid[Float]())
    model
  }
}
