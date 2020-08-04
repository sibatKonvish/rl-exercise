package cn.sibat.ksh.bigdl

import com.intel.analytics.bigdl.models.resnet.Convolution
import com.intel.analytics.bigdl.nn.{BCECriterion, Sequential}

/**
  * Created by kong at 2020/7/23
  */
object KeyPointModel {
  def main(args: Array[String]): Unit = {


    val criterion = BCECriterion[Float]()
  }

  def create_yolov3(): Unit = {
    val model = Sequential[Float]()
      .add(Convolution[Float](1,32,3,3,strideH = 2,strideW = 2))
  }
}
