package cn.sibat.ksh.bigdl.resnet

import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.optim.Top1Accuracy
import com.intel.analytics.bigdl.utils.Engine
import org.apache.spark.SparkContext

/**
  * Created by kong at 2020/7/28
  */
object TestCIFAR10 {

  import DataSetLoadUtil._

  def main(args: Array[String]): Unit = {
    testParser.parse(args, TestParams()).foreach { param =>
      val conf = Engine.createSparkConf().setAppName("TestCIFAR10")
      val sc = new SparkContext(conf)
      Engine.init
      val evaluationSet = Cifar10DataSetM.testDataSet(param.folder, sc, 32, param.batchSize)
        .toDistributed().data(false)
      val model = Module.loadModule[Float](param.model)
      val result = model.evaluate(evaluationSet, Array(new Top1Accuracy[Float]()))
      result.foreach(r => println(s"${r._2} is ${r._1}"))
      sc.stop()
    }
  }
}
