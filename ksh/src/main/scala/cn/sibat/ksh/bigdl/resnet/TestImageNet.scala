package cn.sibat.ksh.bigdl.resnet

import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.optim.{Top1Accuracy, Top5Accuracy}
import com.intel.analytics.bigdl.utils.Engine
import org.apache.spark.SparkContext

/**
  * Created by kong at 2020/7/28
  */
object TestImageNet {

  import DataSetLoadUtil._

  def main(args: Array[String]): Unit = {
    testParser.parse(args, new TestParams()).foreach { param =>
      val conf = Engine.createSparkConf().setAppName("TestImageNet")
      val sc = new SparkContext(conf)
      Engine.init
      val model = Module.loadModule[Float](param.model)
      val evaluationSet = ImageNetDataSetM.testDataSet(param.folder, sc, 224, param.batchSize)
        .toDistributed().data(false)
      val result = model.evaluate(evaluationSet, Array(new Top1Accuracy[Float](), new Top5Accuracy[Float]()))
      result.foreach(r => println(s"${r._2} is ${r._1}"))
      sc.stop()
    }
  }
}
