package cn.sibat.ksh.bigdl

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, CrossEntropyCriterion, TimeDistributedCriterion}
import com.intel.analytics.bigdl.nn.keras.{Dense, LSTM, Masking, Sequential, TimeDistributed}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.{Optimizer, SGD, Trigger}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, Shape}
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession

/**
  * Created by kong at 2020/7/21
  */
object ClinicalTimeSeries {
  def main(args: Array[String]): Unit = {
    val conf = Engine.createSparkConf().setAppName("ClinicalTimeSeries").setMaster("local[*]")
    val sc = new SparkContext(conf)
    Engine.init
    sc.setLogLevel("OFF")
    val spark = SparkSession.builder().getOrCreate()

    val inputNum = 86
    val epochs = 10
    val learningRate = 0.005
    val batchSize = 32
    val hiddenSize = 200
    val labelNum = 2
    val data = sc.wholeTextFiles("D:/testData/aasData/physionet2012/sequence/*.csv")
    val maxLength = data.reduce((t1, t2) => if (t1._2.split("\n").length > t2._2.split("\n").length) t1 else t2)._2.split("\n").length - 1
    val feature = data.map(t => {
      val id = t._1.split("/").last.replace(".csv", "")
      val lines = t._2.split("\n")
      val arr = lines.tail.flatMap(_.split(",").map(_.toFloat)).toBuffer
      val col = lines.head.split(",").length
      val row = maxLength
      val sor = Tensor(Array(1, row, col))
      for (i <- 1 to row; j <- 1 to col) {
        val value = if (arr.nonEmpty) arr.remove(0) else 0
        sor.setValue(1, i, j, value)
      }
      (id, sor)
    })
    val label = sc.wholeTextFiles("D:/testData/aasData/physionet2012/mortality/*.csv").map(t => {
      val id = t._1.split("/").last.replace(".csv", "")
      val lines = t._2.trim.replace("\n", "").toInt
      //val col = maxLength
      val sor = Tensor(Array(1, labelNum))
      //for (j <- 1 to col) {
      sor.setValue(1, lines + 1, 1.0f)
      //}
      (id, sor)
    })

    val Array(train, test) = feature.join(label).map(t => Sample(t._2._1, t._2._2)).randomSplit(Array(0.8, 0.2))

    val sgd = new SGD[Float](learningRate = learningRate)

    val model = Sequential[Float]()
      .add(Masking(0.0, inputShape = Shape(maxLength, inputNum)))
      .add(LSTM(hiddenSize, inputShape = Shape(maxLength, inputNum)))
      .add(Dense(labelNum, activation = "sigmoid"))
    val criterion = CrossEntropyCriterion[Float]()

    val optimizer = Optimizer(model, train, criterion, 4)
    optimizer.setOptimMethod(sgd).setEndWhen(Trigger.maxEpoch(5)).optimize()
  }
}
