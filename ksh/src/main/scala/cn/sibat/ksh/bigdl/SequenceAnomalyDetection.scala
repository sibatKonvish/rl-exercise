package cn.sibat.ksh.bigdl

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim.SGD
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession

/**
  * Created by kong at 2020/7/20
  */
object SequenceAnomalyDetection {
  def main(args: Array[String]): Unit = {
    val conf = Engine.createSparkConf().setAppName("SequenceAnomalyDetection").setMaster("local[*]")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")
    Engine.init
    val inputSize = 6
    val outputSize = 5
    val hiddenSize = 4
    val seqLength = 5

    RNG.setSeed(100)
    val input = Tensor(Array(1, seqLength, inputSize))
    val labels = Tensor(Array(1, seqLength))
    for (i <- 1 to seqLength) {
      val rdmLabel = math.ceil(RNG.uniform(0, 1) * outputSize).toInt
      val rdmInput = math.ceil(RNG.uniform(0, 1) * inputSize).toInt
      input.setValue(1, i, rdmInput, 1.0f)
      labels.setValue(1, i, rdmLabel)
    }

    val rec = Recurrent()

    val model = Sequential[Float]()
      .add(rec.add(LSTM(inputSize, hiddenSize)))
      .add(TimeDistributed(Linear(hiddenSize, outputSize)))

    val criterion = TimeDistributedCriterion(CrossEntropyCriterion(), false)
    val sgd = new SGD(learningRate = 0.1, learningRateDecay = 5e-7, weightDecay = 0.1, momentum = 0.002)
    val params = model.parameters()
    val weight = params._1.head
    val grad = params._2.head
    //    val output = model.forward(input).toTensor
    //    val _loss = criterion.forward(output, labels)
    //    model.zeroGradParameters()
    //    val gradInput = criterion.backward(output, labels)
    //    model.backward(input, gradInput)

    def feval(x: Tensor[Float]): (Float, Tensor[Float]) = {
      val output = model.forward(input).toTensor
      val _loss = criterion.forward(output, labels)
      model.zeroGradParameters()
      val gradInput = criterion.backward(output, labels)
      model.backward(input, gradInput)
      (_loss, grad)
    }

    var loss: Array[Float] = null
    for (i <- 1 to 100) {
      loss = sgd.optimize(feval, weight)._2
      println(s"${i}-th loss = ${loss(0)}")
    }
  }
}
