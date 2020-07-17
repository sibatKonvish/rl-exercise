package cn.sibat.ksh.bigdl

import java.nio.ByteBuffer
import java.nio.file.{Files, Paths}

import com.intel.analytics.bigdl.dataset.image.{BytesToGreyImg, GreyImgNormalizer, GreyImgToBatch}
import com.intel.analytics.bigdl.dataset.{ByteRecord, DataSet}
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Linear, LogSoftMax, Reshape, Sequential, SpatialConvolution, SpatialMaxPooling, Tanh, Transformer}
import com.intel.analytics.bigdl.optim.{Loss, Optimizer, SGD, Top1Accuracy, Top5Accuracy, Trigger}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Engine, File}
import org.apache.spark.SparkContext

/**
  * Created by kong at 2020/7/17
  */
object SequenceAnomalyDetection {
  def main(args: Array[String]): Unit = {
    val conf = Engine.createSparkConf().setAppName("SequenceAnomalyDetection")
    val sc = new SparkContext(conf)
    Engine.init
    val model = Sequential[Float]()
    model.add(Reshape(Array(1, 28, 28)))
      .add(SpatialConvolution(1, 6, 5, 5).setName("conv1_5x5"))
      .add(Tanh())
      .add(SpatialMaxPooling(2, 2, 2, 2))
      .add(SpatialConvolution(6, 12, 5, 5).setName("conv2_5x5"))
      .add(Tanh())
      .add(SpatialMaxPooling(2, 2, 2, 2))
      .add(Reshape(Array(12 * 4 * 4)))
      .add(Linear(12 * 4 * 4, 100).setName("fc1"))
      .add(Tanh())
      .add(Linear(100, 10).setName("fc2"))
      .add(LogSoftMax())

    val trainData = "D:/testData/aasData/train-images-idx3-ubyte"
    val trainLabel = "D:/testData/aasData/train-labels-idx1-ubyte"
    val validationData = "D:/testData/aasData/t10k-images-idx3-ubyte"
    val validationLabel = "D:/testData/aasData/t10k-labels-idx1-ubyte"
    val trainMean = 0.13066047740239506
    val trainStd = 0.3081078
    val testMean = 0.13251460696903547
    val testStd = 0.31048024
    val trainSet = DataSet.array(load(trainData, trainLabel)) -> BytesToGreyImg(28, 28) -> GreyImgNormalizer(trainMean, trainStd) -> GreyImgToBatch(10)

    val optimizer = Optimizer(model = model, dataset = trainSet, criterion = ClassNLLCriterion[Float]())
    val validationSet = DataSet.array(load(validationData, validationLabel)) -> BytesToGreyImg(28, 28) -> GreyImgNormalizer(testMean, testStd) -> GreyImgToBatch(10)
    optimizer.setValidation(trigger = Trigger.everyEpoch, dataset = validationSet, vMethods = Array(new Top1Accuracy[Float](), new Top5Accuracy[Float](), new Loss[Float]()))
      .setOptimMethod(new SGD[Float](learningRate = 0.05, learningRateDecay = 0.0))
      .setEndWhen(Trigger.maxEpoch(5))
      .optimize()
  }

  def load(featureFile: String, labelFile: String): Array[ByteRecord] = {
    val featureBuffer = if (featureFile.startsWith("hdfs:")) {
      ByteBuffer.wrap(File.readHdfsByte(featureFile))
    } else
      ByteBuffer.wrap(Files.readAllBytes(Paths.get(featureFile)))

    val labelBuffer = if (featureFile.startsWith("hdfs:"))
      ByteBuffer.wrap(File.readHdfsByte(labelFile))
    else ByteBuffer.wrap(Files.readAllBytes(Paths.get(labelFile)))
    val labelMagicNumber = labelBuffer.getInt
    val featureMagicNumber = featureBuffer.getInt

    val labelCount = labelBuffer.getInt()
    val featureCount = featureBuffer.getInt()

    val rowNum = featureBuffer.getInt()
    val colNum = featureBuffer.getInt()
    val result = new Array[ByteRecord](featureCount)
    var i = 0
    while (i < featureCount) {
      val img = new Array[Byte](rowNum * colNum)
      var y = 0
      while (y < rowNum) {
        var x = 0
        while (x < colNum) {
          img(x + y * colNum) = featureBuffer.get()
          x += 1
        }
        y += 1
      }
      result(i) = ByteRecord(img, labelBuffer.get().toFloat + 1.0f)
      i += 1
    }
    result
  }
}
