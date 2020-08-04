package cn.sibat.ksh.bigdl.resnet

import java.nio.ByteBuffer
import java.nio.file.{Files, Paths}

import com.intel.analytics.bigdl.dataset.ByteRecord
import com.intel.analytics.bigdl.utils.File
import scopt.OptionParser

import scala.collection.mutable.ArrayBuffer

case class TestParams(folder: String = "./", model: String = "", batchSize: Int = 128)

case class TrainParams(
                        folder: String = "./",
                        checkpoint: Option[String] = None,
                        modelSnapshot: Option[String] = None,
                        stateSnapshot: Option[String] = None,
                        optnet: Boolean = false,
                        depth: Int = 20,
                        classes: Int = 10,
                        shortcutType: String = "A",
                        batchSize: Int = 128,
                        nEpochs: Int = 165,
                        learningRate: Double = 0.1,
                        weightDecay: Double = 1e-4,
                        momentum: Double = 0.9,
                        dampening: Double = 0.0,
                        nesterov: Boolean = true,
                        graphModel: Boolean = false,
                        warmupEpoch: Int = 0,
                        maxLr: Double = 0.0
                      )

/**
  * Created by kong at 2020/7/24
  */
object DataSetLoadUtil {

  val trainParser = new OptionParser[TrainParams]("train") {
    head("Train ResNet model on single node")
    opt[String]('f', "folder")
      .text("where you put your training files")
      .action((x, c) => c.copy(folder = x))
    opt[String]("model")
      .text("model snapshot location")
      .action((x, c) => c.copy(modelSnapshot = Some(x)))
    opt[String]("state")
      .text("state snapshot location")
      .action((x, c) => c.copy(stateSnapshot = Some(x)))
    opt[String]("cache")
      .text("where to cache the model")
      .action((x, c) => c.copy(checkpoint = Some(x)))
    opt[Boolean]("optnet")
      .text("shared gradients and caches to reduce memory usage")
      .action((x, c) => c.copy(optnet = x))
    opt[Int]("depth")
      .text("depth of ResNet, 18 | 20 | 34 | 50 | 101 | 152 | 200")
      .action((x, c) => c.copy(depth = x))
    opt[Int]("classes")
      .text("classes of ResNet")
      .action((x, c) => c.copy(classes = x))
    opt[String]("shortcutType")
      .text("shortcutType of ResNet, A | B | C")
      .action((x, c) => c.copy(shortcutType = x))
    opt[Int]("batchSize")
      .text("batchSize of ResNet,  64 | 128 | 256 | ..")
      .action((x, c) => c.copy(batchSize = x))
    opt[Int]("nEpochs")
      .text("number of epochs of ResNet; default is 165")
      .action((x, c) => c.copy(nEpochs = x))
    opt[Double]("learningRate")
      .text("initial learning rate of ResNet; default is 0.1")
      .action((x, c) => c.copy(learningRate = x))
    opt[Double]("momentum")
      .text("momentum of ResNet; default is 0.9")
      .action((x, c) => c.copy(momentum = x))
    opt[Double]("weightDecay")
      .text("weightDecay of ResNet; default is 1e-4")
      .action((x, c) => c.copy(weightDecay = x))
    opt[Double]("dampening")
      .text("dampening of ResNet; default is 0.0")
      .action((x, c) => c.copy(dampening = x))
    opt[Boolean]("nesterov")
      .text("nesterov of ResNet; default is trye")
      .action((x, c) => c.copy(nesterov = x))
    opt[Unit]('g', "graphModel")
      .text("use graph model")
      .action((x, c) => c.copy(graphModel = true))
    opt[Int]("warmupEpoch")
      .text("warnum epoch")
      .action((x, c) => c.copy(warmupEpoch = x))
    opt[Double]("maxLr")
      .text("maxLr")
      .action((x, c) => c.copy(maxLr = x))
  }

  val testParser = new OptionParser[TestParams]("test") {
    opt[String]('f', "folder")
      .text("the location of cifar10 dataset")
      .action((x, c) => c.copy(folder = x))
    opt[String]('m', "model")
      .text("the location of model snapshot")
      .action((x, c) => c.copy(model = x))
      .required()
    opt[Int]('b', "batchSize")
      .text("batch size")
      .action((x, c) => c.copy(batchSize = x))
  }

  /**
    * 读取训练集
    *
    * @param dataFile 数据目录
    * @return
    */
  def loadTrain(dataFile: String): Array[ByteRecord] = {
    val allFiles = (1 to 5).map(i => s"$dataFile/data_batch_$i.bin")
    val result = new ArrayBuffer[ByteRecord]()
    allFiles.foreach(load(_, result))
    result.toArray
  }

  /**
    * 加载测试数据
    *
    * @param dataFile 数据目录
    * @return
    */
  def loadTest(dataFile: String): Array[ByteRecord] = {
    val result = new ArrayBuffer[ByteRecord]()
    val testFile = dataFile + "/test_batch.bin"
    load(testFile, result)
    result.toArray
  }

  /**
    * 加载cifar数据，可为hdfs文件
    *
    * @param filePath 地址
    * @param records  存储
    */
  def load(filePath: String, records: ArrayBuffer[ByteRecord]): Unit = {
    val rowNum = 32
    val colNum = 32
    val imageOffset = rowNum * colNum * 3 + 1
    val channelOffset = rowNum * colNum
    val bufferOffset = 8

    val featureBuffer = if (filePath.startsWith("hdfs:"))
      ByteBuffer.wrap(File.readHdfsByte(filePath))
    else ByteBuffer.wrap(Files.readAllBytes(Paths.get(filePath)))

    val featureArray = featureBuffer.array()
    val featureCount = featureArray.length / imageOffset
    var i = 0
    while (i < featureCount) {
      val img = new Array[Byte](rowNum * colNum * 3 + bufferOffset)
      val byteBuffer = ByteBuffer.wrap(img)
      byteBuffer.putInt(rowNum)
      byteBuffer.putInt(colNum)

      val label = featureArray(i * imageOffset).toFloat
      var y = 0
      val start = i * imageOffset + 1
      while (y < rowNum) {
        var x = 0
        while (x < colNum) {
          img((x + y * colNum) * 3 + 2 + bufferOffset) = featureArray(start + x + y * colNum)
          img((x + y * colNum) * 3 + 1 + bufferOffset) = featureArray(start + x + y * colNum + channelOffset)
          img((x + y * colNum) * 3 + bufferOffset) = featureArray(start + x + y * colNum + 2 * channelOffset)
          x += 1
        }
        y += 1
      }
      records += ByteRecord(img, label + 1.0f)
      i += 1
    }
  }
}
