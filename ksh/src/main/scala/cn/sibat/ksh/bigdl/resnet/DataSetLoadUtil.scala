package cn.sibat.ksh.bigdl.resnet

import java.nio.ByteBuffer
import java.nio.file.{Files, Paths}

import com.intel.analytics.bigdl.dataset.ByteRecord
import com.intel.analytics.bigdl.utils.File

import scala.collection.mutable.ArrayBuffer

/**
  * Created by kong at 2020/7/24
  */
object DataSetLoadUtil {
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
