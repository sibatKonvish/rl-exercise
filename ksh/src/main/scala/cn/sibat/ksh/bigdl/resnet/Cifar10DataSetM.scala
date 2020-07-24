package cn.sibat.ksh.bigdl.resnet

import java.nio.ByteBuffer

import cn.sibat.ksh.bigdl.resnet.Cifar10DataSetM.{trainMean, trainStd}
import com.intel.analytics.bigdl.DataSet
import com.intel.analytics.bigdl.dataset.DataSet.SeqFileFolder.readLabel
import com.intel.analytics.bigdl.dataset.image.{BGRImgNormalizer, BGRImgRdmCropper, BGRImgToBatch, BytesToBGRImg, CropRandom, HFlip}
import com.intel.analytics.bigdl.dataset.{DataSet, MiniBatch, image}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.{ImageFeature, ImageFrame, MTImageFeatureToBatch, MatToTensor, PixelBytesToMat}
import com.intel.analytics.bigdl.transform.vision.image.augmentation.{CenterCrop, ChannelScaledNormalizer, RandomAlterAspect, RandomCropper, RandomResize}
import com.intel.analytics.bigdl.utils.{Engine, T}
import org.apache.hadoop.io.Text
import org.apache.spark.SparkContext

/**
  * 定义resnet的数据集：train，test
  */
trait ResNetDataSet {
  def trainDataSet(path: String, batchSize: Int, size: Int): DataSet[MiniBatch[Float]]

  def testDataSet(path: String, batchSize: Int, size: Int): DataSet[MiniBatch[Float]]

  def trainDataSet(path: String, sc: SparkContext, imageSize: Int, batchSize: Int): DataSet[MiniBatch[Float]]

  def testDataSet(path: String, sc: SparkContext, imageSize: Int, batchSize: Int): DataSet[MiniBatch[Float]]
}

/**
  * Created by kong at 2020/7/24
  */
object Cifar10DataSetM extends ResNetDataSet {
  val trainMean = (0.4913996898739353, 0.4821584196221302, 0.44653092422369434)
  val trainStd = (0.24703223517429462, 0.2434851308749409, 0.26158784442034005)
  val testMean = (0.4942142913295297, 0.4851314002725445, 0.45040910258647154)
  val testStd = (0.2466525177466614, 0.2428922662655766, 0.26159238066790275)

  override def trainDataSet(path: String, batchSize: Int, size: Int): DataSet[MiniBatch[Float]] = {
    //加载数据->转BGR图片->归一化->按概率翻转图片->随机裁剪32*32的图片->转批次数据
    DataSet.array(DataSetLoadUtil.loadTrain(path)) ->
      BytesToBGRImg() ->
      BGRImgNormalizer(trainMean, trainStd) ->
      HFlip(0.5) ->
      BGRImgRdmCropper(cropWidth = 32, cropHeight = 32, padding = 4) ->
      BGRImgToBatch(batchSize)
  }

  override def testDataSet(path: String, batchSize: Int, size: Int): DataSet[MiniBatch[Float]] = {
    //加载数据->转BGR图片->归一化->转批次数据
    DataSet.array(DataSetLoadUtil.loadTest(path)) ->
      BytesToBGRImg() ->
      BGRImgNormalizer(testMean, testStd) ->
      BGRImgToBatch(batchSize)
  }

  override def trainDataSet(path: String, sc: SparkContext, imageSize: Int, batchSize: Int): DataSet[MiniBatch[Float]] = {
    //加载数据->转BGR图片->归一化->按概率翻转图片->随机裁剪32*32的图片->转批次数据
    DataSet.array(DataSetLoadUtil.loadTrain(path), sc) ->
      BytesToBGRImg() ->
      BGRImgNormalizer(trainMean, trainStd) ->
      HFlip(0.5) ->
      BGRImgRdmCropper(cropWidth = 32, cropHeight = 32, padding = 4) ->
      BGRImgToBatch(batchSize)
  }

  override def testDataSet(path: String, sc: SparkContext, imageSize: Int, batchSize: Int): DataSet[MiniBatch[Float]] = {
    //加载数据->转BGR图片->归一化->转批次数据
    DataSet.array(DataSetLoadUtil.loadTest(path), sc) ->
      BytesToBGRImg() ->
      BGRImgNormalizer(testMean, testStd) ->
      BGRImgToBatch(batchSize)
  }
}

object ImageNetDataSetM extends ResNetDataSet {
  val trainMean = (0.485, 0.456, 0.406)
  val trainStd = (0.229, 0.224, 0.225)
  val testMean = trainMean
  val testStd = trainStd

  override def trainDataSet(path: String, batchSize: Int, size: Int): DataSet[MiniBatch[Float]] = {
    //加载数据->转BGR图片->归一化->按概率翻转图片->随机裁剪32*32的图片->转批次数据
    DataSet.array(DataSetLoadUtil.loadTrain(path)) ->
      BytesToBGRImg() ->
      BGRImgNormalizer(trainMean, trainStd) ->
      HFlip(0.5) ->
      BGRImgRdmCropper(cropWidth = 32, cropHeight = 32, padding = 4) ->
      BGRImgToBatch(batchSize)
  }

  override def testDataSet(path: String, batchSize: Int, size: Int): DataSet[MiniBatch[Float]] = {
    //加载数据->转BGR图片->归一化->转批次数据
    DataSet.array(DataSetLoadUtil.loadTrain(path)) ->
      BytesToBGRImg() ->
      BGRImgNormalizer(testMean, testStd) ->
      BGRImgToBatch(batchSize)
  }

  override def trainDataSet(path: String, sc: SparkContext, imageSize: Int, batchSize: Int): DataSet[MiniBatch[Float]] = {
    DataSet.rdd(filesToImageFrame(path, sc, 1000).toDistributed().rdd)
      .transform(
        MTImageFeatureToBatch(
          width = imageSize,
          height = imageSize,
          batchSize = batchSize,
          transformer = PixelBytesToMat() ->
            RandomAlterAspect() ->
            RandomCropper(224, 224, true, CropRandom) ->
            ChannelScaledNormalizer(104, 117, 123, 0.0078125) ->
            MatToTensor[Float](), toRGB = false
        )
      )
  }

  override def testDataSet(path: String, sc: SparkContext, imageSize: Int, batchSize: Int): DataSet[MiniBatch[Float]] = {
    DataSet.rdd(filesToImageFrame(path, sc, 1000).toDistributed().rdd)
      .transform(
        MTImageFeatureToBatch(
          width = imageSize,
          height = imageSize,
          batchSize = batchSize,
          transformer = PixelBytesToMat() ->
            RandomResize(256, 256) ->
            CenterCrop(224, 224) ->
            ChannelScaledNormalizer(104, 117, 123, 0.0078125) ->
            MatToTensor[Float](), toRGB = false
        )
      )
  }

  def filesToImageFrame(url: String, sc: SparkContext,
                        classNum: Int, partitionNum: Option[Int] = None): ImageFrame = {
    val num = 10
    val rawData = sc.sequenceFile(url, classOf[Text], classOf[Text], num).map(image => {
      val rawBytes = image._2.copyBytes()
      val label = Tensor[Float](T(readLabel(image._1).toFloat))
      val imgBuffer = ByteBuffer.wrap(rawBytes)
      val width = imgBuffer.getInt
      val height = imgBuffer.getInt
      val bytes = new Array[Byte](3 * width * height)
      System.arraycopy(imgBuffer.array(), 8, bytes, 0, bytes.length)
      val imf = ImageFeature(bytes, label)
      imf(ImageFeature.originalSize) = (height, width, 3)
      imf
    }).filter(_[Tensor[Float]](ImageFeature.label).valueAt(1) <= classNum)
    ImageFrame.rdd(rawData)
  }
}
