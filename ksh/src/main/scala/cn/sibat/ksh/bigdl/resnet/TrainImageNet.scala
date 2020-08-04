package cn.sibat.ksh.bigdl.resnet

import cn.sibat.ksh.bigdl.resnet.ResNetM.{DatasetType, ShortcutType}
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.nn.{BatchNormalization, Container, CrossEntropyCriterion, Module}
import com.intel.analytics.bigdl.optim.{OptimMethod, Optimizer, SGD, Top1Accuracy, Top5Accuracy, Trigger}
import com.intel.analytics.bigdl.utils.{Engine, T}
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.apache.spark.SparkContext

/**
  * Created by kong at 2020/7/27
  */
object TrainImageNet {

  import DataSetLoadUtil._

  def imageNetDecay(epoch: Int): Double = {
    if (epoch >= 80)
      3
    else if (epoch >= 60)
      2
    else if (epoch >= 30)
      1
    else
      0.0
  }

  def main(args: Array[String]): Unit = {
    trainParser.parse(args, new TrainParams()).foreach(param => {
      val conf = Engine.createSparkConf().setAppName("TrainImageNet")
      val sc = new SparkContext(conf)
      Engine.init
      val batchSize = param.batchSize
      val (imageSize, dataSetType, maxEpoch, dataset) = (224, DatasetType.ImageNet, param.nEpochs, ImageNetDataSetM)
      val trainDataSet = dataset.trainDataSet(param.folder + "/train", sc, imageSize, batchSize)
      val testDataSet = dataset.testDataSet(param.folder + "/test", sc, imageSize, batchSize)
      val shortcut = ShortcutType.B

      val model = if (param.modelSnapshot.isDefined) {
        Module.loadModule[Float](param.modelSnapshot.get)
      } else {
        val curModel = ResNetM(classNum = param.classes, T("shortcutType" -> shortcut, "depth" -> param.depth, "optnet" -> param.optnet, "dataSet" -> dataSetType))
        if (param.optnet) {
          ResNetM.shareGradInput(curModel)
        }
        ResNetM.modelInit(curModel)
        curModel
      }

      println(model)

      val optimMethod = if (param.stateSnapshot.isDefined) {
        val optim = OptimMethod.load[Float](param.stateSnapshot.get).asInstanceOf[SGD[Float]]
        val baseLr = param.learningRate
        val iterationsPerEpoch = math.ceil(1281167 / param.batchSize).toInt
        val warmUpIteration = iterationsPerEpoch * param.warmupEpoch
        val maxLr = param.maxLr
        val delta = (maxLr - baseLr) / warmUpIteration
        optim.learningRateSchedule = SGD.EpochDecayWithWarmUp(warmUpIteration, delta, imageNetDecay)
        optim
      } else {
        val baseLr = param.learningRate
        val iterationsPerEpoch = math.ceil(1281167 / param.batchSize).toInt
        val warmUpIteration = iterationsPerEpoch * param.warmupEpoch
        val maxLr = param.maxLr
        val delta = (maxLr - baseLr) / warmUpIteration
        new SGD[Float](learningRate = param.learningRate, learningRateDecay = 0.0, weightDecay = param.weightDecay
          , momentum = param.momentum, dampening = param.dampening, nesterov = param.nesterov, learningRateSchedule =
            SGD.EpochDecayWithWarmUp(warmUpIteration, delta, imageNetDecay))
      }

      val optimizer = Optimizer(model, trainDataSet, new CrossEntropyCriterion[Float]())
      if (param.checkpoint.isDefined) {
        optimizer.setCheckpoint(param.checkpoint.get, Trigger.everyEpoch)
      }
      val logdir = "resnet-imagenet"
      val trainSummary = TrainSummary(logdir, sc.applicationId)
      trainSummary.setSummaryTrigger("LearningRate", Trigger.severalIteration(1))
      trainSummary.setSummaryTrigger("Parameters", Trigger.severalIteration(10))
      val validationSummary = ValidationSummary(logdir, appName = sc.applicationId)

      optimizer.setOptimMethod(optimMethod)
        .setValidation(Trigger.everyEpoch, testDataSet, Array(new Top1Accuracy[Float](), new Top5Accuracy[Float]()))
        .setEndWhen(Trigger.maxEpoch(maxEpoch))
        .optimize()
      sc.stop()
    })
  }

  def setParallism(model: AbstractModule[_, _, Float], parallism: Int): Unit = {
    model match {
      case value: BatchNormalization[Float] =>
        value.setParallism(parallism)
      case value: Container[_, _, Float] =>
        value.modules.foreach(sub => setParallism(sub, parallism))
      case _ =>
    }
  }
}
