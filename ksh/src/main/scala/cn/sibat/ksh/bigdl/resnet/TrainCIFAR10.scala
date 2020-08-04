package cn.sibat.ksh.bigdl.resnet

import cn.sibat.ksh.bigdl.resnet.ResNetM.{DatasetType, ShortcutType}
import com.intel.analytics.bigdl.nn.{CrossEntropyCriterion, Module}
import com.intel.analytics.bigdl.optim.{OptimMethod, Optimizer, SGD, Top1Accuracy, Trigger}
import com.intel.analytics.bigdl.utils.{Engine, T}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.apache.spark.SparkContext

/**
  * Created by kong at 2020/7/28
  */
object TrainCIFAR10 {

  import DataSetLoadUtil._

  def cifar10Decay(epoch: Int): Double = if (epoch >= 122) 2.0 else if (epoch >= 81) 1.0 else 0.0

  def main(args: Array[String]): Unit = {
    trainParser.parse(args, TrainParams()).foreach(param => {
      val conf = Engine.createSparkConf().setAppName("TrainCIFAR10").setMaster("local[*]")
      val sc = new SparkContext(conf)
      Engine.init
      val batchSize = param.batchSize
      val (imageSize, lrSchedule, maxEpoch, dataset) = (32, DatasetType.CIFAR10, param.nEpochs, Cifar10DataSetM)
      val trainDataSet = dataset.trainDataSet(param.folder, sc, imageSize, batchSize)
      val testDataSet = dataset.testDataSet(param.folder, sc, imageSize, batchSize)
      val shortcut = param.shortcutType match {
        case "A" => ShortcutType.A
        case "B" => ShortcutType.B
        case _ => ShortcutType.C
      }

      val model = if (param.modelSnapshot.isDefined) {
        Module.loadModule[Float](param.modelSnapshot.get)
      } else {
        val curModel = if (param.graphModel) {
          ResNetM.graph(param.classes, T("shortcutType" -> shortcut, "depth" -> param.depth, "optnet" -> param.optnet))
        } else {
          ResNetM(param.classes, T("shortcutType" -> shortcut, "depth" -> param.depth, "optnet" -> param.optnet))
        }
        if (param.optnet) {
          ResNetM.shareGradInput(curModel)
        }
        ResNetM.modelInit(curModel)
        curModel
      }
      val optimMethod = if (param.stateSnapshot.isDefined) {
        OptimMethod.load[Float](param.stateSnapshot.get)
      } else {
        new SGD[Float](learningRate = param.learningRate, learningRateDecay = 0.0, weightDecay = param.weightDecay
          , momentum = param.momentum, dampening = param.dampening, nesterov = param.nesterov, learningRateSchedule = SGD.EpochDecay(cifar10Decay))
      }

      println(model)
      val optimizer = Optimizer(model, trainDataSet, new CrossEntropyCriterion[Float]())

      if (param.checkpoint.isDefined) {
        optimizer.setCheckpoint(param.checkpoint.get, Trigger.everyEpoch)
      }
      //      optimizer.setOptimMethod(optimMethod)
      //        .setValidation(Trigger.everyEpoch, testDataSet, Array(new Top1Accuracy[Float]()))
      //        .setEndWhen(Trigger.maxEpoch(maxEpoch))
      //        .optimize()
      sc.stop()
    })
  }
}
