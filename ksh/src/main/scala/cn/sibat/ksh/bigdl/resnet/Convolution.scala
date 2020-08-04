package cn.sibat.ksh.bigdl.resnet

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.nn.{CAddTable, Concat, ConcatTable, Container, Graph, Identity, Input, JoinTable, Linear, MsraFiller, MulConstant, Ones, RandomNormal, ReLU, Sequential, SpatialAveragePooling, SpatialBatchNormalization, SpatialConvolution, SpatialMaxPooling, SpatialShareConvolution, View, Zeros}
import com.intel.analytics.bigdl.optim.L2Regularizer
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag
import scala.collection.mutable

/**
  * Created by kong at 2020/7/24
  */
object Convolution {
  /**
    * 创建卷积网络
    *
    * @param nInputPlane   输入维度
    * @param nOutputPlane  输出维度
    * @param kernelW       核函数的宽
    * @param kernelH       核函数的高
    * @param strideW       滑块的宽
    * @param strideH       滑块的高
    * @param padW          补充的宽
    * @param padH          补充的高
    * @param nGroup        Kernel group number
    * @param propagateBack propagate gradient back
    * @param optnet        是否为分享型卷积
    * @param weightDecay   参数衰减
    * @param ev            协变
    * @tparam T 泛型
    * @return
    */
  def apply[@specialized(Float, Double) T: ClassTag](
                                                      nInputPlane: Int,
                                                      nOutputPlane: Int,
                                                      kernelW: Int,
                                                      kernelH: Int,
                                                      strideW: Int = 1,
                                                      strideH: Int = 1,
                                                      padW: Int = 0,
                                                      padH: Int = 0,
                                                      nGroup: Int = 1,
                                                      propagateBack: Boolean = true,
                                                      optnet: Boolean = true,
                                                      weightDecay: Double = 1e-4
                                                    )(implicit ev: TensorNumeric[T]): SpatialConvolution[T] = {
    val wReg = L2Regularizer[T](weightDecay)
    val bReg = L2Regularizer[T](weightDecay)
    val conv = if (optnet) {
      SpatialShareConvolution[T](nInputPlane, nOutputPlane, kernelW, kernelH, strideW, strideH, padW, padH, nGroup, propagateBack, wReg, bReg)
    } else {
      SpatialConvolution[T](nInputPlane, nOutputPlane, kernelW, kernelH, strideW, strideH, padW, padH, nGroup, propagateBack, wReg, bReg)
    }
    //初始化参数，w->高斯分布,b->bias
    conv.setInitMethod(MsraFiller(false), Zeros)
    conv
  }
}

object Sbn {
  /**
    * 归一化
    *
    * @param nOuput   输出
    * @param eps      eps
    * @param momentum momentum
    * @param affine   affine
    * @param en       协变
    * @tparam T 泛型
    * @return
    */
  def apply[@specialized(Float, Double) T: ClassTag](
                                                      nOuput: Int,
                                                      eps: Double = 1e-3,
                                                      momentum: Double = 0.1,
                                                      affine: Boolean = true
                                                    )(implicit en: TensorNumeric[T]): SpatialBatchNormalization[T]
  = SpatialBatchNormalization[T](nOuput, eps, momentum, affine).setInitMethod(Ones, Zeros)
}

object ResNetM {
  var iChannels = 0

  def shareGradInput(model: Module[Float]): Unit = {
    def sharingKey(m: Module[Float]) = m.getClass.getName

    val cache = mutable.Map[Any, Storage[Float]]()
    val packageName = model.getName().stripSuffix("Sequential")
    cache.put("fInput", Storage(Array(1.0f)))
    cache.put("fGradInput", Storage(Array(1.0f)))

    var index = 0

    def matchModels(model: Module[Float]): Unit = {
      model match {
        case container: Container[Activity, Activity, Float] =>
          container.modules.foreach(m => {
            if (m.gradInput.isInstanceOf[Tensor[_]] && !m.getClass.getName.equals(packageName + "ConcatTable")) {
              val key = sharingKey(m)
              if (!cache.contains(key)) {
                cache.put(key, Storage(Array(1.0f)))
              }
              m.gradInput = Tensor(cache(key), 1, Array(0))
            }
            matchModels(m)
          })
        case concatTable if concatTable.isInstanceOf[ConcatTable[Float]] =>
          if (!cache.contains(index % 2)) {
            cache.put(index % 2, Storage(Array(1.0f)))
          }
          concatTable.gradInput = Tensor[Float](cache(index % 2), 1, Array(0))
          index += 1
        case spatialShareConvolution if spatialShareConvolution.isInstanceOf[SpatialShareConvolution[Float]] =>
          val curModel = spatialShareConvolution.asInstanceOf[SpatialShareConvolution[Float]]
          curModel.fInput = Tensor[Float](cache("fInput"))
          curModel.fGradInput = Tensor[Float](cache("fGradInput"))
        case _ => Unit
      }
    }

    matchModels(model)
  }

  def modelInit(model: Module[Float]): Unit = {
    def initModules(model: Module[Float]): Unit = {
      model match {
        case container: Container[Activity, Activity, Float] => container.modules.foreach(m => initModules(m))
        case spatialShareConvolution if spatialShareConvolution.isInstanceOf[SpatialShareConvolution[Float]] =>
          val curModel = spatialShareConvolution.asInstanceOf[SpatialShareConvolution[Float]]
          val n: Float = curModel.kernelW * curModel.kernelH * curModel.nOutputPlane
          curModel.weight.apply1(_ => RNG.normal(0, math.sqrt(2.0f / n)).toFloat)
          curModel.bias.apply1(_ => 0.0f)
        case spatialConvolution if spatialConvolution.isInstanceOf[SpatialConvolution[Float]] =>
          val curModel = spatialConvolution.asInstanceOf[SpatialConvolution[Float]]
          val n = curModel.kernelW * curModel.kernelH * curModel.nOutputPlane
          curModel.weight.apply1(_ => RNG.normal(0, math.sqrt(2.0f / n)).toFloat)
          curModel.bias.apply1(_ => 0.0f)
        case spatialBatchNormalization if spatialBatchNormalization.isInstanceOf[SpatialBatchNormalization[Float]] =>
          val curModel = spatialBatchNormalization.asInstanceOf[SpatialBatchNormalization[Float]]
          curModel.weight.apply1(_ => 1.0f)
          curModel.bias.apply1(_ => 0.0f)
        case linear if linear.isInstanceOf[Linear[Float]] =>
          linear.asInstanceOf[Linear[Float]].bias.apply1(_ => 0.0f)
        case _ => Unit
      }
    }

    initModules(model)
  }

  def apply(classNum: Int, opt: Table): Module[Float] = {
    val depth = opt.get("depth").getOrElse(18)
    val shortcutType = opt.get("shortcutType").getOrElse(ShortcutType.B).asInstanceOf[ShortcutType]
    val dataSet = opt.getOrElse[DatasetType]("dataSet", DatasetType.CIFAR10)
    val optnet = opt.getOrElse("optnet", true)

    def shortcut(nInputPlane: Int, nOutputPlane: Int, stride: Int): Module[Float] = {
      val useConv = shortcutType == ShortcutType.C || (shortcutType == ShortcutType.B && nInputPlane != nOutputPlane)
      if (useConv) {
        Sequential[Float]()
          .add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride, optnet = optnet))
          .add(Sbn[Float](nOutputPlane))
      } else if (nInputPlane != nOutputPlane) {
        Sequential[Float]()
          .add(SpatialAveragePooling(1, 1, stride, stride))
          .add(Concat[Float](2)
            .add(Identity())
            .add(MulConstant(0.0f)))
      } else {
        Identity()
      }
    }

    def basicBlock(n: Int, stride: Int): Module[Float] = {
      val nInputPlane = iChannels
      iChannels = n

      val s = Sequential[Float]()
      s.add(Convolution[Float](nInputPlane, n, 3, 3, stride, stride, 1, 1, optnet = optnet))
        .add(Sbn[Float](n))
        .add(ReLU[Float](ip = true))
        .add(Convolution[Float](n, n, 3, 3, padH = 1, padW = 1, optnet = optnet))
        .add(Sbn[Float](n))
      Sequential[Float]()
        .add(ConcatTable()
          .add(s)
          .add(shortcut(nInputPlane, n, stride)))
        .add(CAddTable(inplace = true))
        .add(ReLU(ip = true))
    }

    def bottleneck(n: Int, stride: Int): Module[Float] = {
      val nInputPlane = iChannels
      iChannels = n * 4
      val s = Sequential[Float]()
        .add(Convolution(nInputPlane, n, 1, 1, optnet = optnet))
        .add(Sbn(n))
        .add(ReLU(ip = true))
        .add(Convolution(n, n * 4, 1, 1, optnet = optnet))
        .add(Sbn(n * 4).setInitMethod(Zeros, Zeros))
      Sequential[Float]()
        .add(ConcatTable()
          .add(s)
          .add(shortcut(nInputPlane, n * 4, stride)))
        .add(CAddTable(inplace = true))
        .add(ReLU[Float](ip = true))
    }

    def layer(block: (Int, Int) => Module[Float], features: Int, count: Int, stride: Int = 1): Module[Float] = {
      val s = Sequential[Float]()
      for (i <- 1 to count) {
        s.add(block(features, if (i == 1) stride else 1))
      }
      s
    }

    val model = Sequential[Float]()
    if (dataSet == DatasetType.ImageNet) {
      val cfg = Map(18 -> ((2, 2, 2, 2), 512, basicBlock: (Int, Int) => Module[Float]),
        34 -> ((3, 4, 6, 3), 512, basicBlock: (Int, Int) => Module[Float]),
        50 -> ((3, 4, 6, 3), 2048, bottleneck: (Int, Int) => Module[Float]),
        101 -> ((3, 4, 23, 3), 2048, bottleneck: (Int, Int) => Module[Float]),
        152 -> ((3, 8, 36, 3), 2048, bottleneck: (Int, Int) => Module[Float]),
        200 -> ((3, 24, 36, 3), 2048, bottleneck: (Int, Int) => Module[Float])
      )
      require(cfg.keySet.contains(depth), s"Invalid depth $depth")
      val (loopConfig, nFeatures, block) = cfg(depth)
      iChannels = 64
      model.add(Convolution(3, 64, 7, 7, 2, 2, 3, 3, optnet = optnet, propagateBack = false)) //先做7*7卷积
        .add(Sbn(64))
        .add(ReLU(ip = true))
        .add(SpatialMaxPooling(3, 3, 2, 2, 1, 1))
        .add(layer(block, 64, loopConfig._1)) //3个64通道的残差单元
        .add(layer(block, 128, loopConfig._2, 2)) //4个128通道的残差单元
        .add(layer(block, 256, loopConfig._3, 2)) //6个156通道的残差单元
        .add(layer(block, 512, loopConfig._4, 2)) //3个512通道的残差单元
        .add(SpatialAveragePooling(7, 7, 1, 1))
        .add(View(nFeatures).setNumInputDims(3))
        .add(Linear[Float](nFeatures, classNum, withBias = true, L2Regularizer(1e-4), L2Regularizer(1e-4)).setInitMethod(RandomNormal(0.0, 0.01), Zeros))
    } else if (dataSet == DatasetType.CIFAR10) {
      require((depth - 2) % 6 == 0, "depth should be one of 20,32,44,56,110,1202")
      val n = (depth - 2) / 6
      iChannels = 16
      model.add(Convolution(3, 16, 3, 3, padH = 1, padW = 1, optnet = optnet, propagateBack = false))
        .add(SpatialBatchNormalization(16))
        .add(ReLU(ip = true))
        .add(layer(basicBlock, 16, n))
        .add(layer(basicBlock, 32, n, 2))
        .add(layer(basicBlock, 64, n, 2))
        .add(SpatialAveragePooling(8, 8, 1, 1))
        .add(View(64).setNumInputDims(3))
        .add(Linear[Float](64, classNum))
    } else {
      throw new IllegalArgumentException(s"Invalid dataset $dataSet")
    }
    model
  }

  def graph(classNum: Int, opt: Table): Module[Float] = {
    val depth = opt.getOrElse("depth", 18)
    val shortcutType = opt.getOrElse("shortcutType", ShortcutType.B).asInstanceOf[ShortcutType]
    val dataSet = opt.getOrElse("dataset", DatasetType.CIFAR10).asInstanceOf[DatasetType]
    val optnet = opt.getOrElse("optnet", true)

    def shortcutFunc(nInputPlane: Int, nOutputPlane: Int, stride: Int, input: ModuleNode[Float]): ModuleNode[Float] = {
      val useConv = shortcutType == ShortcutType.C || (shortcutType == ShortcutType.B && nInputPlane != nOutputPlane)
      if (useConv) {
        val conv1 = Convolution[Float](nInputPlane, nOutputPlane, 1, 1, stride, stride, optnet = optnet).inputs(input)
        val bn1 = Sbn(nOutputPlane).inputs(conv1)
        bn1
      } else if (nInputPlane != nOutputPlane) {
        val pool1 = SpatialAveragePooling(1, 1, stride, stride).inputs(input)
        val mul1 = MulConstant(0.0f).inputs(pool1)
        val concat = JoinTable(2, 0).inputs(pool1, mul1)
        concat
      } else {
        input
      }
    }

    def basicBlockFunc(n: Int, stride: Int, input: ModuleNode[Float]): ModuleNode[Float] = {
      val nInputPlane = iChannels
      iChannels = n
      val conv1 = Convolution(nInputPlane, n, 3, 3, stride, stride, padH = 1, padW = 1).inputs(input)
      val bn1 = Sbn[Float](n).inputs(conv1)
      val relu1 = ReLU[Float](ip = true).inputs(bn1)
      val conv2 = Convolution(n, n, 3, 3, padH = 1, padW = 1).inputs(relu1)
      val bn2 = Sbn(n).inputs(conv2)
      val shortcut = shortcutFunc(nInputPlane, n, stride, input)
      val add = CAddTable(inplace = true).inputs(bn2, shortcut)
      val output = ReLU(ip = true).inputs(add)
      output
    }

    def bottleneckFunc(n: Int, stride: Int, input: ModuleNode[Float]): ModuleNode[Float] = {
      val nInputPlane = iChannels
      iChannels = n * 4
      val conv1 = Convolution(nInputPlane, n, 1, 1, optnet = optnet).inputs(input)
      val bn1 = Sbn(n).inputs(conv1)
      val relu1 = ReLU(ip = true).inputs(bn1)
      val conv2 = Convolution(n, n, 3, 3, stride, stride, padH = 1, padW = 1, optnet = optnet).inputs(relu1)
      val bn2 = Sbn(n).inputs(conv2)
      val relu2 = ReLU(ip = true).inputs(bn2)
      val conv3 = Convolution(n, n * 4, 1, 1, optnet = optnet).inputs(relu2)
      val sbn = Sbn(n * 4).setInitMethod(Zeros, Zeros).inputs(conv3)
      val shortcut = shortcutFunc(nInputPlane, n * 4, stride, input)
      val add = CAddTable(inplace = true).inputs(sbn, shortcut)
      val output = ReLU(ip = true).inputs(add)
      output
    }

    def layer(block: (Int, Int, ModuleNode[Float]) => ModuleNode[Float], features: Int, count: Int, stride: Int = 1)(input: ModuleNode[Float]): ModuleNode[Float] = {
      var output = block(features, stride, input)
      for (_ <- 2 to count) {
        output = block(features, 1, output)
      }
      output
    }

    val model = if (dataSet == DatasetType.ImageNet) {
      val cfg = Map(
        18 -> ((2, 2, 2, 2), 512, basicBlockFunc: (Int, Int, ModuleNode[Float]) => ModuleNode[Float]),
        34 -> ((3, 4, 6, 3), 512, basicBlockFunc: (Int, Int, ModuleNode[Float]) => ModuleNode[Float]),
        50 -> ((3, 4, 6, 3), 2048, bottleneckFunc: (Int, Int, ModuleNode[Float]) => ModuleNode[Float]),
        101 -> ((3, 4, 23, 3), 2048, bottleneckFunc: (Int, Int, ModuleNode[Float]) => ModuleNode[Float]),
        152 -> ((3, 8, 36, 3), 2048, bottleneckFunc: (Int, Int, ModuleNode[Float]) => ModuleNode[Float]),
        200 -> ((3, 24, 36, 3), 2048, bottleneckFunc: (Int, Int, ModuleNode[Float]) => ModuleNode[Float])
      )

      require(cfg.keySet.contains(depth), s"Invalid depth $depth")
      val (loopConfig, nFeatures, block) = cfg(depth)
      iChannels = 64
      val input = Input[Float]()
      val conv1 = Convolution[Float](3, 64, 7, 7, 2, 2, 3, 3, optnet = optnet, propagateBack = false).inputs(input)
      val bn = Sbn[Float](64).inputs(conv1)
      val relu = ReLU[Float](ip = true).inputs(bn)
      val pool = SpatialMaxPooling[Float](3, 3, 2, 2, 1, 1).inputs(relu)
      val layer1 = layer(block, 64, loopConfig._1)(pool)
      val layer2 = layer(block, 128, loopConfig._2, 2)(layer1)
      val layer3 = layer(block, 256, loopConfig._3, 2)(layer2)
      val layer4 = layer(block, 512, loopConfig._4, 2)(layer3)
      val pool2 = SpatialAveragePooling(7, 7, 1, 1).inputs(layer4)
      val view = View(nFeatures).setNumInputDims(3).inputs(pool2)
      val output = Linear[Float](nFeatures, classNum, true, L2Regularizer(1e-4), L2Regularizer(1e-4)).setInitMethod(RandomNormal(0.0, 0.01), Zeros).inputs(view)
      Graph(input, output)
    } else if (dataSet == DatasetType.CIFAR10) {
      require((depth - 2) % 6 == 0, "depth should be one of 20, 32, 44, 56, 110, 1202")
      val n = (depth - 2) / 6
      iChannels = 16
      val input = Input[Float]()
      val conv1 = Convolution(3, 16, 3, 3, 1, 1, 1, 1, optnet = optnet, propagateBack = false).inputs(input)
      val bn = SpatialBatchNormalization(16).inputs(conv1)
      val relu = ReLU(ip = true).inputs(bn)
      val layer1 = layer(basicBlockFunc, 16, n)(relu)
      val layer2 = layer(basicBlockFunc, 32, n, 2)(layer1)
      val layer3 = layer(basicBlockFunc, 64, n, 2)(layer2)
      val pool = SpatialAveragePooling(8, 8, 1, 1).inputs(layer3)
      val view = View(64).setNumInputDims(3).inputs(pool)
      val output = Linear[Float](64, 10).inputs(view)
      Graph(input, output)
    } else {
      throw new IllegalArgumentException(s"Invalid dataset $dataSet")
    }
    model
  }

  sealed abstract class DatasetType(typeId: Int) extends Serializable

  object DatasetType {

    case object CIFAR10 extends DatasetType(0)

    case object ImageNet extends DatasetType(1)

  }

  sealed abstract class ShortcutType(typeId: Int) extends Serializable

  object ShortcutType {

    case object A extends ShortcutType(0)

    case object B extends ShortcutType(1)

    case object C extends ShortcutType(2)

  }

}