package cn.sibat.ksh.bigdl.resnet

import com.intel.analytics.bigdl.nn.{MsraFiller, Ones, SpatialBatchNormalization, SpatialConvolution, SpatialShareConvolution, Zeros}
import com.intel.analytics.bigdl.optim.L2Regularizer
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
  * Created by kong at 2020/7/24
  */
object Convolution {
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
    conv.setInitMethod(MsraFiller(false), Zeros)
    conv
  }
}

object Sbn {
  def apply[@specialized(Float, Double) T: ClassTag](
                                                      nOuput: Int,
                                                      eps: Double = 1e-3,
                                                      momentum: Double = 0.1,
                                                      affine: Boolean = true
                                                    )(implicit en: TensorNumeric[T]): SpatialBatchNormalization[T]
  = SpatialBatchNormalization[T](nOuput, eps, momentum, affine).setInitMethod(Ones, Zeros)
}
