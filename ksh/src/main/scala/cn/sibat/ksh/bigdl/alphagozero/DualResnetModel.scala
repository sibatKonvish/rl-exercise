package cn.sibat.ksh.bigdl.alphagozero

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.{CAddTable, Graph, Input, Linear, ReLU, Sequential, Sigmoid, SoftMax, SpatialBatchNormalization, SpatialConvolution}
import com.intel.analytics.bigdl.numeric.NumericFloat

class DualResnetModel(i: Int, i1: Int) {
  def getModel(): Module[Float] = {
    Sequential()
  }

  def getGraph(blocks: Int): Module[Float] = {
    val input = Input() //11*19*19
    val conv = convBatchNormBlockG(11, 3, false)(input)
    val tower = residualTowerG(blocks)(conv)
    val policy = policyHeadG(tower)
    val value = valueHeadG(tower)
    Graph(input, Array(policy, value))
  }

  /**
    * 卷积网络层
    * conv2d -> batch norm -> ReLU
    *
    * @param nIn           输入特征
    * @param kernelSize    核函数大小
    * @param useActivation 是否使用激活函数
    * @param nOut          输出特征
    * @param stride        滑步
    * @param pad           padding，默认-1，SAME模式
    * @return
    */
  def convBatchNormBlock(nIn: Int, kernelSize: Int, useActivation: Boolean, nOut: Int = 256, stride: Int = 1, pad: Int = -1): Module[Float] = {
    val sequential = Sequential()
    sequential.add(SpatialConvolution(nIn, nOut, kernelSize, kernelSize, strideH = stride, strideW = stride, padH = pad, padW = pad))
    sequential.add(SpatialBatchNormalization(nOut))
    if (useActivation) {
      sequential.add(ReLU())
    }
    sequential
  }

  /**
    * 卷积网络层
    * conv2d -> batch norm -> ReLU
    *
    * @param nIn           输入特征
    * @param kernelSize    核函数大小
    * @param useActivation 是否使用激活函数
    * @param nOut          输出特征
    * @param stride        滑步
    * @param pad           padding，默认-1，SAME模式
    * @param input         输入层
    * @return
    */
  def convBatchNormBlockG(nIn: Int, kernelSize: Int, useActivation: Boolean, nOut: Int = 256, stride: Int = 1, pad: Int = -1)(input: ModuleNode[Float]): ModuleNode[Float] = {
    val conv = SpatialConvolution(nIn, nOut, kernelSize, kernelSize, strideH = stride, strideW = stride, padH = pad, padW = pad).inputs(input)
    val bn = SpatialBatchNormalization(nOut).inputs(conv)
    if (useActivation) {
      ReLU().inputs(bn)
    } else
      bn
  }

  def residualBlockG(input: ModuleNode[Float]): ModuleNode[Float] = {
    val conv1 = convBatchNormBlockG(256, 3, true)(input)
    val conv2 = convBatchNormBlockG(256, 3, false)(conv1)
    val add = CAddTable().inputs(conv1, conv2)
    ReLU().inputs(add)
  }

  def residualTowerG(numBlocks: Int)(input: ModuleNode[Float]): ModuleNode[Float] = {
    var output = residualBlockG(input)
    for (_ <- 2 to numBlocks) {
      output = residualBlockG(output)
    }
    output
  }

  def policyHeadG(input: ModuleNode[Float]): ModuleNode[Float] = {
    val conv = convBatchNormBlockG(256, 3, true, 2)(input)
    val linear = Linear(2 * 19 * 19, 19 * 19 + 1).inputs(conv)
    SoftMax().inputs(linear)
  }

  def valueHeadG(input: ModuleNode[Float]): ModuleNode[Float] = {
    val conv = convBatchNormBlockG(256, 3, true, 1)(input)
    val linear = Linear(19 * 19, 256).inputs(conv)
    val reLU = ReLU().inputs(linear)
    val linear2 = Linear(256, 1).inputs(reLU)
    Sigmoid().inputs(linear2)
  }
}

/**
  * Created by kong at 2020/8/4
  */
object DualResnetModel {
  def apply(numResidualBlocks: Int, numFeaturePlanes: Int): Module[Float] = new DualResnetModel(numResidualBlocks, numFeaturePlanes).getModel()
}
