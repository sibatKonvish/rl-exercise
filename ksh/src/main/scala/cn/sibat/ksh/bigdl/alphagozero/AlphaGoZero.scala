package cn.sibat.ksh.bigdl.alphagozero

import com.intel.analytics.bigdl.dataset.{DataSet, Sample}
import com.intel.analytics.bigdl.nn.MSECriterion
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.Optimizer

/**
  * Created by kong at 2020/8/4
  */
object AlphaGoZero {
  def main(args: Array[String]): Unit = {
    val miniBatchSize = 32
    val boardSize = 19
    val numResidualBlocks = 20
    val numFeaturePlanes = 11

    val model = DualResnetModel(numResidualBlocks, numFeaturePlanes)
    val input = Tensor(miniBatchSize, numFeaturePlanes, boardSize, boardSize)
    val policyOutput = Tensor(miniBatchSize, boardSize * boardSize + 1)
    val valueOutput = Tensor(miniBatchSize, 1)
    val train = Sample(Array(input), Array(policyOutput, valueOutput))

  }
}
