package cn.sibat.ksh

import cn.sibat.ksh.bigdl.yolo3.YOLOUtil
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat

object Test {
  def main(args: Array[String]): Unit = {
    val ten1 = Tensor(Array(2f, 1f, 2f, 2f).map(_ / 8), Array(1, 4)).repeatTensor(Array(3, 1))
    val ten2 = Tensor(Array(2f / 8, 1f / 8, 1.25f, 1.625f, 2f / 8, 1f / 8, 2.0f, 3.75f, 2f / 8, 1f / 8, 4.125f, 2.875f), Array(3, 4))
    print(YOLOUtil.iou(ten1, ten2))
  }
}
