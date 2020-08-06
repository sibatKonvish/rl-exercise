package cn.sibat.ksh.bigdl.yolo3

import com.intel.analytics.bigdl.nn.abstractnn.TensorCriterion
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
  * Created by kong at 2020/8/5
  */
@SerialVersionUID(20200805L)
class YOLOCriterion[@specialized(Float, Double) T: ClassTag](num_classes: Int, iou_thresh: Double, anchors: Array[Array[Double]], alpha_1: Double, alpha_2: Double, alpha_3: Double)(implicit ev: TensorNumeric[T]) extends TensorCriterion[T] {
  var sizeAverage = true


  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    gradInput.resizeAs(input).copy(input)
    gradInput.sub(target)
    output = gradInput.dot(gradInput)
    if (sizeAverage) output = ev.divide(output, ev.fromType[Int](input.nElement()))
    output
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    var norm = ev.fromType[Int](2)
    if (sizeAverage) norm = ev.fromType[Double](2.0 / input.nElement())
    gradInput.mul(norm)
    gradInput
  }
}

object YOLOCriterion {
  def apply[@specialized(Float, Double) T: ClassTag](num_classes: Int, iou_thresh: Double, anchors: Array[Array[Double]], alpha_1: Double, alpha_2: Double, alpha_3: Double)(implicit ev: TensorNumeric[T]): YOLOCriterion[T] = new YOLOCriterion[T](num_classes, iou_thresh, anchors, alpha_1, alpha_2, alpha_3)
}
