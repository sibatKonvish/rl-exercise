package cn.sibat.ksh.bigdl.yolo3

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim.L2Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.Shape

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/**
  * Created by kong at 2020/7/24
  */
class YOLO3Detection(nBoxes: Int, numClass: Int) {
  /**
    * dbl模块
    *
    * @param nInputPlane  输入通道
    * @param nOutputPlane 输出通道
    * @param kernel       核函数大小
    * @param stride       滑步，默认1
    * @param pad          补充，默认-1即SAME模式其他为Valid
    * @param poolSize     pooling大小，默认0
    * @param poolStride   pooling滑步，默认0
    * @param input        输入模块
    * @return
    */
  def dbl(nInputPlane: Int, nOutputPlane: Int, kernel: Int, stride: Int = 1, pad: Int = -1, poolSize: Int = 0, poolStride: Int = 0)(input: ModuleNode[Float]): ModuleNode[Float] = {
    val conv = SpatialConvolution(nInputPlane, nOutputPlane, kernel, kernel, strideH = stride, strideW = stride, padH = pad, padW = pad, wRegularizer = L2Regularizer(5e-4), bRegularizer = L2Regularizer(5e-4)).inputs(input)
    val bn = SpatialBatchNormalization(nOutputPlane).inputs(conv)
    val relu = LeakyReLU().inputs(bn)
    if (poolSize > 0)
      SpatialMaxPooling(poolSize, poolSize, poolStride, poolStride).inputs(relu)
    else
      relu
  }

  def res_unit(nInputPlane: Int, nOutputPlane: Int)(input: ModuleNode[Float]): ModuleNode[Float] = {
    val d1 = dbl(nOutputPlane, nInputPlane, 1)(input) //208*208*64
    val d2 = dbl(nInputPlane, nOutputPlane, 3)(d1)
    val identity = Identity().inputs(input)
    val add = CAddTable(inplace = true).inputs(d2, identity)
    LeakyReLU().inputs(add)
  }

  def res_n(n: Int, nFeatures: Int)(input: ModuleNode[Float]): ModuleNode[Float] = {
    var output = dbl(nFeatures / 2, nFeatures, 3, stride = 2)(input)
    for (_ <- 1 to n) {
      output = res_unit(nFeatures / 2, nFeatures)(output)
    }
    output
  }

  def dbl_5(nFeatures: Int)(input: ModuleNode[Float]*): ModuleNode[Float] = {
    var nInput = nFeatures
    val in = if (input.length > 1) {
      nInput = nFeatures + nFeatures / 2
      JoinTable(2, 0).inputs(input: _*)
    } else
      input.head
    val dbl1 = dbl(nInput, nFeatures / 2, 1)(in)
    val dbl2 = dbl(nFeatures / 2, nFeatures, 3)(dbl1)
    val dbl3 = dbl(nFeatures, nFeatures / 2, 1)(dbl2)
    val dbl4 = dbl(nFeatures / 2, nFeatures, 3)(dbl3)
    val dbl5 = dbl(nFeatures, nFeatures / 2, 1)(dbl4)
    dbl5
  }

  def upSampling(nFeatures: Int)(input: ModuleNode[Float]): ModuleNode[Float] = {
    val d1 = dbl(nFeatures, nFeatures / 2, 1)(input)
    UpSampling2D(Array(2, 2)).inputs(d1)
  }

  def createModel(): Module[Float] = {
    val out = nBoxes * (5 + numClass)
    val input = Input() //416*416*3
    val dbl1 = dbl(3, 32, 3)(input) //416*416*32
    val res1 = res_n(1, 64)(dbl1) //208*208*64
    val res2 = res_n(2, 128)(res1) //104*104*128
    val res8 = res_n(8, 256)(res2) //52*52*256
    val res8_2 = res_n(8, 512)(res8) //26*26*512
    val res4 = res_n(4, 1024)(res8_2) //13*13*1024 //darknet53
    val dbl2 = dbl_5(1024)(res4) //13*13*512
    val y1_dbl = dbl(512, 1024, 3)(dbl2) //13*13*1024
    val y1_conv = SpatialConvolution(1024, out, 1, 1).inputs(y1_dbl) //13*13*out
    val sample1 = upSampling(512)(dbl2) //13*13*512 -> 13*13*256 -> 26*26*256
    val y2_dbl5 = dbl_5(512)(sample1, res8_2) //26*26*256
    val y2_dbl = dbl(256, 512, 3)(y2_dbl5) //26*26*512
    val y2_conv = SpatialConvolution(512, out, 1, 1).inputs(y2_dbl) //26*26*out
    val sample2 = upSampling(256)(y2_dbl5) //26*26*256 -> 26*26*128 -> 52*52*128
    val y3_dbl5 = dbl_5(256)(sample2, res8) //52*52*128
    val y3_dbl = dbl(128, 256, 3)(y3_dbl5) //52*52*256
    val y3_conv = SpatialConvolution(256, out, 1, 1).inputs(y3_dbl) //52*52*out
    Graph(input, Array(y1_conv, y2_conv, y3_conv))
  }

  def darknet53(): Module[Float] = {
    val input = Input() //416*416*3
    val dbl1 = dbl(3, 32, 3)(input) //416*416*32
    val res1 = res_n(1, 64)(dbl1) //208*208*64
    val res2 = res_n(2, 128)(res1) //104*104*128
    val res8 = res_n(8, 256)(res2) //52*52*256
    val res8_2 = res_n(8, 512)(res8) //26*26*512
    val res4 = res_n(4, 1024)(res8_2) //13*13*1024
    Graph(input, Array(res8, res8_2, res4))
  }
}

class YOLOLayer(lambdaCoord: Float, lambdaNoObj: Float) {
  /**
    * 处理输出的feature map
    *
    * @param out     Tensor(N,N,nBoxes,4+1+classes)
    * @param anchors anchor box列表
    * @return (boxes(N,N,nBoxes,4),confidence(N,N,nBoxes,1),prob(N,N,nBoxes,classes))
    */
  def process_feats(out: Tensor[Float], anchors: Array[(Int, Int)] = null, shape: (Int, Int) = (416, 416)): (Tensor[Float], Tensor[Float], Tensor[Float]) = {
    val Array(grid_h, grid_w, num_boxes) = (1 to 3).map(out.size).toArray
    require(anchors.length == num_boxes, s"anchor box 不一致,${anchors.length}!=$num_boxes")
    val stride = shape._1 / grid_h
    val anchor_ten = Tensor[Float](anchors.flatMap(t => Array(t._1.toFloat, t._2.toFloat)), Array(anchors.length, 2)).repeatTensor(Array(grid_h * grid_w, 1))
    val xy_wh = out.narrow(4, 1, 4)
    val boxes = Tensor[Float]().resizeAs(xy_wh).copy(xy_wh)
    val xy = boxes.narrow(4, 1, 2)
    val wh = boxes.narrow(4, 3, 2)
    val confidence = out.narrow(4, 5, 1)
    val classes = out.narrow(4, 6, out.size(4) - 5)
    val box_xy = xy.mul(-1).exp().add(1).pow(-1)
    wh.exp().cmul(anchor_ten) //计算相对统一图的anchor边长
    val box_confidence = sigmoid(confidence)
    val box_class_probs = sigmoid(classes)

    //将网格偏移添加到预测的中心坐标
    //每个网格的offsets
    val temp = Tensor[Float](grid_h, grid_w, num_boxes, 2)
    for (i <- 1 to grid_h; j <- 1 to grid_w; k <- 1 to num_boxes) {
      temp.setValue(i, j, k, 1, i - 1)
      temp.setValue(i, j, k, 2, j - 1)
    }
    box_xy.add(temp).mul(stride) //真实尺寸像素点

    (boxes, box_confidence, box_class_probs)
  }

  /**
    * sigmoid函数
    *
    * @param tensor tensor
    * @return
    */
  private def sigmoid(tensor: Tensor[Float]): Tensor[Float] = {
    val xy_exp = Tensor[Float]().resizeAs(tensor).copy(tensor).mul(-1).exp().add(1)
    Tensor[Float]().resizeAs(xy_exp).fill(1).cdiv(xy_exp)
  }

  /**
    * 将中心点、高、宽坐标转为左上右下坐标形式
    *
    * @param boxes tensor[x,y,w,h]
    * @return
    */
  def boxes2TopLR(boxes: Tensor[Float]): Tensor[Float] = {
    val w = boxes.narrow(4, 3, 1) / 2
    val h = boxes.narrow(4, 4, 1) / 2
    val x0 = boxes.narrow(4, 1, 1) - w
    val y0 = boxes.narrow(4, 2, 1) - h
    val x1 = boxes.narrow(4, 1, 1) + w
    val y1 = boxes.narrow(4, 2, 1) + h
    concat(x0, y0, x1, y1)
  }

  /**
    * 计算两个box的iou
    * box[x0,y0,x1,y1]
    *
    * @param box1 anchor box
    * @param box2 anchor box
    * @return
    */
  def iou(box1: (Float, Float, Float, Float, Float), box2: (Float, Float, Float, Float, Float)): Float = {
    val x0 = math.max(box1._1, box2._1)
    val y0 = math.max(box1._2, box2._2)
    val x1 = math.min(box1._3, box2._3)
    val y1 = math.min(box1._4, box2._4)
    val area = (x1 - x0) * (y1 - y0)
    val b1_area = (box1._3 - box1._1) * (box1._4 - box1._2)
    val b2_area = (box2._3 - box2._1) * (box2._4 - box2._2)

    area / (b1_area + b2_area - area + 1e-5f)
  }

  /**
    * 非极大值抑制
    *
    * @param predict tensor
    */
  def nonMaxSuppression(predict: Tensor[Float]): mutable.Map[String, Array[(Float, Float, Float, Float, Float)]] = {
    val result = new mutable.HashMap[String, Array[(Float, Float, Float, Float, Float)]]() //key=class,value=(x0,y0,x1,y1,score)
    for (i <- 1 to predict.size(1)) { //batch size
      val ith = predict.select(1, i) //tensor[N*N,5+c]
      for (j <- 1 to ith.size(1)) { //N*N
        val jth = ith.select(1, j) //tensor[5+c]
        val confidence = jth.valueAt(5)
        if (confidence > lambdaNoObj) {
          val x0 = jth.valueAt(1)
          val y0 = jth.valueAt(2)
          val x1 = jth.valueAt(3)
          val y1 = jth.valueAt(4)
          val classes = jth.select(1, 6).max(2)._2.value()
          val old = result.getOrElse(classes.toString, Array())
          result.put(classes.toString, old ++ Array((x0, y0, x1, y1, confidence)))
        }
      }
    }
    val iter = result.toIterator
    while (iter.hasNext) {
      val next = iter.next()
      val arr = new ArrayBuffer[(Float, Float, Float, Float, Float)]()
      var value = next._2.sortBy(_._5).reverse
      while (value.nonEmpty) {
        arr += value.head
        value = value.tail.filter(t => iou(t, value.head) >= lambdaCoord)
      }
      result.put(next._1, arr.toArray)
    }
    result
  }

  /**
    * concat 多个tensor
    * e.g.
    * t1:
    * 0 0
    * 0 0
    * t2:
    * 1 1
    * 1 1
    * concat(t1,t2) =>
    * 0 0 1 1
    * 0 0 1 1
    *
    * @param t1 tensor
    * @param t2 tensor
    * @return
    */
  def concat(t1: Tensor[Float], t2: Tensor[Float]*): Tensor[Float] = {
    require(t2.forall(t => t.size().take(t.size().length - 1).mkString(",").equals(t1.size().take(t.size().length - 1).mkString(","))))
    val data = t1.storage().array()
    var offset = t1.storageOffset() - 1
    val lastDim = t1.size().last
    val size = t2.map(_.size().last)
    val datas = t2.map(_.storage().array().toIterator)
    val result = new ArrayBuffer[Float]()
    var lastIndex = 0
    while (offset <= data.length) {
      val index = offset / lastDim
      if (index != lastIndex) {
        for (i <- size.indices) {
          var j = 0
          while (j < size(i)) {
            result += datas(i).next()
            j += 1
          }
        }
      }
      if (offset < data.length)
        result += data(offset)
      lastIndex = index
      offset += 1
    }
    val newShape = t1.size().take(t1.size().length - 1) ++ Array(lastDim + size.sum)
    Tensor(result.toArray, newShape)
  }
}

object YOLO3Detection {
  def main(args: Array[String]): Unit = {
    val tensor = Tensor[Float](1, 3, 416, 416).rand()
    val yolo = new YOLO3Detection(3, 5)
    val model = yolo.createModel()
    println(model.forward(tensor))
  }
}
