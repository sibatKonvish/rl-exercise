package cn.sibat.ksh.bigdl.yolo3

import java.awt.image.BufferedImage
import java.io.{BufferedReader, ByteArrayInputStream, ByteArrayOutputStream, File, FileInputStream, FileReader}
import java.nio.channels.Channels
import java.nio.file.{Path, Paths}

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dataset.image.{BGRImage, BGRImgToBatch, BytesToBGRImg, BytesToGreyImg}
import com.intel.analytics.bigdl.dataset.{ByteRecord, DataSet, Image, Sample}
import com.intel.analytics.bigdl.example.imageclassification.RowToByteRecords
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim.{L2Regularizer, Optimizer, Top1Accuracy, Trigger}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.Shape
import javax.imageio.ImageIO

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/**
  * yolo模型
  *
  * Created by kong at 2020/7/24
  */
class YOLO3Model(nBoxes: Int, numClass: Int) {
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

object YOLOUtil {
  /**
    * 处理输出的feature map
    *
    * @param out     Tensor(b,N,N,nBoxes,4+1+classes)
    * @param anchors anchor box列表
    * @return (boxes(N,N,nBoxes,4),confidence(N,N,nBoxes,1),prob(N,N,nBoxes,classes))
    */
  def process_output(out: Tensor[Float], anchors: Array[(Int, Int)] = null, shape: (Int, Int) = (416, 416)): (Tensor[Float], Tensor[Float], Tensor[Float]) = {
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
    * @param boxes tensor[b,N,N,nBox,xywh]
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
    * box[x0,y0,x1,y1,score]
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
    * 计算iou
    *
    * @param boxes1 tensor[:::,xywh]
    * @param boxes2 tensor[:::,xywh]
    */
  def iou(boxes1: Tensor[Float], boxes2: Tensor[Float]): Tensor[Float] = {
    require(boxes1.size().mkString(",").equals(boxes2.size().mkString(",")), s"boxes1 shape ${boxes1.size().mkString(",")} != boxes2 shape ${boxes2.size().mkString(",")}")
    val shape1 = boxes1.size()
    val shape2 = boxes2.size()
    val wh1 = boxes1.narrow(shape1.length, 3, 2) / 2
    val wh2 = boxes2.narrow(shape2.length, 3, 2) / 2
    val xy_1 = boxes1.narrow(shape1.length, 1, 2)
    val xy_2 = boxes2.narrow(shape2.length, 1, 2)
    val box1_x0y0 = xy_1 - wh1
    val box1_x1y1 = xy_1 + wh1
    val box2_x0y0 = xy_2 - wh2
    val box2_x1y1 = xy_2 + wh2
    //避免坐标混乱,改变原始值
    box1_x0y0.cmin(box1_x1y1)
    box1_x1y1.cmax(box1_x0y0)
    box2_x0y0.cmin(box2_x1y1)
    box2_x1y1.cmax(box1_x0y0)

    //两个矩形的面积
    val box1_area = (box1_x1y1.narrow(shape1.length, 1, 1) - box1_x0y0.narrow(shape1.length, 1, 1)) cmul (box1_x1y1.narrow(shape1.length, 2, 1) - box1_x0y0.narrow(shape1.length, 2, 1))
    val box2_area = (box2_x1y1.narrow(shape2.length, 1, 1) - box2_x0y0.narrow(shape2.length, 1, 1)) cmul (box2_x1y1.narrow(shape2.length, 2, 1) - box2_x0y0.narrow(shape2.length, 2, 1))

    //相交矩形的左上和右下坐标
    val left_up = box1_x0y0.cmax(box2_x0y0) //改变原始值
    val right_down = box1_x1y1.cmin(box2_x1y1) //改变原始值

    //相交矩阵的面积
    val intersection = (right_down - left_up).apply1(math.max(0.0f, _))
    val inter_area = intersection.narrow(shape1.length, 1, 1) cmul intersection.narrow(shape2.length, 2, 1)
    val union_area = box1_area + box2_area - inter_area
    inter_area / union_area.apply1(t => if (t == 0f) 1f else t) //避免分母为0
  }

  /**
    * 计算ciou
    * p2:相交矩阵的对角线的平方
    * c2:相融矩阵的对角线的平方
    * v:atan(w/h)
    * a:v / (1 - iou + v)
    * ciou = iou - p2 / c2 - a * v
    *
    * @param boxes1 tensor[batchSize,grid,grid,nBox,xywh]
    * @param boxes2 tensor[batchSize,grid,grid,nBox,xywh]
    */
  def ciou(boxes1: Tensor[Float], boxes2: Tensor[Float]): Tensor[Float] = {
    require(boxes1.size().mkString(",").equals(boxes2.size().mkString(",")), s"boxes1 shape ${boxes1.size().mkString(",")} != boxes2 shape ${boxes2.size().mkString(",")}")
    val shape = boxes1.size()
    val wh1 = boxes1.narrow(shape.length, 3, 2) / 2
    val wh2 = boxes2.narrow(shape.length, 3, 2) / 2
    val xy_1 = boxes1.narrow(shape.length, 1, 2)
    val xy_2 = boxes2.narrow(shape.length, 1, 2)
    val box1_x0y0 = xy_1 - wh1
    val box1_x1y1 = xy_1 + wh1
    val box2_x0y0 = xy_2 - wh2
    val box2_x1y1 = xy_2 + wh2
    //避免坐标混乱,改变原始值
    box1_x0y0.cmin(box1_x1y1)
    box1_x1y1.cmax(box1_x0y0)
    box2_x0y0.cmin(box2_x1y1)
    box2_x1y1.cmax(box1_x0y0)

    //两个矩形的面积
    val box1_area = (box1_x1y1.narrow(shape.length, 1, 1) - box1_x0y0.narrow(shape.length, 1, 1)) cmul (box1_x1y1.narrow(shape.length, 2, 1) - box1_x0y0.narrow(shape.length, 2, 1))
    val box2_area = (box2_x1y1.narrow(shape.length, 1, 1) - box2_x0y0.narrow(shape.length, 1, 1)) cmul (box2_x1y1.narrow(shape.length, 2, 1) - box2_x0y0.narrow(shape.length, 2, 1))

    //相交矩形的左上和右下坐标
    val left_up = box1_x0y0.clone().cmax(box2_x0y0) //不改变原始值
    val right_down = box1_x1y1.clone().cmin(box2_x1y1) //不改变原始值

    //相交矩阵的面积
    val intersection = (right_down - left_up).apply1(math.max(0.0f, _))
    val inter_area = intersection.narrow(shape.length, 1, 1) cmul intersection.narrow(shape.length, 2, 1)
    val union_area = box1_area + box2_area - inter_area
    val iou = inter_area / union_area.apply1(t => if (t == 0f) 1f else t) //避免分母为0

    //包围矩形的左上右下坐标
    val enclose_left_up = box1_x0y0.cmin(box2_x0y0)
    val enclose_right_down = box1_x1y1.cmax(box2_x1y1)

    //包围矩阵的对角线的平方
    val enclose_c2 = (enclose_right_down - enclose_left_up).pow(2).sum(shape.length)

    //两矩形中心点距离的平方
    val p2 = (xy_1 - xy_2).pow(2).sum(shape.length)

    //增加av
    val atan1 = (wh1.narrow(shape.length, 1, 1) / wh1.narrow(shape.length, 2, 1)).apply1(math.atan(_).toFloat)
    val atan2 = (wh2.narrow(shape.length, 1, 1) / wh2.narrow(shape.length, 2, 1).apply1(t => if (t > 0) t else 1f)).apply1(math.atan(_).toFloat)
    val v = (atan1 - atan2).pow(2).mul(4).div(math.pow(math.Pi, 2).toFloat)
    val a = v / (iou.clone().mul(-1).add(1) + v)
    iou.sub(p2 cdiv enclose_c2).sub(a cmul v)
  }

  /**
    * 计算diou
    * p2:相交矩阵的对角线的平方
    * c2:相融矩阵的对角线的平方
    * diou = iou - p2 / c2
    *
    * @param boxes1 tensor[batchSize,grid,grid,nBox,xywh]
    * @param boxes2 tensor[batchSize,grid,grid,nBox,xywh]
    */
  def diou(boxes1: Tensor[Float], boxes2: Tensor[Float]): Tensor[Float] = {
    require(boxes1.size().mkString(",").equals(boxes2.size().mkString(",")), s"boxes1 shape ${boxes1.size().mkString(",")} != boxes2 shape ${boxes2.size().mkString(",")}")
    val shape = boxes1.size()
    val wh1 = boxes1.narrow(shape.length, 3, 2) / 2
    val wh2 = boxes2.narrow(shape.length, 3, 2) / 2
    val xy_1 = boxes1.narrow(shape.length, 1, 2)
    val xy_2 = boxes2.narrow(shape.length, 1, 2)
    val box1_x0y0 = xy_1 - wh1
    val box1_x1y1 = xy_1 + wh1
    val box2_x0y0 = xy_2 - wh2
    val box2_x1y1 = xy_2 + wh2
    //避免坐标混乱,改变原始值
    box1_x0y0.cmin(box1_x1y1)
    box1_x1y1.cmax(box1_x0y0)
    box2_x0y0.cmin(box2_x1y1)
    box2_x1y1.cmax(box1_x0y0)

    //两个矩形的面积
    val box1_area = (box1_x1y1.narrow(shape.length, 1, 1) - box1_x0y0.narrow(shape.length, 1, 1)) cmul (box1_x1y1.narrow(shape.length, 2, 1) - box1_x0y0.narrow(shape.length, 2, 1))
    val box2_area = (box2_x1y1.narrow(shape.length, 1, 1) - box2_x0y0.narrow(shape.length, 1, 1)) cmul (box2_x1y1.narrow(shape.length, 2, 1) - box2_x0y0.narrow(shape.length, 2, 1))

    //相交矩形的左上和右下坐标
    val left_up = box1_x0y0.clone().cmax(box2_x0y0) //不改变原始值
    val right_down = box1_x1y1.clone().cmin(box2_x1y1) //不改变原始值

    //相交矩阵的面积
    val intersection = (right_down - left_up).apply1(math.max(0.0f, _))
    val inter_area = intersection.narrow(shape.length, 1, 1) cmul intersection.narrow(shape.length, 2, 1)
    val union_area = box1_area + box2_area - inter_area
    val iou = inter_area / union_area.apply1(t => if (t == 0f) 1f else t) //避免分母为0

    //包围矩形的左上右下坐标
    val enclose_left_up = box1_x0y0.cmin(box2_x0y0)
    val enclose_right_down = box1_x1y1.cmax(box2_x1y1)

    //包围矩阵的对角线的平方
    val enclose_c2 = (enclose_right_down - enclose_left_up).pow(2).sum(shape.length)

    //两矩形中心点距离的平方
    val p2 = (xy_1 - xy_2).pow(2).sum(shape.length)
    iou.sub(p2 cdiv enclose_c2)
  }

  /**
    * 计算giou
    * C:相融的面积
    * AUB:AUB的面积
    * giou = iou - (C-AUB)/C
    *
    * @param boxes1 tensor[batchSize,grid,grid,nBox,xywh]
    * @param boxes2 tensor[batchSize,grid,grid,nBox,xywh]
    */
  def giou(boxes1: Tensor[Float], boxes2: Tensor[Float]): Tensor[Float] = {
    require(boxes1.size().mkString(",").equals(boxes2.size().mkString(",")), s"boxes1 shape ${boxes1.size().mkString(",")} != boxes2 shape ${boxes2.size().mkString(",")}")
    val shape = boxes1.size()
    val wh1 = boxes1.narrow(shape.length, 3, 2) / 2
    val wh2 = boxes2.narrow(shape.length, 3, 2) / 2
    val xy_1 = boxes1.narrow(shape.length, 1, 2)
    val xy_2 = boxes2.narrow(shape.length, 1, 2)
    val box1_x0y0 = xy_1 - wh1
    val box1_x1y1 = xy_1 + wh1
    val box2_x0y0 = xy_2 - wh2
    val box2_x1y1 = xy_2 + wh2
    //避免坐标混乱,改变原始值
    box1_x0y0.cmin(box1_x1y1)
    box1_x1y1.cmax(box1_x0y0)
    box2_x0y0.cmin(box2_x1y1)
    box2_x1y1.cmax(box1_x0y0)

    //两个矩形的面积
    val box1_area = (box1_x1y1.narrow(shape.length, 1, 1) - box1_x0y0.narrow(shape.length, 1, 1)) cmul (box1_x1y1.narrow(shape.length, 2, 1) - box1_x0y0.narrow(shape.length, 2, 1))
    val box2_area = (box2_x1y1.narrow(shape.length, 1, 1) - box2_x0y0.narrow(shape.length, 1, 1)) cmul (box2_x1y1.narrow(shape.length, 2, 1) - box2_x0y0.narrow(shape.length, 2, 1))

    //相交矩形的左上和右下坐标
    val left_up = box1_x0y0.clone().cmax(box2_x0y0) //不改变原始值
    val right_down = box1_x1y1.clone().cmin(box2_x1y1) //不改变原始值

    //相交矩阵的面积
    val intersection = (right_down - left_up).apply1(math.max(0.0f, _))
    val inter_area = intersection.narrow(shape.length, 1, 1) cmul intersection.narrow(shape.length, 2, 1)
    val union_area = box1_area + box2_area - inter_area
    val iou = inter_area / union_area.apply1(t => if (t == 0f) 1f else t) //避免分母为0

    //包围矩形的左上右下坐标
    val enclose = (box1_x1y1.cmax(box2_x1y1) - box1_x0y0.cmin(box2_x0y0)).apply1(math.max(0.0f, _))
    val enclose_area = enclose.narrow(shape.length, 1, 1) cmul enclose.narrow(shape.length, 2, 1)

    iou.sub((enclose_area - union_area) / enclose_area)
  }

  /**
    * 非极大值抑制
    *
    * @param predict tensor
    */
  def nonMaxSuppression(predict: Tensor[Float], lambdaNoObj: Double, lambdaCoord: Double): mutable.Map[String, Array[(Float, Float, Float, Float, Float)]] = {
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

  /**
    * 读取图片
    *
    * @param path 图片路径
    * @return
    */
  def readRawImage(path: Path): BufferedImage = {
    var fis: FileInputStream = null
    try {
      fis = new FileInputStream(path.toString)
      val channel = fis.getChannel
      val byteArrayOutputStream = new ByteArrayOutputStream
      channel.transferTo(0, channel.size, Channels.newChannel(byteArrayOutputStream))
      val image = ImageIO.read(new ByteArrayInputStream(byteArrayOutputStream.toByteArray))
      require(image != null, "Can't read file " + path + ", ImageIO.read is null")
      image
    } catch {
      case ex: Exception =>
        ex.printStackTrace()
        System.err.println("Can't read file " + path)
        throw ex
    } finally {
      if (fis != null) {
        fis.close()
      }
    }
  }
}

object YOLO3Detection {

  def load(str: String, num_class: Int, max_bbox_per_scale: Int, anchors: Array[Array[Double]]): Array[ByteRecord] = {
    val file = new BufferedReader(new FileReader(str))

    val train_input_sizes = 416
    val strides = Array(8, 16, 32)
    val grid = strides.map(train_input_sizes / _)

    val result = new ArrayBuffer[Sample[Float]]()
    var line = file.readLine()
    while (line != null) {
      val split = line.split(" ")
      val image_path = split(0)
      val bi = YOLOUtil.readRawImage(Paths.get(image_path))
      val height = bi.getHeight
      val width = bi.getWidth()
      val img = BGRImage.resizeImage(bi, train_input_sizes, train_input_sizes) //映射为416*416
      val bgr = new BGRImage(train_input_sizes, train_input_sizes).copy(img)
      val label_sbbox = Tensor(grid(0), grid(0), 3, 5 + num_class)
      val label_mbbox = Tensor(grid(1), grid(1), 3, 5 + num_class)
      val label_lbbox = Tensor(grid(2), grid(2), 3, 5 + num_class)

      val sbboxes = Tensor(max_bbox_per_scale, 4)
      val mbboxes = Tensor(max_bbox_per_scale, 4)
      val lbboxes = Tensor(max_bbox_per_scale, 4)

      val boxes = new ArrayBuffer[(Int, Int, Int, Int, Int)]()
      if (split.length == 1) {
        boxes += ((10, 10, 101, 103, 0)) //没有标注物，当做背景图
      } else {
        split.tail.foreach(s => { //多个box
          val Array(x0, y0, x1, y1, classes) = s.split(",").map(_.toInt)
          val scale_x0 = train_input_sizes * x0 / height
          val scale_y0 = train_input_sizes * y0 / width
          val scale_x1 = train_input_sizes * x1 / height
          val scale_y1 = train_input_sizes * y1 / width
          val x = (scale_x0 + scale_x1) * 0.5f
          val y = (scale_y0 + scale_y1) * 0.5f
          val w = scale_x1 - scale_x0
          val h = scale_y1 - scale_y0
          val bbox_xywh_scale = Tensor(strides.flatMap(t => Array(x / t, y / t, w * 1f / t, h * 1f / t)), Array(3, 4)) //3*4
          for (elem <- anchors) {
            val elem_tensor = Tensor(elem.map(_.toFloat), Array(3, 2))
            val anchors_scale = bbox_xywh_scale.narrow(2, 1, 2).apply1(t => math.floor(t).toInt + 0.5f)
            YOLOUtil.concat(anchors_scale, elem_tensor)
          }
          boxes += ((scale_x0, scale_y0, scale_x1, scale_y1, classes))
        })
      }
      val image = Tensor(bgr.content, Array(train_input_sizes, train_input_sizes))
      result += Sample(Array(image), Array(image, image))
      line = file.readLine()
    }

    Array()
  }

  def main(args: Array[String]): Unit = {
    val train_path = "voc2012_train.txt"
    val val_path = "voc2012_val.txt"
    val classes_path = "voc_classes.txt"
    val file = new BufferedReader(new FileReader(classes_path))
    var num_classes = 1
    var line = file.readLine()
    while (line != null) {
      num_classes += 1
      line = file.readLine()
    }
    val anchors = Array(Array(1.25, 1.625, 2.0, 3.75, 4.125, 2.875), Array(1.875, 3.8125, 3.875, 2.8125, 3.6875, 7.4375), Array(3.625, 2.8125, 4.875, 6.1875, 11.65625, 10.1875))
    val max_bbox_per_scale = 150
    val iou_loss_thresh = 0.7
    val alpha_1 = 0.5
    val alpha_2 = 0.5
    val alpha_3 = 0.5

    //读取数据
    val train = DataSet.array(load(train_path, num_classes, max_bbox_per_scale, anchors)) -> BytesToBGRImg() -> BGRImgToBatch(10)
    val valid = DataSet.array(load(val_path, num_classes, max_bbox_per_scale, anchors)) -> BytesToBGRImg() -> BGRImgToBatch(1)

    val model = new YOLO3Model(3, num_classes)
    val criterion = YOLOCriterion(num_classes, iou_loss_thresh, anchors, alpha_1, alpha_2, alpha_3)

    val optimizer = Optimizer(model.createModel(), train, criterion)
    optimizer.setValidation(Trigger.everyEpoch, valid, Array(new Top1Accuracy[Float]()))
      .setEndWhen(Trigger.maxEpoch(4))
      .optimize()
  }
}
