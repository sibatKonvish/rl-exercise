package cn.sibat.ksh

import cn.sibat.ksh.bigdl.resnet.TestParams
import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.commons.io.FilenameUtils
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader
import org.datavec.api.split.NumberedFileInputSplit
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer
import org.deeplearning4j.nn.conf.{ConvolutionMode, GradientNormalization, WorkspaceMode}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.transferlearning.{FineTuneConfiguration, TransferLearning}
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.zoo.model.TinyYOLO
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam

object Test {

  import cn.sibat.ksh.bigdl.resnet.DataSetLoadUtil._

  def main(args: Array[String]): Unit = {
    val priorBoxes = Array(Array(2.0, 5.0), Array(2.5, 6.0), Array(3.0, 7.0), Array(3.5, 8.0), Array(4.0, 9.0))
    val seed = 123
    val learningRate = 1e-4
    val nBoxes = 5
    val nClasses = 10
    val lambdaCoord = 0.5
    val lambdaNoObj = 1.0

    val pretrained = TinyYOLO.builder.build.initPretrained.asInstanceOf[ComputationGraph]
    val priors = Nd4j.create(priorBoxes)

    val fineTuneConf = new FineTuneConfiguration.Builder()
      .seed(seed)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
      .gradientNormalizationThreshold(1.0)
      .updater(new Adam.Builder()
        .learningRate(learningRate).build).l2(0.00001) //.updater(new Nesterovs.Builder().learningRate(learningRate).momentum(lrMomentum).build())
      .activation(Activation.IDENTITY).trainingWorkspaceMode(WorkspaceMode.ENABLED).inferenceWorkspaceMode(WorkspaceMode.ENABLED).build

    val model = new TransferLearning.GraphBuilder(pretrained)
      .fineTuneConfiguration(fineTuneConf)
      .removeVertexKeepConnections("conv2d_9")
      .removeVertexKeepConnections("outputs")
      .addLayer("convolution2d_9", new ConvolutionLayer.Builder(1, 1)
        .nIn(1024)
        .nOut(nBoxes * (5 + nClasses))
        .stride(1, 1)
        .convolutionMode(ConvolutionMode.Same)
        .weightInit(WeightInit.XAVIER)
        .activation(Activation.IDENTITY)
        .build, "leaky_re_lu_8")
      .addLayer("outputs", new Yolo2OutputLayer.Builder()
        .lambdaNoObj(lambdaNoObj)
        .lambdaCoord(lambdaCoord)
        .boundingBoxPriors(priors)
        .build, "convolution2d_9")
      .setOutputs("outputs").build
    println(model.summary())
  }
}
