package cn.sibat.ksh

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer
import org.deeplearning4j.nn.conf.{ConvolutionMode, GradientNormalization, NeuralNetConfiguration}
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.zoo.model.helper.DarknetHelper
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.IUpdater

/**
  * Created by kong at 2020/7/6
  */
class KeyPointDetection {
  private var nBoxes = 0
  private var priorBoxes = Array(Array[Double]())
  private var seed = 123L
  private var inputShape = Array[Int]()
  private var numClasses = 0
  private var updater: IUpdater = null
  private var cacheMode = null
  private var workspaceMode = null
  private var cudnnAlgoMode = null

  def conf(): Unit = {
    val priors = Nd4j.create(this.priorBoxes)
    val graphBuilder = new NeuralNetConfiguration.Builder().seed(this.seed)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
      .gradientNormalizationThreshold(1.0)
      .updater(this.updater)
      .l2(1.0e-5)
      .activation(Activation.IDENTITY)
      .cacheMode(this.cacheMode)
      .trainingWorkspaceMode(this.workspaceMode)
      .inferenceWorkspaceMode(this.workspaceMode)
      .cudnnAlgoMode(this.cudnnAlgoMode)
      .graphBuilder()
      .addInputs("input")
      .setInputTypes(InputType.convolutional(inputShape(2), inputShape(1), inputShape(0)))

    DarknetHelper.addLayers(graphBuilder, 1, 3, this.inputShape(0), 16, 2, 2)
    DarknetHelper.addLayers(graphBuilder, 2, 3, 16, 32, 2, 2)
    DarknetHelper.addLayers(graphBuilder, 3, 3, 32, 64, 2, 2)
    DarknetHelper.addLayers(graphBuilder, 4, 3, 64, 128, 2, 2)
    DarknetHelper.addLayers(graphBuilder, 5, 3, 128, 256, 2, 2)
    DarknetHelper.addLayers(graphBuilder, 6, 3, 256, 512, 2, 1)
    DarknetHelper.addLayers(graphBuilder, 7, 3, 512, 1024, 0, 0)
    DarknetHelper.addLayers(graphBuilder, 8, 3, 1024, 1024, 0, 0)
    val layerNumber = 9
    graphBuilder.addLayer("convolution2d_" + layerNumber, new ConvolutionLayer.Builder(1, 1).nIn(1024).nOut(this.nBoxes * (5 + this.numClasses)).weightInit(WeightInit.XAVIER).stride(1, 1).convolutionMode(ConvolutionMode.Same).weightInit(WeightInit.RELU).activation(Activation.IDENTITY).build(), "activation_" + (layerNumber - 1))
      .addLayer("outputs", new Yolo2OutputLayer.Builder().boundingBoxPriors(priors).build(), "convolution2d_" + layerNumber)
      .setOutputs("outputs")
  }
}
