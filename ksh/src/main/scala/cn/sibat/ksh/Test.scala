package cn.sibat.ksh

import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.commons.io.FilenameUtils
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader
import org.datavec.api.split.NumberedFileInputSplit
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator

object Test {
  def main(args: Array[String]): Unit = {
    val path = "D:/testData/aasData/physionet2012"
    val featuresBaseDir = FilenameUtils.concat(path, "sequence")
    val mortalityBaseDir = FilenameUtils.concat(path, "mortality")

    val trainFeatures = new CSVSequenceRecordReader(1, ",")
    trainFeatures.initialize(new NumberedFileInputSplit(featuresBaseDir + "/%d.csv", 0, 3200 - 1))
    val trainLabels = new CSVSequenceRecordReader()
    trainLabels.initialize(new NumberedFileInputSplit(mortalityBaseDir + "/%d.csv", 0, 3200 - 1))
    val trainData = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, 32, 2, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END)
    val ten = Tensor[Float](2, 2)
    val arr = (1 to 3).toBuffer
    for (i <- 1 to 2; j <- 1 to 2) {
      val value = if (arr.nonEmpty) arr.remove(0) else 0
      ten.setValue(i, j, value)
    }
    println(ten)
  }
}
