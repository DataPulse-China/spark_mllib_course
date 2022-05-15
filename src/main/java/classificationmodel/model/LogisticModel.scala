package classificationmodel.model

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS, LogisticRegressionWithSGD}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object LogisticModel {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    val spark = SparkSession
      .builder
      .master("local[*]")
      .appName("MultilayerPerceptronClassifierExample")
      .getOrCreate()

    val records = spark.sparkContext.textFile("/home/rjxy/IdeaProjects/spark/spark_mllib_course/src/main/resources/stumbleupon/train_noheader.tsv")
      .map(line => line.split("\t"))

    // 对于线性模型
    val data: RDD[LabeledPoint] = records.map { r =>
      val trimmed = r.map(_.replaceAll("\"", ""))
      val label = trimmed(r.size - 1).toInt
      val features = trimmed.slice(4, r.size - 1).map(d => if (d == "?") 0.0 else d.toDouble)
      LabeledPoint(label, Vectors.dense(features))
    }
    import spark.implicits._
    data.toDF().show(false)


    // 将数据分割为训练数据和测试数据
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
    val train = splits(0)
    val test = splits(1)

    val numIterations = 20
    val maxTreeDepth = 5

    val SGDModel: LogisticRegressionModel = LogisticRegressionWithSGD.train(data, numIterations)
    val BFGSmodel: LogisticRegressionModel = new LogisticRegressionWithLBFGS().run(data)

    val dataPoint = spark.sparkContext.parallelize(data.take(5))
    val SGD: RDD[Double] = SGDModel.predict(dataPoint.map((_: LabeledPoint).features))
    val BFGS: RDD[Double] = BFGSmodel.predict(dataPoint.map((_: LabeledPoint).features))
    println("============")
    dataPoint.map(_.label).foreach(println)
    println("============")
    SGD.foreach(println)
    println("============")
    BFGS.foreach(println)

    //SGDModel逻辑回归完全正确数量
    val lrTotalCorrect: Double = data.map { point: LabeledPoint =>
      if(SGDModel.predict(point.features) == point.label) 1 else 0
    }.sum()
    //进行精确度
    val lrAccuracy: Double = lrTotalCorrect / data.count()
    println("SGDModel进行精确度")
    println(BigDecimal.valueOf(lrAccuracy))


    //BFGSmodel逻辑回归完全正确数量
    val lrTotalCorrect2: Double = data.map { point: LabeledPoint =>
      if(BFGSmodel.predict(point.features) == point.label) 1 else 0
    }.sum()
    //进行精确度
    val lrAccuracy2: Double = lrTotalCorrect2 / data.count()
    println("BFGSmodel进行精确度")
    println(BigDecimal.valueOf(lrAccuracy2))


//    进行精确度
//    0.5145368492224476
//    进行精确度
//    0.6277214334009465


  }
}
