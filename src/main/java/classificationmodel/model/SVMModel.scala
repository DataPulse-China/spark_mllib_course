package classificationmodel.model

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

/**
 * SVMModel 支持向量机模型
 *
 */
object SVMModel {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    val spark = SparkSession
      .builder
      .master("local[*]")
      .appName("MultilayerPerceptronClassifierExample")
      .getOrCreate()

    val records = spark.sparkContext.textFile("/home/rjxy/IdeaProjects/spark/spark_mllib_course/src/main/resources/stumbleupon/train_noheader.tsv")
      .map(line => line.split("\t"))

    //迭代次数
    val numIterations = 20

    // 线性模型
    val data: RDD[LabeledPoint] = records.map { r: Array[String] =>
      //修剪过的数据
      val trimmed: Array[String] = r.map((_: String).replaceAll("\"", ""))
      //标签
      val label: Int = trimmed(r.length - 1).toInt
      //特征 数组
      val features: Array[Double] = trimmed.slice(4, r.length - 1).map(d => if (d == "?") 0.0 else d.toDouble)
      LabeledPoint(label, Vectors.dense(features))
    }

    //支持向量机模型
    val svmModel: SVMModel = SVMWithSGD.train(data, numIterations,100,0.001,1)

    val value1: RDD[LabeledPoint] = spark.sparkContext.parallelize(data.take(5))
    value1.foreach(println)
    val value: RDD[Double] = svmModel.predict(value1.map(_.features))
    value.foreach(println)

    val lrTotalCorrect: Double = data.map { point: LabeledPoint =>
      if(svmModel.predict(point.features) == point.label) 1 else 0
    }.sum()
    //进行精确度
    val lrAccuracy2: Double = lrTotalCorrect / data.count()
    println("BFGSmodel进行精确度")
    println(BigDecimal.valueOf(lrAccuracy2))

  }
}
