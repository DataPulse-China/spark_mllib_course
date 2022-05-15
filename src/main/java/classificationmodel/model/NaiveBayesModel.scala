package classificationmodel.model

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel, SVMModel, SVMWithSGD}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

/**
 * SVMModel 支持向量机模型
 *
 */
object NaiveBayesModel {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    val spark = SparkSession
      .builder
      .master("local[*]")
      .appName("MultilayerPerceptronClassifierExample")
      .getOrCreate()

    val records = spark.sparkContext.textFile("/home/rjxy/IdeaProjects/spark/spark_mllib_course/src/main/resources/stumbleupon/train_noheader.tsv")
      .map(line => line.split("\t"))


    // 线性模型  朴素贝叶斯模型 需要这些值为非负数,若有负数则会抛出异常。所以,这里先将所输入特征向量中的负数全部转为 0 。
    val data: RDD[LabeledPoint] = records.map { r: Array[String] =>
      //修剪过的数据
      val trimmed: Array[String] = r.map((_: String).replaceAll("\"", ""))
      //标签
      val label: Int = trimmed(r.length - 1).toInt
      //特征 数组
      val features: Array[Double] = trimmed.slice(4, r.length - 1).map(d => if (d == "?") 0.0 else d.toDouble).map(d => if (d < 0) 0.0 else d)
      LabeledPoint(label, Vectors.dense(features))
    }

    val nbmodel: NaiveBayesModel = NaiveBayes.train(data)

//    BFGSmodel进行精确度
//    0.5803921568627451

    val nbTotalCorrect: Double = data.map { point: LabeledPoint =>
      if(nbmodel.predict(point.features) == point.label) 1 else 0
    }.sum()
    //进行精确度
    val nbAccuracy: Double = nbTotalCorrect / data.count()
    println("nbmodel  朴素贝叶斯模型  进行精确度")
    println(BigDecimal.valueOf(nbAccuracy))

  }
}
