package classificationmodel.model

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{OneHotEncoder }
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.Entropy
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object DecisionTreeModel {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    val spark = SparkSession
      .builder
      .master("local[*]")
      .appName("MultilayerPerceptronClassifierExample")
      .getOrCreate()

    val records = spark.sparkContext.textFile("/home/rjxy/IdeaProjects/spark/spark_mllib_course/src/main/resources/stumbleupon/train_noheader.tsv")
      .map(line => line.split("\t"))

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
    val maxDepth = 10
    val decisonModel: DecisionTreeModel = DecisionTree.train(data, Algo.Classification, Entropy, maxDepth)


    val decisonTotalCorrect: Double = data.map { point: LabeledPoint =>
      if(decisonModel.predict(point.features) == point.label) 1 else 0
    }.sum()

    //进行精确度
    val decisonAccuracy: Double = decisonTotalCorrect / data.count()

    println("decison Smodel进行精确度")
    println(BigDecimal.valueOf(decisonAccuracy))

  }
}
