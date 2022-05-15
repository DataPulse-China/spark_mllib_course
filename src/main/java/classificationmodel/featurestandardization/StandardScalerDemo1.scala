package classificationmodel.featurestandardization

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.feature.{StandardScaler, StandardScalerModel}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object StandardScalerDemo1 {
  def main(args: Array[String]): Unit = {
    val conf: SparkConf = new SparkConf().setMaster("local[2]").setAppName("Classification")
    val builder: SparkSession = new SparkSession.Builder().config(conf).getOrCreate()

    val sc: SparkContext = builder.sparkContext
    import builder.implicits._
    // get StumbleUpon dataset 'https://www.kaggle.com/c/stumbleupon'
    val records = sc.textFile("/home/rjxy/IdeaProjects/spark/spark_mllib_course/src/main/resources/stumbleupon/train_noheader.tsv").map(line => line.split("\t"))
    //.map(records => (records(0), records(1), records(2)))

    //records.foreach(println)


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

    // 特征标准化 feature standardization
    val vectors: RDD[linalg.Vector] = data.map(lp => lp.features)
    //特征数据
    val matrix: RowMatrix = new RowMatrix(vectors)
    val matrixSummary: MultivariateStatisticalSummary = matrix.computeColumnSummaryStatistics()
    println(matrixSummary.mean)
    println(matrixSummary.min)
    println(matrixSummary.max)
    println(matrixSummary.variance)
    println(matrixSummary.numNonzeros)



    val scaler: StandardScaler = new StandardScaler(true,true)

    //特征数据 训练得到的模型 标准缩放器
    val scalerModel: StandardScalerModel = scaler.fit(vectors)

    //锻炼的模型进行数据缩放
    val scaledData = data.map(lp => LabeledPoint(lp.label,scalerModel.transform(lp.features)))

    println(data.first.features)
    println(scaledData.first.features)



    /*

      val scaler = new StandardScaler(withMean = true, withStd = true).fit(vectors)
      val scaledData = data.map(lp => LabeledPoint(lp.label,
      scaler.transform(lp.features)))

     */
    println((0.789131 - 0.4122580529952672) / math.sqrt(0.10974244167559001))

  }
}
