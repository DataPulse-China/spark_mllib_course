package basicstatistics

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.evaluation.{RankingMetrics, RegressionMetrics}
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.stddev
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

/**
 * K 值平均准确率
 *
 *
 */
object KValueAccuracy {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    val session: SparkSession = SparkSession.builder().master("local[*]").appName("name").getOrCreate()
    val context: SparkContext = session.sparkContext

    val value: RDD[String] = context.textFile("/home/rjxy/IdeaProjects/spark/spark_mllib_course/src/main/resources/ml-100k/u.data")

    //取出 用户评分数据：包含4个字段:userid (用户🆔id) 、item id (项目id) 、 rating(评分)、timestamp(日期时间)
    //这里只需要三个
    /**
     *准备ALS训练数据
     * 返回Rating格式数据类型
     */
    val userData:RDD[Rating]  = value.map(
      (item: String) => {
        val strings: Array[String] = item.split("\t").take(3)
        //        Rating()
        Rating(strings(0).toInt, strings(1).toInt, strings(2).toDouble)
      }
    )

    import session.implicits._
    //矩阵分解模型
    val matrixFactorizationModel: MatrixFactorizationModel = ALS.train(userData, 10, 10, 0.01)

    val frame: DataFrame = userData.toDF()
    frame.select(stddev("TIME_LAG"))
      .show()

    frame.na

    val ratings: Array[Rating] = matrixFactorizationModel.recommendProducts(789, 10)
    val value1: RDD[(Int, Int)] = userData.map { case Rating(user, product, rating) => (user.toString.toInt, product.toString.toInt) }
    val testPredict: RDD[Rating] = matrixFactorizationModel.predict(value1)

    println("789 推荐数据")
    ratings.foreach(println)


    println("789 实际数据")
    val data2: RDD[Rating] = userData.filter((_: Rating).user == 789)
    data2.foreach(println)

    val d: Double = avgPrecisionK(data2.map(_.product).collect(), ratings.map((_: Rating).product), 10)
    println(d)


    //实际数据 测试数据
    //@param predictionAndObservations an RDD of (prediction, observation) pairs  由(预测,观察)对组成的RDD
    val value2: RDD[(Double, Double)] = testPredict.toDF().as("a").join(data2.toDF().as("b"), $"a.user" === $"b.user" and $"a.product" === $"b.product").rdd.map { case Row(a, b, c, d, e, f) => (c.toString.toDouble, f.toString.toDouble) }
    val regressionMetrics = new RegressionMetrics(value2)
    value2.toDF().show()
    println("均方根误差  "+regressionMetrics.rootMeanSquaredError)
    println("解释的方差  "+regressionMetrics.explainedVariance)
    println("平均绝对误差  "+regressionMetrics.meanAbsoluteError)
    println("均方误差  "+regressionMetrics.meanSquaredError)

    /**
     * 789 sai    wei 塞维
      Rating(789,320,10.560373375480532)
      Rating(789,1006,9.979988426404729)
      Rating(789,962,9.480997605549806)
      Rating(789,361,8.909733644832372)
      Rating(789,74,8.505563162863439)
      Rating(789,668,7.860591920777541)
      Rating(789,703,7.7688367828333735)
      Rating(789,865,7.657816687173675)
      Rating(789,374,7.648705489649051)
      Rating(789,1598,7.537320690722787)
     */

//    new RankingMetrics()
  }

  /* Function to compute average precision given a set of actual and predicted ratings */
  // Code for this function is based on: https://github.com/benhamner/Metrics
  def avgPrecisionK(actual: Seq[Int], predicted: Seq[Int], k: Int): Double = {
    val predK: Seq[Int] = predicted.take(k)
    var score = 0.0
    var numHits = 0.0
    for ((p, i) <- predK.zipWithIndex) {
      if (actual.contains(p)) {
        numHits += 1.0
        score += numHits / (i.toDouble + 1.0)
      }
    }
    if (actual.isEmpty) {
      1.0
    } else {
      score / scala.math.min(actual.size, k).toDouble
    }
  }
}
