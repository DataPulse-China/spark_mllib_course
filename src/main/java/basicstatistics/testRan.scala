package basicstatistics

import dimensionalityreduction.TrainingDimensionalityReductionModel.approxEqual
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.random.{LogNormalGenerator, PoissonGenerator, RandomRDDs}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object testRan {
  def main(args: Array[String]): Unit = {
    println(approxEqual(Array(1.0, 2.0, 3.0), Array(1.0, 2.0, 3.0)))
    println(BigDecimal.valueOf(1e9))
    Logger.getLogger("org").setLevel(Level.OFF)
    val session: SparkSession = SparkSession.builder().master("local[*]").appName("name").getOrCreate()
    val context: SparkContext = session.sparkContext
    println("\n LogNormalGenerator 日志正常发电机  从给定平均值和标准偏差的对数正态分布中生成i. id样本 mean 对数正态分布的均值。   std 对数正态分布的标准差")
    val logNormalGenerator: RDD[linalg.Vector] = RandomRDDs.randomVectorRDD(context, new LogNormalGenerator(4, 1), 50, 1)
    logNormalGenerator.foreach(
      (i: linalg.Vector) => {
       println(BigDecimal.valueOf(i.toArray(0)))
      }
    )
  }
}
