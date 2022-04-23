package algorithm

import breeze.linalg.DenseVector
import breeze.linalg.functions.cosineDistance
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession

/**
 * 余弦相似度
 *
 *
 *
 */
object CosineSimilarity {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
//    val session: SparkSession = SparkSession.builder().master("local[*]").appName("l").getOrCreate()
//    val context: SparkContext = session.sparkContext

    val a: DenseVector[Double] = DenseVector(1, 2, 3, 4)
    val b: DenseVector[Double] = DenseVector(1, 2, 3, 15)

    val d: Double = 1- cosineDistance(a, b)
    println(d)
  }
}
