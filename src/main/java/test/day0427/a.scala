package test.day0427

import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object a {
  def main(args: Array[String]): Unit = {

    val session: SparkSession = SparkSession.builder().appName("a").master("local[*]").getOrCreate()
    val value: RDD[Int] = session.sparkContext.parallelize(Seq((1), (2)))
    val value1: RDD[MatrixEntry] = value.map(
      i => {
        MatrixEntry(i, i, 1.0)
      }
    )
    import session.implicits._
    value.toDF()
    val matrix = new CoordinateMatrix(value1)
    println(matrix.toBlockMatrix().toLocalMatrix())
    println(matrix.numRows())

  }
}
