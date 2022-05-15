package dimensionalityreduction

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.{Matrix, Vectors}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.rdd.RDD

object a {
  def main(args: Array[String]): Unit = {

    println(BigDecimal.valueOf(100000.123e2))
    println(BigDecimal.valueOf(2.594214957851726E-6))


    Logger.getLogger("org").setLevel(Level.OFF)
    val context = new SparkContext("local[*]", "li")

    val value: RDD[linalg.Vector] = context.parallelize(Seq(
      Vectors.dense(1, 2, 3, 4, 5)
      ,Vectors.dense(1, 21, 3, 4, 5)
    ))

    val matrix: RowMatrix = new RowMatrix(value)
    matrix.rows.foreach(println)
    println(matrix.numCols())
    println(matrix.numRows())

    val matrix1: Matrix = matrix.computePrincipalComponents(3)
    matrix1.rowIter.foreach(println)
    val matrix2: RowMatrix = matrix.multiply(matrix1)
    println("-------")
    matrix2.rows.foreach(println)




  }
}
