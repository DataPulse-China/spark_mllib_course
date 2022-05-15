package test.day0427

import breeze.linalg.DenseVector
import breeze.linalg.functions.cosineDistance
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

object Count4 {
  def main(args: Array[String]): Unit = {
    System.setProperty("HADOOP_USER_NAME","root")
    Logger.getLogger("org").setLevel(Level.OFF)
    val session: SparkSession = SparkSession.builder()
      .enableHiveSupport()
      .appName("li")
      .master("local[*]")
      .config("dfs.client.use.datanode.hostname", "true")
      .config("hive.metastore.uris", "thrift://192.168.23.34:9083")
      .getOrCreate()
    import session.implicits._

    val partData: DataFrame = session.table("dwd.fact_part_machine_data")

    val data: Dataset[(Int, linalg.Vector)] = partData.map(
      item => {
        val Arrlen: Array[String] = item.toString().replace("[", "").replaceAll("\\]", "").split(",")
        (item.get(0).toString.toInt, Vectors.dense(Arrlen.slice(5, Arrlen.length).map((_: String).toDouble)))
      }
    )

    data.show(false)
    data.toDF("partkey","vec").createTempView("temp")
    val frame: RDD[(Int, DenseVector[Double], Int, DenseVector[Double])] = session.sql(
      """
        |select
        |a.partkey as ak,
        |a.vec as av,
        |b.partkey bk,
        |b.vec as bv
        | from
        |temp as a
        | cross join
        |  temp as b
        |on b.partkey in (64031,98854,33830,162260,166390) and a.partkey not in (64031,98854,33830,162260,166390)
        |""".stripMargin).rdd.map(
      i => {
        (i.get(0).toString.toInt,
          DenseVector(i.get(1).toString.replace("[", "").replaceAll("\\]", "").split(",").map(_.toDouble)),
          i.get(2).toString.toInt,
          DenseVector(i.get(3).toString.replace("[", "").replaceAll("\\]", "").split(",").map(_.toDouble))
        )
      }
    )
    val value: RDD[(Int, Double)] = frame.map {
      case (ak: Int, av: DenseVector[Double], bk: Int, bv: DenseVector[Double]) =>
        (ak, 1 - cosineDistance(av, bv))
    }

    value.toDF("partkey","cons").createTempView("o")
    session.sql(
      """
        |select partkey,avg(cons)  as cons from
        |o
        |group by partkey order by cons desc limit 5
        |""".stripMargin).show()



  }
}
