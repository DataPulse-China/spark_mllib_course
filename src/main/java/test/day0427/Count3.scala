package test.day0427

import breeze.linalg.DenseVector
import breeze.linalg.functions.cosineDistance
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{col, lit, row_number}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SaveMode, SparkSession}

import java.util.Properties

object Count3 {
  def main(args: Array[String]): Unit = {
    System.setProperty("HADOOP_USER_NAME","root")
    Logger.getLogger("org").setLevel(Level.OFF)
    val session: SparkSession = SparkSession.builder()
      .enableHiveSupport()
      .appName("li")
      .master("local[*]")
      .config("dfs.client.use.datanode.hostname", "true")
      .config("hive.metastore.uris", "thrift://master:9083")
      .getOrCreate()
    import session.implicits._
    val frame1: DataFrame = session.sql(
      """
        |select
        |custkey,(row_number() over(order by cast(custkey as int)) -1) as index
        |from
        |dwd.fact_orders as o
        |join
        |dwd.fact_lineitem as count
        |on
        |o.orderkey = count.orderkey
        |group by custkey
        |order by cast(custkey as int) desc
        |""".stripMargin)
    frame1.createTempView("index")
    val index100051: Int = frame1.where("custkey = 220").rdd.map { case Row(user,index) => (index.toString.toInt) }.collect()(0)

    val value: DataFrame = session.read.textFile("hdfs://master:9000/li")
//      .selectExpr("*","row_number() over (order by value) - 1")
      .withColumn("index", row_number().over(Window.orderBy(lit(0))) - 1)

    value.show()
    println(index100051+"ssss")
    value.where(s"index = $index100051").show(false)
//  value.toDF().selectExpr("*","row_number() over (order by value)").show(false)

    val userData: RDD[(Int, DenseVector[Double], Double)] = value.rdd.map {
      case Row(value, index) =>
        val doubles: Array[Double] = value.toString.split(",").map(_.toDouble)
        //数量
        val productNum: Double = doubles.sum
        (index.toString.toInt, DenseVector(doubles), productNum)
    }.cache()

    //100051的 矩阵向量
    val vector100051: DenseVector[Double] = userData.filter(_._1 == index100051).first()._2

    val top10: RDD[(Int, Double)] = userData.map(
      (item: (Int, DenseVector[Double], Double)) => {
        (item._1, 1 - cosineDistance(item._2, vector100051) /*+ item._3*/)
      }
    ).sortBy(_._2, false)

    /**
       用户  相似度
    |custkey|cons|
    +-------+----+
    | 103517| 4.0|
    |  98983| 4.0|
    | 100087| 3.0|
    | 102113| 3.0|
    | 102784| 3.0|
    | 105115| 3.0|
    | 106843| 3.0|
    | 107161| 3.0|
    | 111446| 3.0|
    | 112667| 3.0|
    +-------+----+
     */

    top10
      .toDF("inde","cons")
      .as("a")
      .join(frame1.as("b"),col("a.inde") === col("b.index"))
      .select("custkey","cons")
      .orderBy(col("cons").desc,col("custkey"))
      .where(col("custkey") =!= 220).limit(10).toDF().createTempView("top10")

    session.table("top10").show()
    session.sql(
      """
        |select a.custkey,c.partkey from dwd.fact_orders a
        |join ods.lineitem  b join ods.part c
        |on
        |a.orderkey = b.orderkey
        |and
        |b.partkey = c.partkey
        |order by a.custkey asc
        |""".stripMargin).distinct().createTempView("part")

    session.table("part").show()

    val finaldata: Dataset[Row] = session.sql(
      """
        |select '220' as user , t.custkey,p.partkey ,t.cons from
        |top10 as t join part as p
        |on t.custkey = p.custkey
        |""".stripMargin).orderBy(col("cons").desc)
    finaldata.show()

    val properties = new Properties()
    properties.setProperty("user","root")
    properties.setProperty("password","123456")
    finaldata.write.jdbc("jdbc:mysql://master/shtd_store?useSSL=false","part_machine",properties)


/*
    val mysqlConf: Map[String, String] = Map("driver" -> "com.mysql.jdbc.Driver", "url" -> "jdbc:mysql://master/shtd_store?useSSL=false&useUnicode=true&characterEcoding=utf8", "user" -> "root", "password" -> "123456")

    //所有索引 矩阵 sum
    value.map(o => {
      //索引  相似度
      (o._1, 1.0 - cosineDistance(o._2, tmp) + o._3)
    }).toDF("id", "s")
      .as("a")
      .join(frame1.as("b"), col("a.id") === col("b.idx"))
      .select("custkey", "s")
      .orderBy(col("s").desc, col("custkey"))
      .where(col("custkey") =!= 100051)

      //10行
      .limit(10)
      //写
      .write
      .format("jdbc")
      .options(mysqlConf)
      .option("dbtable","part_machine")
      .mode(SaveMode.Overwrite)
      .save()

      */

  }
}
