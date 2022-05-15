package test.day0502

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

object Demo1 {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    val session: SparkSession = SparkSession.builder().master("local[*]").appName("li")
      .config("dfs.client.use.datanode.hostname","true")
      .getOrCreate()

    val value: DataFrame = session.read.format("LIBSVM").load("/home/rjxy/IdeaProjects/spark/spark_mllib_course/src/main/resources/part-00000")

    println("1")
    value.show(2,false)
    println("2")

    val kmeans: KMeans = new KMeans().setK(10).setSeed(1L).setMaxIter(50)


    //聚类模型
    val model: KMeansModel = kmeans.fit(value)

    val frame: DataFrame = model.transform(value)

    //预测
    val rows: Array[Row] = frame.take(10)
    rows.foreach(println)
    val value1: RDD[(String, Iterable[Int])] = frame.rdd.map {
      case Row(label, features, prediction) => {
        (prediction.toString, 1)
      }
    }.groupByKey()

    value1.map(
      item => {
        (item._1,item._2.size)
      }
    ).foreach(i => {println("【"+i._1+"类--------"+i._2+"】")})

    //集合内误差平方和

    val WSSSEItems: Double = model.computeCost(value)
    println(WSSSEItems)

    println("Items - Cluster Centers: ")
    model.clusterCenters.foreach(println)

    session.stop()
  }
}
