package ml100k

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}


object Test1 {
  def main(args: Array[String]): Unit = {

    val context = SparkContext.getOrCreate(new SparkConf().setAppName("li").setMaster("local[*]"))

    // 数据的字段
    // userid (用户🆔id) 、item id (项目id) 、 rating(评分)、timestamp(日期时间)
    val value: RDD[String] = context.textFile("/home/rjxy/IdeaProjects/spark/spark_mllib_course/src/main/resources/ml-100k/u.data")

    value.take(10).foreach(println)

    //获取字段的状态
    //- count 计数
    //- mean 平均
    //- stdev 标准偏差
    //- max 最大值
    //- min 最小值
    println("UserId"+value.map(_.split("\t")(0).toDouble).stats())
    println("ItemId"+value.map(_.split("\t")(1).toDouble).stats())
    println("rateId"+value.map(_.split("\t")(2).toDouble).stats())
    println("timestamp"+value.map(_.split("\t")(3).toDouble).stats())
    println("timestamp"+value.map(_.split("\t")(3).toDouble).stats())


    //


   context.stop()

  }
}
