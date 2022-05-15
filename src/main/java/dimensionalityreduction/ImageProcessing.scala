package dimensionalityreduction

import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object ImageProcessing {
  def main(args: Array[String]): Unit = {
//    Logger.getLogger("org").setLevel(Level.OFF)
    val spConfig: SparkConf = (new SparkConf).setMaster("local[1]").setAppName("SparkApp").
      set("spark.driver.allowMultipleContexts", "true")
    val sc = new SparkContext(spConfig)
    val path = "/home/rjxy/IdeaProjects/spark/spark_mllib_course/src/main/resources" +  "/lfw/*"
    val rdd: RDD[(String, String)] = sc.wholeTextFiles(path)
    val first: (String, String) = rdd.first
    val files: RDD[String] = rdd.map { case (fileName, content) => fileName.replace("file:", "") }



    println(first)

    files.foreach(println)

  }
}
