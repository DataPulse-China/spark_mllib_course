package classificationandregression

import org.apache.log4j.Level
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer, VectorIndexerModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

object Hour {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder
      .appName("BikeSharing")
      .master("local[1]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    // 读取 csv 文件
    val df: DataFrame = spark.read.format("csv").option("header", "true")
      .load("/home/rjxy/IdeaProjects/spark/spark_mllib_course/src/main/resources/BikeSharing/hour.csv")

    df.cache()
    df.createTempView("BikeSharing")
    println(df.count())
    spark.sql("SELECT * FROM BikeSharing").show()

    // 丢弃 record id、 date、casual 和 registered 列
    val df1 = df.drop("instant")
      .drop("dteday")
      .drop("casual")
      .drop("registered")
    df1.printSchema()

    // 将各列转为 Double 类型
    val df2 = df1.withColumn("season", df1("season").cast("double"))
      .withColumn("yr", df1("yr").cast("double"))
      .withColumn("mnth", df1("mnth").cast("double"))
      .withColumn("hr", df1("hr").cast("double"))
      .withColumn("holiday", df1("holiday").cast("double"))
      .withColumn("weekday", df1("weekday").cast("double"))
      .withColumn("workingday", df1("workingday").cast("double"))
      .withColumn("weathersit", df1("weathersit").cast("double"))
      .withColumn("temp", df1("temp").cast("double"))
      .withColumn("atemp", df1("atemp").cast("double"))
      .withColumn("hum", df1("hum").cast("double"))
      .withColumn("windspeed", df1("windspeed").cast("double"))
      .withColumn("label", df1("label").cast("double"))
    df2.printSchema()

    // 丢弃 label 并创建特征向量
    val df3 = df2.drop("label")

    //获取dataframe的所有特征列
    val featureCols: Array[String] = df3.columns

    //特征转换器
    val vectorAssembler: VectorAssembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("rawFeatures")

    //特征转换器 转换后的数据
    val vectorAssemblerFrame = vectorAssembler.transform(df3)

    vectorAssemblerFrame.show(false)

    //用于在“向量”数据集中索引分类特征列的类。
    /*
        VectorIndexer 会根据
      不同的值来确定哪些特征应该为类别型。类别型的分类最多有 maxCategories 种。
     */
    val vectorIndexer = new VectorIndexer()
      .setInputCol("rawFeatures")
      .setOutputCol("features")
      //设置向量 最大分类 类别 数量 后面视为连续值
      .setMaxCategories(4)


    val vectorIndexerModel: VectorIndexerModel = vectorIndexer.fit(vectorAssemblerFrame)

    val frame1: DataFrame = vectorIndexerModel
      .transform(vectorAssemblerFrame)


    frame1.show(false)
  }
}
