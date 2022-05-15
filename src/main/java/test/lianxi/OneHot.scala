package test.lianxi

import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, StringIndexerModel, VectorAssembler}
import org.apache.spark.ml.linalg
import org.apache.spark.mllib
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, SparkSession}

object OneHot {
  def main(args: Array[String]): Unit = {
    val session = SparkSession.builder().appName("a").master("local[*]").enableHiveSupport().getOrCreate()
    import session.implicits._



    val student: DataFrame = session.sparkContext.parallelize(Seq(
    ("英语", 100, "女"),
    ("语文", 10, "女"),
    ("数学", 19, "女"),
    ("论语", 150, "男"),
    ("英语", 60, "男"),
    ("数学", 90, "男")
    )).toDF("kemu","score","sex")


    val indexer: StringIndexer = new StringIndexer()
    val encoder: OneHotEncoder = new OneHotEncoder()

    val model: StringIndexerModel = indexer.setInputCol("kemu").setOutputCol("kemuVec").fit(student)

    val frame = indexer.setInputCol("sex").setOutputCol("sexVec").fit(student).transform(model.transform(student))

    val kemuFrame = encoder.setInputCol("kemuVec").setOutputCol("kemuHot").transform(frame)

    val function: UserDefinedFunction = udf((v: linalg.Vector) => {
      println(v)
      v.toArray
    })



//    breeze.linalg.Vector


//    val vector: linalg.Vector = Vectors.dense(Array(1D))

    val frame1: DataFrame = new VectorAssembler().setInputCols(Array("kemuHot", "sexVec", "kemuVec")).setOutputCol("VectorAssembler")
      .transform(kemuFrame)
    frame1.printSchema()
    frame1.rdd.foreach(println)

    val frame2 = frame1
      .withColumn("VectorAssembler", function.apply(col("VectorAssembler")))

//    frame2



  }
}
