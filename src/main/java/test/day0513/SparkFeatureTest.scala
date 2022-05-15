package test.day0513

import breeze.linalg.DenseVector
import breeze.linalg.functions.cosineDistance
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, StringIndexerModel}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.linalg
import org.apache.spark.sql.expressions.{UserDefinedFunction, Window}
import org.apache.spark.sql.functions.{array, col, desc_nulls_first, lit, regexp_replace, row_number, udf}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SaveMode, SparkSession}

import scala.collection.mutable

object SparkFeatureTest {
  System.setProperty("HADOOP_USER_NAME","root")
  val session = SparkSession.builder()
    .config("hive.metastore.uris", "thrift://master:9083")
    .config("dfs.client.use.datanode.hostname", "true")
    .master("local[*]").appName("li")
    .enableHiveSupport()
    .getOrCreate()

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    val context = session.sparkContext

    println("1111")
//        getOne()
    println("2222")
    //    getTwo()
    println("3333")
//    getThree()
    println("cosinedistance")
    getThreeCosineDistance()
  }

  def getOne(){
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

    frame1.show()

    val frame: DataFrame = session.sql(
      """
        |select
        |o.custkey,partkey,i.index
        |from
        |dwd.fact_orders as o
        |join
        |dwd.fact_lineitem as count
        |on
        |o.orderkey = count.orderkey
        |join
        |index as i
        |on
        |i.custkey = o.custkey
        |order by cast(o.custkey as int)
        |""".stripMargin)
    frame.createTempView("temp")

    val frame2: DataFrame = session.sql(
      """
        |select
        |t1.index index1,
        |t1.partkey,t2.index as index2
        |from
        |temp t1
        |join
        |temp as t2
        |on
        |t1.partkey = t2.partkey and t1.index != t2.index
        |""".stripMargin)
    println(frame2.count())
    frame2.show()

    val matrixEntry: RDD[MatrixEntry] = frame2.map {
      case Row(a, b, c) => MatrixEntry(a.toString.toLong, c.toString.toLong, 1.0)
    }.rdd

    val iterator: Seq[Array[Double]] = new CoordinateMatrix(matrixEntry).toBlockMatrix().toLocalMatrix().transpose.rowIter.map(_.toArray).toSeq
    val value: RDD[Array[Double]] = session.sparkContext.parallelize(iterator)


    val value1: RDD[String] = value.map(_.mkString(","))
    value.foreach(println)
//    value1.toDF().repartition(1).write.mode(SaveMode.Overwrite)text("hdfs://master:9000/li/")



  }

  def getThree(): Unit ={
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
        |""".stripMargin).sort("index")

    //220用户矩阵索引 0
    val Index220: Int = frame1.select(col("index")).where(col("custkey") === "220").take(1)(0)(0).toString.toInt
    println("220用户矩阵索引 0 "+Index220)

    //特征
    val frame: Dataset[String] = session.read.textFile("hdfs://master:9000/li/part-00000-00b79795-19c0-4025-bab3-72d588782c19.txt")

//    array(frame.columns.map(col): _*).as("value")

    //用户特征向量 特征
    val features: DataFrame = frame
      .select((row_number().over(Window.orderBy(lit(0))) - 1).as("index"),
        col("value")
      )
      .as("a")
      .join(frame1.as("b"), col("a.index") === col("b.index"))
      .select("custkey", "value")

    val userData: RDD[(Int, DenseVector[Double])] = features.rdd.map {
      case Row(custkey, value) => {
        (custkey.toString.toInt, DenseVector(value.toString.split(",").map(_.toDouble)))
      }
    }

    val userData220: (Int, DenseVector[Double]) = userData.filter(_._1 == 220).first()

    import session.implicits._
    val Top10: Dataset[Row] = userData.map(
      (item: (Int, DenseVector[Double])) => {
        (item._1, cosineDistance(item._2, userData220._2))
      }
    ).sortBy(_._2, false).toDF().limit(10)

    //  所有产品特征
    var allPartkey: DataFrame = session.read.table("dwd.fact_part_machine_data")
      //regexp_replace(col("mfgr"), "Manufacturer#", "")
      //匹配mfgr 将Manufacturer#字段 匹配成 null 最后还命名为mfgr
      .withColumn("mfgr", regexp_replace(col("mfgr"), "Manufacturer#", ""))
      .withColumn("brand", regexp_replace(col("brand"), "Brand#", ""))
      .drop(col("name"))
    allPartkey = allPartkey.select(col("partkey"),array(allPartkey.columns.map(col):_*).as("value"))

    allPartkey.show(false)

    //所有的partkey 对应的 custkey
    val partCustFrame: DataFrame = session.table("dwd.fact_orders").as("o")
      .join(session.table("dwd.fact_lineitem").as("l"), col("o.orderkey") === col("l.orderkey")).select(col("custkey"), col("partkey"))

    //当前用户的 partkey
    val userFrame = partCustFrame.where(col("custkey") === userData220._1)
      .select(col("custkey"), col("partkey"))

    //top10用户的 key
    val top10Frame = partCustFrame.as("part")
      .join(Top10.toDF("custkey", "cons").as("top10"), col("top10.custkey") === col("part.custkey"))
      .select(col("top10.custkey"), col("partkey"))

    //user购买的part 特征
    val user220part: DataFrame = allPartkey.as("all").join(userFrame.as("user"), col("all.partkey") === col("user.partkey"))
      .select(col("all.partkey"), col("all.value"))

    //所有和user top10购买的part 特征
    val top100part: DataFrame = allPartkey.as("all")
      .join(top10Frame.as("user"), col("all.partkey") === col("user.partkey"))
      .select(col("all.partkey"), col("all.value"))
      .as("all2")
      .join(
        user220part.as("user2"),col("user2.partkey") =!= col("all2.partkey")
      ).selectExpr("all2.partkey","all2.value","'key' as key")



//    user220part.write.saveAsTable("dwd.user220part")
    top100part.repartition(1).write.partitionBy("key").mode(SaveMode.Overwrite).saveAsTable("dwd.top100part2")

    println(top100part.count())
    //finnalDate
  /*  val finnalDate: RDD[(String, Array[String], Double)] = user220part.as("user").join(top100part.as("top"),col("user.partkey") =!= col("top.partkey") ).rdd.map {
      case Row(a: String, b: Array[String], c: String, d: Array[String]) => (
        (
          a, b,
          //余弦相似度
          cosineDistance(DenseVector(b.map(_.toDouble)), DenseVector(d.map(_.toDouble)))
        )
        )
    }*/

//    finnalDate.foreach(println)



//    features.show(false)

  }

  def getThreeCosineDistance(): Unit ={
    val frame = session.table("dwd.user220part")
    import session.implicits._
    val value: RDD[(String, String, Double)] = session.table("dwd.top100part2").as("top").select(col("partkey"), col("value"))
      .join(frame.as("user"), col("user.partkey") =!= col("top.partkey"))
      .selectExpr("user.partkey as userpart","user.value as uservalue", "top.partkey as toppart","top.value as topvalue")
      .rdd.map {
      case Row(a: String, b: mutable.WrappedArray[String], c: String, d: mutable.WrappedArray[String]) => (
          (
            a,c,
            //余弦相似度
            cosineDistance(DenseVector(b.map(_.toDouble).array), DenseVector(d.map(_.toDouble).array))
          )
        )
    }

    value.sortBy(_._3,false).toDF("userpart","toppart","cosineValue").write.mode(SaveMode.Overwrite).saveAsTable("dwd.tuijianpart")

    value.sortBy(_._3,false).repartition(1).foreach(i =>(println(BigDecimal.valueOf(i._3))))
  }

  def getTwo(): Unit ={

    val frame = session.table("ods.part")
    val indexer = new StringIndexer()
    val encoder = new OneHotEncoder()
    
    val brandIndexermodel = indexer
      //输入列
      .setInputCol("brand")
      //输出列
      .setOutputCol("brandIndex")
      //锻炼 品牌 索引器
      .fit(frame)
    
    val mfgrIndexermodel: StringIndexerModel = indexer
      //输入列
      .setInputCol("mfgr")
      //输出列
      .setOutputCol("mfgrIndex")
      //锻炼 厂商 索引器
      .fit(frame)
    
    //使用 品牌索引器 训练数据
    val brandFrame: DataFrame = brandIndexermodel.transform(frame)
    
    //使用 厂商索引器 训练数据
    val indexFrame = mfgrIndexermodel.transform(brandFrame)

    indexFrame.show(false)
    val frame1 = encoder.setInputCol("mfgrIndex").setOutputCol("mfgrOneHota").setDropLast(false).transform(
      indexFrame
    )
    frame1.show(false)
    //厂商转换之后那数据 brand  进行转换 one hot
    val encoderFrame: DataFrame = encoder.setInputCol("brandIndex").setOutputCol("brandOneHota").transform(frame1)

    val a: UserDefinedFunction = session.udf.register("a", (i: linalg.Vector) => {
      i.toArray
    })

    val function: UserDefinedFunction = udf((i: linalg.Vector) => {
        i.toArray
    })



    var frame2 = encoderFrame
      .withColumn("mfgrOneHota", function.apply(col("mfgrOneHota")))
      .withColumn("brandOneHota", a.apply(col("brandOneHota")))

    val brandindex: Array[(String, Int)] = brandIndexermodel.labels.zipWithIndex
    val mfgrindex: Array[(String, Int)] = mfgrIndexermodel.labels.zipWithIndex
    mfgrindex.foreach(println)
    mfgrindex.foreach{
      case (field, index)=> frame2 = frame2.withColumn(field,col("mfgrOneHota")(index))
    }
    brandindex.foreach{
      case (field, index)=> frame2 = frame2.withColumn(field,col("brandOneHota")(index))
    }

    frame2
      .drop("name")
      .drop("type")
      .drop("container")
      .drop("comment")
      .drop("etldate")
      .drop("brandIndex")
      .drop("mfgrIndex")
      .drop("mfgrOneHota")
      .drop("brandOneHota")
      .sort(col("partkey"),col("size"))
      .write.saveAsTable("dwd.fact_part_machine_data")


  }
}
