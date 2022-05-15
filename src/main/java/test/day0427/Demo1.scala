package test.day0427

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.{Matrices, Vectors}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, IndexedRow, IndexedRowMatrix, MatrixEntry}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SaveMode, SparkSession}

import scala.collection.mutable.ArrayBuffer

object Demo1 {
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


    val frame1: DataFrame = session.sql(
      """
        |select
        |custkey,(row_number() over(order by cast(custkey as int)) -1) as index
        |from
        |ods.orders as o
        |join
        |ods.lineitem as count
        |on
        |o.orderkey = count.orderkey
        |group by custkey
        |order by cast(custkey as int) desc
        |""".stripMargin)
    frame1.createTempView("index")
    frame1.show()

    frame1.rdd.collect().foreach(println)

    val frame: DataFrame = session.sql(
    """
        |select
        |o.custkey,partkey,i.index
        |from
        |ods.orders as o
        |join
        |ods.lineitem as count
        |on
        |o.orderkey = count.orderkey
        |join
        |index as i
        |on
        |i.custkey = o.custkey
        |order by cast(o.custkey as int)
        |""".stripMargin)

    frame.createTempView("temp")
    frame.show()

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
    val count: Long = frame1.count()

    val rows = new ArrayBuffer[Row](count.toInt)
    for (i <- 0 until count.toInt){
      rows.append(Row(i,0,0))
    }

    frame2.collect().foreach(
      (i: Row) => {
        rows.update(i(0).toString.toInt ,i)
      }
    )

    println("rows")
    rows.foreach(println)
    println("rows.size"+rows.size)

    import session.implicits._
    val data: RDD[Row] = session.sparkContext.parallelize(rows)
    println("datasize"+data.count())
    val value: RDD[MatrixEntry] = data.map {
      case Row(index, partkey, index2) => MatrixEntry(
        index.toString.toLong
        ,
        if (index2 == 0)
          count
        else
          index2.toString.toLong
        ,
        if (index2 == 0)
          0.0
        else
          1.0
      )
    }

    val matrix: CoordinateMatrix = new CoordinateMatrix(value)

    val iter: Seq[Array[Double]] = matrix.toBlockMatrix().toLocalMatrix().transpose.rowIter.toSeq.map((i: linalg.Vector) => i.toArray )

    val value1: RDD[String] = session.sparkContext.parallelize(iter,1).map((i: Array[Double]) => i.mkString(",") )

    val dataFrame: DataFrame = value1.toDF()
//    val allClumnName: String = dataFrame.columns.mkString(",")
//    val result: DataFrame = dataFrame.selectExpr(s"concat_ws(',',$allClumnName) as allclumn")

//    value1.toDF().write.mode(SaveMode.Overwrite)text("hdfs://192.168.23.34:9000/templi")

    println(matrix.numRows())
    println(frame1.count())
    val value2: RDD[IndexedRow] = matrix.toIndexedRowMatrix().rows.sortBy(_.index)
    value2.foreach((item: IndexedRow) => {
      println(item.index + item.vector.toDense.values.mkString("[","\t\t","]"))
    })



//    frame1.write..text("hdfs://192.168.23.34:9000/templi")

//    session.createDataFrame(entries).toDF().write.text("hdfs://192.168.23.34:9000/templi")

  }
}
