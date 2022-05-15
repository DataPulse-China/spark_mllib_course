package test.day0427

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.functions.{col, lit}
import org.apache.spark.sql.{Column, DataFrame, Dataset, Row, SaveMode, SparkSession, functions}

object Count2 {
  def main(args: Array[String]): Unit = {

    System.setProperty("HADOOP_USER_NAME","root")
    Logger.getLogger("org").setLevel(Level.ERROR)
    val session: SparkSession = SparkSession.builder()
      .enableHiveSupport()
      .appName("li")
      .master("local[*]")
      .config("dfs.client.use.datanode.hostname", "true")
      .config("hive.metastore.uris", "thrift://192.168.23.34:9083")
      .getOrCreate()
    val frame: DataFrame = session.table("ods.part")
    import session.implicits._

    case class mfgr(mfgr:String)
    case class brand(brand:String)

    frame.select('mfgr).distinct().show()
    val brand2: Array[String] = frame.select('brand).distinct().sort("brand").rdd.map(_(0).toString).collect()
    val mfgr2: Array[String] = frame.select('mfgr).distinct().sort("mfgr").rdd.map(_(0).toString).collect()


    val frame1: DataFrame = frame.groupBy("partkey", "mfgr", "brand", "size", "retailprice")
      .pivot("mfgr", mfgr2)
      .agg(lit(1))
      .na.fill(0)

    val strings: Seq[String] = frame1.columns.toSeq
    val columns: Seq[Column] = strings.map(col(_))



    val frame2: DataFrame = frame1.groupBy(columns: _*)
      .pivot("brand", brand2)
      .agg(lit(1))
      .na.fill(0)

    frame2.show(false)

//
    frame2.write.mode(SaveMode.Overwrite).saveAsTable("dwd.fact_part_machine_data")

  }
}
