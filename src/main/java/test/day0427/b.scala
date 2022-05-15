package test.day0427

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.sql.{DataFrame, Row, SparkSession, functions}
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}

import java.util

object b {
  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.WARN)
    val session: SparkSession = SparkSession.builder.master("local[*]").appName("TestAPP").enableHiveSupport().getOrCreate()

    val structType: StructType = StructType(List(StructField("科目", StringType), StructField("姓名", StringType), StructField("分数", IntegerType)))

    val dataList = new util.ArrayList[Row]()
    dataList.add(Row("数学", "张三", 88))
    dataList.add(Row("英语", "张三", 77))
    dataList.add(Row("语文", "张三", 92))
    dataList.add(Row("数学", "王五", 65))
    dataList.add(Row("语文", "王五", 87))
    dataList.add(Row("英语", "王五", 90))
    dataList.add(Row("数学", "李雷", 67))
    dataList.add(Row("语文", "李雷", 33))
    dataList.add(Row("英语", "李雷", 24))
    dataList.add(Row("数学", "宫九", 77))
    dataList.add(Row("语文", "宫九", 87))
    dataList.add(Row("英语", "宫九", 90))
    dataList.add(Row("英语", "宫九s", 90))

    import session.implicits._

    val frame: DataFrame = session.createDataFrame(dataList, structType)

    val strings: Array[String] = frame.selectExpr("`科目`").distinct().rdd.map(_ (0).toString).collect()

    frame
    new OneHotEncoder()

    frame.groupBy("姓名")  //  分组字段
      .pivot("科目", strings) // 列转行字段
      .agg(functions.first($"分数"))
      .show()

    session.stop()
  }
}
