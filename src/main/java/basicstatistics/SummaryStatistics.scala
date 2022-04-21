package basicstatistics

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.rdd.RDD

/**
 * 概括统计  summary statistics [摘要统计]
 *
 * 单词
 *  linalg 分开
 *  linear + algebra: 线性代数
 *
 */
object SummaryStatistics {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val sc = new SparkContext("local[*]", "")

    val observations: RDD[linalg.Vector] = sc.parallelize(
      Seq(
        Vectors.dense(1.0, 10.0, 100.0),
        Vectors.dense(2.0, 20.0, 200.0),
        Vectors.dense(3.0, 30.0, 300.0)
      )
    )
    /**
     * 对于RDD[Vector]类型的变量，
     *    Spark MLlib提供了一种叫colStats()的统计方法，调用该方法会返回一个类型为MultivariateStatisticalSummary的实例。
     * 通过这个实例看，我们可以获得每一列的最大值，最小值，均值、方差、总数等。具体操作如下所示：
     */
    val summary: MultivariateStatisticalSummary = Statistics.colStats(observations)
    println("包含每列平均值的稠密向量\n"+summary.mean)  // a dense vector containing the mean value for each column  包含每列平均值的稠密向量
    println("列方差\n"+summary.variance)  // column-wise variance 列方差
    println("每列中的非零数\n"+summary.numNonzeros)  // number of nonzeros in each colum  每列中的非零数

    //读取要分析的数据，把数据转变成RDD[Vector]类型：
    val flower: RDD[linalg.Vector] = {
      //读取本地数据
      sc.textFile("/home/rjxy/IdeaProjects/spark/spark_mllib_course/src/main/resources/data/iris.data")
        //转换数组
      .map((_: String).split(","))
        //转换RDD Vector
      .map((item: Array[String]) => {
        //稠密向量 double
        val strings: Array[String] = item.filter(
          item =>{
            if (!item.equals("Iris-setosa") && !item.equals("Iris-versicolor") && !item.equals("Iris-virginica")  )  true else  false
          }
        )
        Vectors.dense(strings.map(_.toDouble))
      })
    }
    //上面我们就把莺尾花的四个属性，即萼片长度，萼片宽度，花瓣长度和花瓣宽度存储在 flower 中，类型为 RDD[linalg.Vector] 。
    // 然后，我们调用colStats()方法，得到一个MultivariateStatisticalSummary类型的变量：

    val summaryFlower: MultivariateStatisticalSummary = Statistics.colStats(flower)

    println( "  样本大小  "+summaryFlower.count )
    println( "  样本均值向量  "+summaryFlower.mean )
    println( "  样本方差向量。应该返回一个零向量，如果样本大小是1  "+summaryFlower.variance )
    println( "  每列的最大值  "+summaryFlower.max )
    println( "  每列的最小值  "+summaryFlower.min )
    println( "  每列的L1范数  "+summaryFlower.normL1 )
    println( "  每一列的欧几里得大小  欧几里德距离   "+summaryFlower.normL2 )
    println( "  每列中非零元素的数目(包括显式显示的零值)  "+summaryFlower.numNonzeros )


  }
}
