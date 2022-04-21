package basicstatistics

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.{Matrix, Vectors}
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.rdd.RDD

/**
 *
 *    相关性Correlations
 *
 *      Correlations，相关度量，目前Spark支持两种相关性系数：皮尔逊相关系数（pearson）和斯皮尔曼等级相关系数（spearman）。
 *  相关系数是用以反映变量之间相关关系密切程度的统计指标。简单的来说就是相关系数绝对值越大（值越接近1或者-1）,
 *  当取值为0表示不相关，取值为(0~-1]表示负相关，取值为(0, 1]表示正相关。
 *      Pearson相关系数表达的是两个数值变量的线性相关性, 它一般适用于正态分布。其取值范围是[-1, 1], 当取值为0表示不相关，
 *  取值为[-1~0)表示负相关，取值为(0, 1]表示正相关。
 *
 *  公式  http://mocom.xmu.edu.cn/blog/58482eb8e083c990247075aa.png
 *
 *      Spearman相关系数也用来表达两个变量的相关性，但是它没有Pearson相关系数对变量的分布要求那么严格，
 *  另外Spearman相关系数可以更好地用于测度变量的排序关系。其计算公式为：
 *
 *  公式  http://mocom.xmu.edu.cn/blog/58482eb3e083c990247075a9.png
 */
object CorrelationsDemo {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val sc = new SparkContext("local[*]", "")

    /**
     *    根据输入类型的不同，输出的结果也产生相应的变化。如果输入的是两个RDD[Double]，则输出的是一个double类型的结果；
     * 如果输入的是一个RDD[Vector]，则对应的输出的是一个相关系数矩阵。具体操作如下所示：
     */

    // 接下来我们先从数据集中获取两个series，这两个series要求必须数量相同，这里我们取莺尾花的前两个属性：
    val seriesX: RDD[Double] = sc.textFile("/home/rjxy/IdeaProjects/spark/spark_mllib_course/src/main/resources/data/iris.data")
      .map(_.split(",")).map(p => p(0).toDouble)

    val seriesY: RDD[Double] = sc.textFile("/home/rjxy/IdeaProjects/spark/spark_mllib_course/src/main/resources/data/iris.data")
      .map(_.split(",")).map(p => p(1).toDouble)


    // 然后，我们调用Statistics包中的corr()函数来获取相关性，这里用的是"pearson" 皮尔逊，当然根据不同需要也可以选择"spearman" 斯皮尔曼等级：
    val seriesCorrelation: Double = Statistics.corr(seriesX, seriesY, "pearson")
    println(seriesCorrelation+" double")

    //-0.1594565184858299 spearman
    //-0.10936924995064387 pearson 说明数据集的前两列，即花萼长度和花萼宽度具有微小的  负相关性  。

    //  上面介绍了求两个series的相关性，接下来介绍一下如何求  [相关系数矩阵]。
    // 方法是类似的，首先还是先从数据集中获取一个RDD[Vector]，为了进行对照，我们同样选择前两个属性：

    val data: RDD[linalg.Vector] = sc.textFile("/home/rjxy/IdeaProjects/spark/spark_mllib_course/src/main/resources/data/iris.data")
      .map(_.split(",")).map(p => Vectors.dense(p(0).toDouble, p(1).toDouble))

    //我们调用Statistics包中的corr()函数，这里也同样可以选择"pearson"或者"spearman"，得到相关系数矩阵：
    //默认使用 皮尔逊
    val dataCorrelation: Matrix = Statistics.corr(data)

    println(dataCorrelation+"矩阵")

  }
}
