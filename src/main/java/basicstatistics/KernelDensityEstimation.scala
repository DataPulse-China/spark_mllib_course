package basicstatistics

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.stat.KernelDensity
import org.apache.spark.rdd.RDD

/**
 * KernelDensityEstimation  核密度估计
 *    单词
 *    Kernel 内核
 *    Density 密度
 *    Estimation 估计
 *
 *      Spark ML 提供了一个工具类 KernelDensity 用于核密度估算，核密度估算的意思是根据已知的样本估计未知的密度，
 * 属於非参数检验方法之一。核密度估计的原理是。观察某一事物的已知分布，如果某一个数在观察中出现了，
 * 可认为这个数的概率密度很大，和这个数比较近的数的概率密度也会比较大，而那些离这个数远的数的概率密度会比较小。
 * Spark1.6.2版本支持高斯核(Gaussian kernel)
 *
 */
object KernelDensityEstimation {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    val sc = new SparkContext("local[*]", "li")

    val test: RDD[Double] = sc.textFile("/home/rjxy/IdeaProjects/spark/spark_mllib_course/src/main/resources/data/iris.data")
      .map(_.split(","))
      //取第一列
      .map(p => p(0).toDouble)

    //用样本数据构建核函数，这里用假设检验中得到的iris的第一个属性的数据作为样本数据进行估计：
    //其中setBandwidth表示高斯核的宽度，为一个平滑参数，可以看做是高斯核的标准差。
    val value = new KernelDensity().setSample(test).setBandwidth(3.0)

    //构造了核密度估计kd，就可以对给定数据数据进行核估计：
    value.estimate(Array(-1.0,2.0,5.0,6.1)).foreach(
      println
    )
    //这里表示的是，在样本-1.0, 2.0, 5.0, 5.8等样本点上，其估算的概率密度函数值分别是：0.011372003554433524, 0.059925911357198915, 0.12365409462424519, 0.12816280708978114。


  }
}
