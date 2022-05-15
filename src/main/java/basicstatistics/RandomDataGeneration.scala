package basicstatistics

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.random.{LogNormalGenerator, PoissonGenerator, RandomDataGenerator, RandomRDDs, UniformGenerator}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * 随机数生成 Random data generation
 *
 *      RandomRDDs 是一个工具集，用来生成含有随机数的RDD，可以按各种给定的分布模式生成数据集，
 *  Random RDDs包下现支持正态分布、泊松分布和均匀分布三种分布方式。
 *  RandomRDDs提供随机double RDDS或vector RDDS。
 *
 */

  object RandomDataGeneration {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    val session: SparkSession = SparkSession.builder().master("local[*]").appName("name").getOrCreate()
    val context: SparkContext = session.sparkContext
    //生成1000000个服从正态分配N(0,1)的RDD[Double]，并且分布在 10 个分区中：
    val value: RDD[Double] = RandomRDDs.normalRDD(context, 3  , 10)
    println("\n生成3个服从正态分配N(0,1)的RDD[Double]，并且分布在 10 个分区中：")
    value.foreach(println)
    //把生成的随机数转化成N(1,4) 正态分布：
    println("\n把生成的随机数转化成N(1,4) 正态分布：")
    value.map(_*2.0+1).foreach(println)

    //new PoissonGenerator(10)  DeveloperApi::从给定平均值的泊松分布中生成i. id样本
    println("\nnew PoissonGenerator(10)  DeveloperApi::从给定平均值的泊松分布中生成i. id样本")
    /*
    new PoissonGenerator(100 平均数 ),2  数量)
    例子 100 2
    113.0
    101.0

    例子 100 2
    86.0
    105.0
    108.0
    99.0
    92.0
     */
    RandomRDDs.randomRDD(context,new PoissonGenerator(100),5).foreach(println)

    //mean 对数正态分布的均值。   std 对数正态分布的标准差
    println("\n LogNormalGenerator 日志正常发电机  从给定平均值和标准偏差的对数正态分布中生成i. id样本 mean 对数正态分布的均值。   std 对数正态分布的标准差")
    val logNormalGenerator: RDD[linalg.Vector] = RandomRDDs.randomVectorRDD(context, new LogNormalGenerator(3, 1), 5, 5)
    logNormalGenerator.foreach(println)

    //随机数发生器 UniformGenerator
    println("\n随机数发生器 UniformGenerator")
    val value1: RDD[linalg.Vector] = RandomRDDs.randomVectorRDD(context, new UniformGenerator(), 3, 3)
    value1.foreach(println)
    /*import session.implicits._
    val list: List[Array[Double]] = unit.collect().map(_.toArray).toList k值准确率
    val frame: DataFrame = session.sparkContext.parallelize(list).toDF()*/

    //随即向量
    println("\n随即向量")
    RandomRDDs.normalVectorRDD(context,3,3,2).foreach(println)

    //七、核密度估计 Kernel density estimation


    context.stop()
  }
}
