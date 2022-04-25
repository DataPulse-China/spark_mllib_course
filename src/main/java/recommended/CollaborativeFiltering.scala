package recommended

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD

/**
 * Collaborative filtering 协同过滤推荐算法
 *    一、方法简介
 *
 *            协同过滤是一种基于一组兴趣相同的用户或项目进行的推荐，它根据邻居用户(与目标用户兴趣相似的用户)
 *        的偏好信息产生对目标用户的推荐列表。关于协同过滤的一个经典的例子就是看电影。如果你不知道哪一部
 *        电影是自己喜欢的或者评分比较高的，那么通常的做法就是问问周围的朋友，看看最近有什么好的电影推荐。
 *        而在问的时候，肯定都习惯于问跟自己口味差不多的朋友，这就是协同过滤的核心思想。因此，协同过滤是
 *        在海量数据中挖掘出小部分与你品味类似的用户，在协同过滤中，这些用户成为邻居，然后根据他们喜欢的
 *        东西组织成一个排序的目录推荐给你（如下图所示）。
 *        http://dblab.xmu.edu.cn/blog/wp-content/uploads/2017/01/user-item.png
 *
 *            协同过滤算法主要分为基于用户的协同过滤算法和基于项目的协同过滤算法。MLlib当前支持基于模型
 *        的协同过滤，其中用户和商品通过一小组隐语义因子进行表达，并且这些因子也用于预测缺失的元素。
 *        Spark MLlib实现了 交替最小二乘法 (ALS) 来学习这些隐性语义因子。
 *
 *    二、隐性反馈 vs 显性反馈
 *            显性反馈行为包括用户明确表示对物品喜好的行为，隐性反馈行为指的是那些不能明确反应用户喜好的行为。
 *       在许多的现实生活中的很多场景中，我们常常只能接触到隐性的反馈，例如页面游览，点击，购买，喜欢，分享等等。
 *
 *           基于矩阵分解的协同过滤的标准方法，一般将用户商品矩阵中的元素作为用户对商品的显性偏好。在 MLlib 中
 *       所用到的处理这种数据的方法来源于文献： Collaborative Filtering for Implicit Feedback Datasets 。
 *       本质上，这个方法将数据作为二元偏好值和偏好强度的一个结合，而不是对评分矩阵直接进行建模。因此，评价就不是
 *       与用户对商品的显性评分，而是与所观察到的用户偏好强度关联起来。然后，这个模型将尝试找到隐语义因子来预估
 *       一个用户对一个商品的偏好。
 *
 *
 *
 */
object CollaborativeFiltering {
  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.OFF)
    val sc = new SparkContext("local[*]", "li")

    /**
     *  三、示例

                下面代码读取spark的示例文件，文件中每一行包括一个用户id、商品id和评分。我们使用默认的ALS.train()
            方法来构建推荐模型并评估模型的均方差。
     */
    //读取数据
    val data: RDD[String] = sc.textFile("/home/rjxy/IdeaProjects/spark/spark_mllib_course/src/main/resources/mllib/als/test.data")

    //Rating中的第一个int是user编号，第二个int是item编号，最后的double是user对item的评分。
    val value: RDD[Rating] = data.map(
      (item: String) => {
        item.split(",") match {
          case Array(user, ite, rate) => Rating(user.toInt, ite.toInt, rate.toDouble)
        }
      }
    )

    //构建模型  划分训练集和测试集，比例分别是0.8和0.2。

    val dataArray: Array[RDD[Rating]] = value.randomSplit(Array(0.8, 0.2))

    val modelData: RDD[Rating] = dataArray(0)
    val testData: RDD[Rating] = dataArray(1)

    println("\n模型用的数据")
    modelData.foreach(println)

    //指定参数值，然后使用ALS训练数据建立推荐模型：

    val rank = 10
    val iterationsnum = 20
    val lambda: Double = 0.001
    val implicitPrefs = true
    /**
     * 在 MLlib 中的实现有如下的参数:

          numBlocks 是用于并行化计算的分块个数 (设置为-1，为自动配置)。
          rank 是模型中隐语义因子的个数。
          iterations 是迭代的次数。
          lambda 是ALS的正则化参数。
          implicitPrefs 决定了是用显性反馈ALS的版本还是用适用隐性反馈数据集的版本。
          alpha 是一个针对于隐性反馈 ALS 版本的参数，这个参数决定了偏好行为强度的基准。
     */
    val model: MatrixFactorizationModel = ALS.train(modelData, rank, iterationsnum, lambda)

    /**
     *      可以调整这些参数，不断优化结果，使均方差变小。比如：iterations越多，lambda较小，均方差会较小，
     *  推荐结果较优。上面的例子中调用了 ALS.train(ratings, rank, numIterations, 0.01) ，我们还
     *  可以设置其他参数，调用方式如下：
     */

    val model2: MatrixFactorizationModel = new ALS()
      //要使用的特性数量 因子数量
      .setRank(rank)
      //设置迭代
      .setIterations(iterationsnum)
      //ALS的正则化参数
      .setLambda(lambda)
      //设置隐含的首选项  决定了是用显性反馈ALS的版本还是用适用隐性反馈数据集的版本。
      .setImplicitPrefs(implicitPrefs)
      //设置并行计算的乘积块的数量。
      .setProductBlocks(2).run(modelData)

    //从 test训练集中获得只包含用户和商品的数据集 ：
    val value1: RDD[(Int, Int)] = testData.map(i => (i.user, i.product))

    //使用model训练好的推荐模型对用户商品进行预测评分，得到预测评分的数据集：
    val testPredicts: RDD[((Int, Int), Double)] = model.predict(value1)
      .map(
        (item: Rating) => {
          ((item.user, item.product), item.rating)
        }
      )

    //使用model2训练好的推荐模型对用户商品进行预测评分，得到预测评分的数据集：
    val testPredicts2: RDD[((Int, Int), Double)] = model2.predict(value1)
      .map(
        (item: Rating) => {
          ((item.user, item.product), item.rating)
        }
      )

    //将真实评分数据集与预测评分数据集进行合并。这里，Join操作类似于SQL的inner join操作，返回结果是前面和后面集合中配对成功的，过滤掉关联不上的。
    val value2: RDD[((Int, Int), Double)] = testData.map(i => ((i.user, i.product), i.rating))

    //率和预测
    val ratesAndPredict1: RDD[((Int, Int), (Double, Double))] = testPredicts.join(value2)
    val ratesAndPredict2: RDD[((Int, Int), (Double, Double))] = testPredicts2.join(value2)

    println("\nmodel将真实评分数据集与预测评分数据集进行合并")
    ratesAndPredict1.foreach(println)

    println("\nmode2将真实评分数据集与预测评分数据集进行合并")
    ratesAndPredict2.foreach(println)

    // 我们把结果输出，对比一下真实结果与预测结果：
    /*
      model将真实评分数据集与预测评分数据集进行合并
      ((1,3),(2.506444094838034,5.0))
      ((3,1),(1.3323196840061964,1.0))
      ((4,1),(1.3323196840061964,1.0))

      mode2将真实评分数据集与预测评分数据集进行合并
      ((3,1),(0.011756400428011293,1.0))
      ((4,1),(0.011756400428011293,1.0))
      ((1,3),(0.017207996598748942,5.0))
     */

    //比如，第一条结果记录((3,1),(1.0,-0.22756397347958202))中，(3,1)分别表示3号用户和1号商品，而1.0是实际的估计分值，-0.22756397347958202是经过推荐的预测分值。

    // 然后计算均方差，这里的r1就是真实结果，r2就是预测结果：
    val MSE1: Double = ratesAndPredict1.map(
      i => {
        val err: Double = i._2._2 - i._2._1
        err * err
      }
    ).mean()

    val MSE2: Double = ratesAndPredict2.map(
      i => {
        val err: Double = i._2._2 - i._2._1
        err * err
      }
    ).mean()
    println("MSE1:   " + MSE1)
    println("MSE2:   " + MSE2)


    sc.stop()

  }
}
