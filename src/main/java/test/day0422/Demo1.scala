package test.day0422

import breeze.linalg.functions.cosineDistance
import breeze.linalg.{DenseVector, norm}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

/**
 * 协同过滤

    1.	根据u.data数据集，将数据集以7：3的比例分为训练数据和测试数据
    使用最小交替二乘法完成模型训练，因子为100，其他参数任意
    （a）输出用户和商品因子数量
    （b）通过余弦相似度找到与电影id为617相似度最高的10部电影，并结合电影数据		集（u.item）输出该结果10部电影的信息（余弦相似度方法自定义）
    （c）为用户303推荐10部电影
    （d）将训练的模型预测数据与源数据做笛卡尔积，求根方差和均方差误根，均方差和均方差误根<1.5为模型准确度（均方差和均方差误根方法自定义）
    （e）根据训练模型输出测试数据集预测结果
    （f）根据mllib的内置函数求均方差和均方差误根
 */
object Demo1 {
  def main(args: Array[String]): Unit = {
    //根据u.data数据集，将数据集以7：3的比例分为训练数据和测试数据
    Logger.getLogger("org").setLevel(Level.OFF)
    val session: SparkSession = SparkSession.builder().config(new SparkConf().setMaster("local[*]").setAppName("asd")).getOrCreate()
    val context = session.sparkContext

    val value: RDD[Rating] = context.textFile("/home/rjxy/IdeaProjects/spark/spark_mllib_course/src/main/resources/ml-100k/u.data")
      .map(
        item => {
          val strings: Array[String] = item.split("\t")
          // userid (用户🆔id) 、item id (项目id) 、 rating(评分)、timestamp(日期时间)
          Rating(strings(0).toInt, strings(1).toInt, strings(2).toInt)
        }
      )

    val array: Array[RDD[Rating]] = value.randomSplit(Array(0.7, 0.3), 1l)


//    array.foreach((item: RDD[Rating]) => item.foreach(println))

    val v1: RDD[Rating] = array(0)
    val v2: RDD[Rating] = array(1)


    /**
     *    训练一个矩阵分解模型，给定用户对产品子集的评级RDD。
     * 评级矩阵近似为给定秩的两个低秩矩阵(特征数)的乘积。
     * 为了解决这些问题，ALS会根据“评级”中的分区数量，以一定的并行度迭代运行。
     *
        numBlocks 是用于并行化计算的分块个数 (设置为-1，为自动配置)。
        rank 是模型中隐语义因子的个数。
        iterations 是迭代的次数。
        lambda 是ALS的正则化参数。
        implicitPrefs 决定了是用显性反馈ALS的版本还是用适用隐性反馈数据集的版本。
        alpha 是一个针对于隐性反馈 ALS 版本的参数，这个参数决定了偏好行为强度的基准。
     */

    //使用最小交替二乘法完成模型训练，因子为100，其他参数任意  矩阵分解模型
    val model: MatrixFactorizationModel = ALS.train(v1, 100, 10, 0.01)

    //可以调整这些参数，不断优化结果，使均方差变小。比如：  iterations越多，  lambda较小，  均方差会较小，  推荐结果较优。
    // 上面的例子中调用了 ALS.train(ratings, rank, numIterations, 0.01) ，我们还可以设置其他参数，调用方式如下：

    println("训练模型用户因子数量  "+model.productFeatures.count())
    println("商品因子数量  "+model.userFeatures.count())

    //测试原数据
    val testRating: RDD[((Int, Int), Double)] = v1.map(
      (item: Rating) => {
        ((item.user, item.product), item.rating)
      }
    )
    //测试去掉评分的数据
    val test: RDD[(Int, Int)] = v1.map(
      (item: Rating) =>{
        (item.user,item.product)
      }
    )

    //第一条结果记录((3,1),(1.0,-0.22756397347958202))中，(3,1)分别表示3号用户和1号商品，而1.0是实际的估计分值，-0.22756397347958202是经过推荐的预测分值。
    val ratingsAndPredictions: RDD[((Int, Int), (Double, Double))] = model.predict(test).map((item: Rating) =>
      ((item.user, item.product), item.rating)
    ).join(testRating)

    println(" (user,product) 预测结果 与 真实结果 10个 ")
    ratingsAndPredictions.take(10).foreach(println)

    val value1: RDD[Double] =  ratingsAndPredictions.map {
      case ((user, pro), (r1, r2)) =>
        //计算方差
        val err: Double = r1 - r2
        err * err
    }
//    println("均方差   "+ value1.mean())
    println("自定义均方误差  (Mean squared error) (MSE) "+ BigDecimal.valueOf(value1.sum()/value1.count()))
    println("自定义均方根误差 (Root Mean squared error) (RMSE)   "+ BigDecimal.valueOf(math.sqrt(value1.sum()/value1.count())))

    //样本均值
    val testMean: Double = testRating.map(_._2).mean()
    //样本均方差
    val testSD: Double = testRating.map(
      (item: ((Int, Int), Double)) => {
        math.pow(item._2 - testMean, 2)
      }
    ).mean()

    //开根
    println("自定义均方差 (Standard Deviation)   "+ BigDecimal.valueOf(math.sqrt(testSD)))

    //通过余弦相似度找到与电影id为617相似度最高的10部电影，并结合电影数据集（u.item）输出该结果10部电影的信息（余弦相似度方法自定义）
    //余弦相似度

    val productFeatures: RDD[(Int, Array[Double])] = model.productFeatures

    val moviehead: Array[Double] = productFeatures.lookup(617).head

    val movie617: DenseVector[Double] = DenseVector(moviehead)

    val value2: RDD[(Int, Double)] = productFeatures.map(
      (item: (Int, Array[Double])) => {
        // cosineDistance(a,b) = (a dot b)/(norm(a) * norm(b))
        val itemVector: DenseVector[Double] = DenseVector(item._2)

        (item._1, itemVector.dot(movie617) / (norm(itemVector) * norm(movie617)))
      }
    )
//    guangwang
    /**
     * implicit
     *    //
     *    dot: OpMulInner.Impl2[T, U, Double],
     *    //
          normT: norm.Impl[T, Double],
          //
          normU: norm.Impl[U, Double]
     */
//    cosineDistance
//      .cosineDistanceFromDotProductAndNorm[RDD[(Int, Array[Double])], DenseVector[Double]](
//        dot = OpMulInner(productFeatures,movie617),
//        normT = norm(productFeatures),
//        normU = norm(movie617))

    //cosineDistance  余弦距离  余弦相似度  CosineSimilarity

    val value3: RDD[(Int, Double)] = productFeatures.map(
      i => {
        (i._1,cosineDistance(DenseVector(i._2), movie617))
      }
    )
    //所有电影(id : name)
    val movie: collection.Map[String, String] = context.textFile("/home/rjxy/IdeaProjects/spark/spark_mllib_course/src/main/resources/ml-100k/u.item")
      .map(_.split("\\|")).map(item => (item(0),item(1))).collectAsMap()

    println("（b）通过余弦相似度找到与电影id为617相似度最高的10部电影，并结合电影数据集（u.item）输出该结果10部电影的信息（余弦相似度方法自定义）")
    value2.sortBy(-_._2).take(11).tail.foreach(
      (item: (Int, Double)) => println(item._1+"   "+ item._2+ "      "+movie(item._1.toString))
    )
    println("（2222222222222222 过余弦相似度找到与电影id为617相似度最高的10部电影，并结合电影数据集（u.item）输出该结果10部电影的信息（余弦相似度方法自定义）")
    value3.sortBy(-_._2).take(11).tail.foreach(
      (item: (Int, Double)) => println(item._1+"   "+ item._2+ "      "+movie(item._1.toString))
    )


    //（c）为用户303推荐10部电影
    println("为用户303推荐10部电影")

    model.recommendProducts(303,10).foreach(
      (item: Rating) => {
        println(item+"  电影名字"+ movie(item.product.toString))
      }
    )


    /**
        为用户303推荐10部电影
        Rating(303,197,5.555111063968932)  电影名字Graduate, The (1967)
        Rating(303,272,5.271673086383508)  电影名字Good Will Hunting (1997)
        Rating(303,523,5.174819241439634)  电影名字Cool Hand Luke (1967)
        Rating(303,212,5.163752858040008)  电影名字Unbearable Lightness of Being, The (1988)
        Rating(303,59,5.09228998893902)  电影名字Three Colors: Red (1994)
        Rating(303,23,5.05293043296605)  电影名字Taxi Driver (1976)
        Rating(303,246,5.0341458527101794)  电影名字Chasing Amy (1997)
        Rating(303,318,5.031931490739359)  电影名字Schindler's List (1993)
        Rating(303,47,5.027232925900102)  电影名字Ed Wood (1994)
        Rating(303,479,5.019819060187922)  电影名字Vertigo (1958)
     */


    //输出测试数据集预测结果
    println("输出测试数据集预测结果")
    val testPredictRes: RDD[Rating] = model.predict(v2.map(item => {
      (item.user, item.product)
    }))

    testPredictRes.take(10).foreach(println)
    println("输出测试数据集预测结果结束")
    import session.implicits._
    val frame: DataFrame = testPredictRes.toDF().as("a").join(v2.toDF().as("b"),Seq("user","product") )//col("a.user") === col("b.user") && col("a.product") === col("b.product")

    val testDataSetPredictionResults: RegressionMetrics = new RegressionMetrics(
      frame.rdd.map {
        case Row(user, product, rating, rating2) => (rating.toString.toDouble, rating2.toString.toDouble)
      }
    )

    println("testDataSetPredictionResults  ")
    println("testDataSetPredictionResults  根据mllib的内置函数求  均方误差(MSE) 和 均方根误差(RMSE) 和")
    println("testDataSetPredictionResults  均方误差   "+BigDecimal.valueOf(testDataSetPredictionResults.meanSquaredError))//均方误差
    println("testDataSetPredictionResults  平均绝对误差   "+BigDecimal.valueOf(testDataSetPredictionResults.meanAbsoluteError))//平均绝对误差
    println("testDataSetPredictionResults  方差   "+BigDecimal.valueOf(testDataSetPredictionResults.explainedVariance))//方差
    println("testDataSetPredictionResults  均方根误差   "+BigDecimal.valueOf(testDataSetPredictionResults.rootMeanSquaredError))//均方根误差



//    cosineDistance.cosineDistanceFromDotProductAndNorm


      //根据mllib的内置函数求 均方误差(MSE) 和 均方根误差(RMSE) 和 均方差(Standard Deviation)

    //回归指标
    val regressionMetrics = new RegressionMetrics(
      ratingsAndPredictions
        .map {
          case ((user, product), (predicted, actual)) => (predicted, actual)
        }
    )
    println("根据mllib的内置函数求  均方误差(MSE) 和 均方根误差(RMSE) 和")
    println("均方误差   "+BigDecimal.valueOf(regressionMetrics.meanSquaredError))//均方误差
    println("平均绝对误差   "+BigDecimal.valueOf(regressionMetrics.meanAbsoluteError))//平均绝对误差
    println("方差   "+BigDecimal.valueOf(regressionMetrics.explainedVariance))//方差
    println("均方根误差   "+BigDecimal.valueOf(regressionMetrics.rootMeanSquaredError))//均方根误差


    val regressionMetrics2 = new RegressionMetrics(
      v1.map(
        (item) => {
          (item.rating,item.rating )
        }
      )
    )

    println("根据mllib的内置函数求   均方差(Standard Deviation)")
    println("方差   "+BigDecimal.valueOf(regressionMetrics2.explainedVariance))//方差
    println("方差开根 = 均方差(Standard Deviation)"+BigDecimal.valueOf(math.sqrt(regressionMetrics2.explainedVariance)))//方差

//            1.1258227931225748
//    方差开根 1.1258227931225138
    context.stop()


  }
}
