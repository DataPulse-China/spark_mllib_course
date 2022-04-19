package ml100k

import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}


object Test2 {
  def main(args: Array[String]): Unit = {
    val context = SparkContext.getOrCreate(new SparkConf().setAppName("li").setMaster("local[*]"))

    // 数据的字段
    // userid (用户🆔id) 、item id (项目id) 、 rating(评分)、timestamp(日期时间)
    val value: RDD[String] = context.textFile("/home/rjxy/IdeaProjects/spark/spark_mllib_course/src/main/resources/ml-100k/u.data")

    var a: RDD[Array[String]] = value.map(
      item => {
        val strings = item.split("\t").take(3)
        strings
//        Rating()
      }
    )

    /**
     *准备ALS训练数据
     * 返回Rating格式数据类型
     */
    //
    val ratingsRDD: RDD[Rating] = a.map {
      case Array(user, movie, rating) => Rating(user.toInt, movie.toInt, rating.toDouble)
    }

    /**
     * 使用ALS.train命令进行训练模型
     */
    //参数
    //ratings  数据类型 – 具有用户 ID、产品 ID 和评级的Rating对象的 RDD
    //rank[分解的参数] – 要使用的功能数量 指的是当我们矩阵分解Matrix Factorization 时，将原本矩阵A(m x n)分解成 X(m x rank)[用户特征] 矩阵与 Y(rank x n)[产品特征]矩阵
    //迭代 – ALS 的迭代次数 ALS算法重复执行次数
    //lambda – 正则化参数 建议值0.01

    //矩阵分解模型
    val matrixFactorizationModel: MatrixFactorizationModel = ALS.train(ratingsRDD, 10, 10, 0.01)

    /**
     * 01这里我们已经训练完成模型，接下来将使用此模型进行推荐
     */
    //我们可以针对每一位会员，定期发送短信或E-mail或在会员登录时，向会员推荐可能会感兴趣的电影
    //针对用户推荐电影，我们可以使用model.recommendProducts方法来推荐
    println(
      //进行推荐
      matrixFactorizationModel
      //推荐产品方法
       //参数
       //user 要被推荐的用户id
       //num  推荐的记录数
       //返回 Array[Rating] 数组中每一条都是系统
      .recommendProducts(196, 5).mkString("\n"))

    //返回结果
    // 推荐用户id 196  产品id 1093 推荐评分8.717295625887079
    //Rating(196,1093,8.717295625887079)
    //Rating(196,1426,8.62397896114282)
    //Rating(196,464,8.449810620478454)
    //Rating(196,1207,8.42618890896982)
    //Rating(196,1160,8.410600806799987)

    /**
     * 02查看预测一个用户对一个产品的评级。  predict 预测
     */
    println(matrixFactorizationModel.predict(196, 1093))
    println(matrixFactorizationModel.predict(196, 1093))


    /**
     * 03针对电影推荐给用户 matrixFactorizationModel.recommendUsers
     */
    println(matrixFactorizationModel
      //输入参数product,针对此product推荐给可能有兴趣的用户
      .recommendUsers(464, 5).mkString("\n"))
    //其意义是针对电影ID 464 推荐给用户153 推荐评分是13.203886960140531
    //Rating(153,464,13.203886960140531)
    //Rating(333,464,9.60028524712623)
    //Rating(866,464,9.53347122435875)
    //Rating(97,464,9.44858373374393)
    //Rating(604,464,9.378408042553756)

    /**
     * 显示推荐的电影名称
     */
    /**
     * 01创建电影id与名称的对照表
     */
    val itemRdd: RDD[String] = context.textFile("/home/rjxy/IdeaProjects/spark/spark_mllib_course/src/main/resources/ml-100k/u.item")
    val movieitle: collection.Map[Int, String] = itemRdd
      //使用map方法针对每一条数据进行转换；每一条数据以|符号分割字段，并以take取出前两个字段
      .map(line => line.split("\\|").take(2))
      //江上一个命令读取的2个字段进行转换，array(0).toInt 电影id   array(1)电影名称
      .map(array => (array(0).toInt, array(1)))
      //使用collectAsMap 创建movieTitle电影id / 名称 对照表
      .collectAsMap()

    /**
     * 显示对照表的前5条数据 brightness
     */
    movieitle.take(5).foreach(println)

    //获取制定的电影名称
    println("获取指定的电影名称 146："+movieitle(146))


    /**
     * 显示前五条推荐的电影名称
     */
    matrixFactorizationModel.recommendProducts(196,5)
      .map(rating => (rating.product,movieitle(rating.product),rating.rating)).foreach(println)



    context.stop()
  }
}
