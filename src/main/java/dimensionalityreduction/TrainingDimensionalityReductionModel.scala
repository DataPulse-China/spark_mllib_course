package dimensionalityreduction

import breeze.linalg.{DenseMatrix, DenseVector, csvwrite}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.feature.{StandardScaler, StandardScalerModel}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.{Matrix, SingularValueDecomposition, Vectors}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Training dimensionality reduction model 训练降维模型
 */
object TrainingDimensionalityReductionModel {
  val PATH = "/home/rjxy/IdeaProjects/spark/spark_mllib_course/src/main/resources"

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    val spConfig = (new SparkConf).setMaster("local[1]").setAppName("SparkApp").
      set("spark.driver.allowMultipleContexts", "true")
    val sc = new SparkContext(spConfig)
    val session = SparkSession.builder().config(spConfig).getOrCreate()
    val path = PATH + "/lfw/*"
    val rdd = sc.wholeTextFiles(path)
    val first = rdd.first
    val files = rdd.map { case (fileName, content) => fileName.replace("file:", "") }
    import session.implicits._

    //文件数量
    println(files.count)

    val aePath = PATH + "/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg"
    //加载图片
    val aeImage = Util.loadImageFromFile(aePath)

    //灰色图片
    val grayImage = Util.processImage(aeImage, 100, 100)
    import java.io.File
    import javax.imageio.ImageIO

    //图片写入
    ImageIO.write(grayImage, "jpg", new File("/tmp/aeGray.jpg"))

    //提取像素 灰色像素
    val pixels = files.map((f: String) => Util.extractPixels(f, 50, 50))
    println(pixels.take(10).map(_.take(10).mkString("", ",", ", ...")).mkString("\n"))

    //每个像素转成稠密向量
    val vectors: RDD[linalg.Vector] = pixels.map(p => Vectors.dense(p))

    println(vectors.count)

    //setName方法创建一个可在Spark Web UI中显示的人类可读的名称
    vectors.setName("image-vectors")

    // remember to cache the vectors to speed up computation
    vectors.cache

    //标准化模型
    val scaler: StandardScalerModel = new StandardScaler(withMean = true, withStd = false).fit(vectors)

    //数据进行标准化
    val scaledVectors: RDD[linalg.Vector] = vectors.map((v: linalg.Vector) => scaler.transform(v))
    //    vectors.take(10).foreach(println)
    //    println("------------")
    //    scaledVectors.take(10).foreach(println)

    //标准向量 转 行矩阵
    val matrix: RowMatrix = new RowMatrix(scaledVectors)
    println(scaledVectors.count())
    println(matrix.numRows())
    val k: Int = 10

    //行矩阵 计算主成分 主成分提取
    val pc: Matrix = matrix.computePrincipalComponents(k)

    pc.rowIter.foreach(println)
    println(pc.numRows)
    println(pc.numCols)

    //将降维 纬度后的矩阵本地持久化
    val value: DenseMatrix[Double] = new DenseMatrix(pc.numRows, pc.numCols, pc.toArray)
    //    csvwrite(new  File("/tmp/pc.csv"),value)

    //用矩阵乘法把图像矩阵和主成分矩阵相乘来实现投影
    val projected: RowMatrix = matrix.multiply(pc)

    /**
     * 这些以向量形式表示的投影后的数据可以作为另一个机器学习模型的输入。例如,我们可以
     * 通过使用这些投影后的脸的投影数据和一些没有脸的图像产生的投影数据,共同训练一个面部识
     * 别模型。另外也可以训练一个多分类识别器,每个人是一个类,从而创建一个识别某个输入脸是
     * 否是某个人的识别模型。
     */
    projected.rows.foreach(println)

    //PCA 和 SVD 模型的关系
    /**
     * 我们之前提到 PCA 和 SVD 有着密切的联系。事实上,可以使用 SVD 恢复出相同的主成分
     * 向量,并且应用相同的投影矩阵投射到主成分空间。
     *
     */
    //SVD
    val svd: SingularValueDecomposition[RowMatrix, Matrix] = matrix.computeSVD(k, computeU = true)
    println(s"U dimension: (${svd.U.numRows}, ${svd.U.numCols})")
    println(s"S dimension: (${svd.s.size}, )")
    println(s"V dimension: (${svd.V.numRows}, ${svd.V.numCols})")

    //比较svd 的 V 和 PCA投影数据
    //我们在一些数据上测试这个函数:
      println(approxEqual(Array(1.0, 2.0, 3.0), Array(1.0, 2.0, 3.0)))
    //来尝试另一组测试数据:
      println(approxEqual(Array(1.0, 2.0, 3.0), Array(3.0, 2.0, 1.0)))

    //可以这样使用我们的相等函数:
    println("这样使用我们的相等函数\n")
    println(approxEqual(svd.V.toArray, pc.toArray))

    //将SVD的S转向量
    val breezeS: DenseVector[Double] = breeze.linalg.DenseVector(svd.s.toArray)
    //将SVD的U的每一行转为向量 然后和 breezeS 点积 算出 和 PCA 中用来把原始图像数据投影到 10 个主成分构成的空间中的投影矩阵相等
    val projectedSVD: RDD[linalg.Vector] = svd.U.rows.map {
      v: linalg.Vector =>
        val breezeV: DenseVector[Double] = breeze.linalg.DenseVector(v.toArray)
        val multV: DenseVector[Double] = breezeV :* breezeS
        Vectors.dense(multV.data)
    }

    //连接两个 pca 的 10 个主成分构成的空间中的投影矩阵  和  SVD U*S 的点积 做
    val count: Long = projected.rows.zip(projectedSVD).map {
      case (v1: linalg.Vector, v2: linalg.Vector) =>
        //approxEqual 自定义 数据约等于 方法 判断
        approxEqual(v1.toArray, v2.toArray)
    }
      //总量
      .filter((_: Boolean) == true).count
    //运行结果是 1055,因此基本可以确定 PCA 投影后的每一行和 SVD 投影后的每一行相等。
    println(count)

    /**
     * 注意在前面的代码中,加粗的 :* 运算符表示对向量执行对应元素和元素的乘法。
     */

    /**
     *    PCA 和 SVD 都是确定性模型,就是对于给定输入数据,总可以产生确定结果的模型。这和我
      们之前看到的很多依赖一些随机素的模型不同(大部分是由模型的初始化权重向量等原因导致)

          这两个模型都可以返回多个主成分或者奇异值,因此控制模型的唯一参数就是 k。就像聚类
      模型一样,增加 k 总是可以提高模型的表现(对于聚类,表现在相对误差函数值;对于 PCA 和
      SVD,整体的不确定性表现在 k 个成分上)。因此,选择 k 的值需要折中,看是要包含尽量多的
      数据结构信息,还是要保持投影数据的低维度。
     */

//    在 LFW 数据集上估计 SVD 的 k 值
//    通过观察在我们的图像数据集上计算 SVD 得到的奇异值,可以确定每次运行中奇异值都相
//    同,并且是按照递减的顺序返回的,如下所示。
    val sValues = (1 to 5).map { i => matrix.computeSVD(i, computeU =
      false).s }
    sValues.foreach(println)
//    这会生成类似下面的输出:

    /*
      奇异值
      在降维时,奇异值能作为合适精度取舍的参考,让我们在时间和空间之间取得平衡。
      为了估算 SVD(和 PCA)做聚类时的 k 值,以一个较大的 k 的变化范围绘制一个奇异值图
      是很有用的。可以看到,每增加一个奇异值时增加的变化总量是否基本保持不变。
      首先计算最大的 300 个奇异值:
    */

    val svd300: SingularValueDecomposition[RowMatrix, Matrix] = matrix.computeSVD(300, computeU = false)
    val sMatrix: DenseMatrix[Double] = new DenseMatrix(1, 300, svd300.s.toArray)
    println(sMatrix)
    csvwrite(new File("/home/rjxy/IdeaProjects/spark/spark_mllib_course/src/main/resources/lfw/s.csv"), sMatrix)
    //可以看到,在 k 值的某个区间之后(本例中大概是 100),图形基本变平。这表明与某一 k
    //值相对应的奇异值(或者主成分)可能足以解释原始数据的变化。
    //图片 /home/rjxy/IdeaProjects/spark/spark_mllib_course/src/main/resources/lfw/sendpix34.jpg
  }

  /**
   * 矩阵 V 和 PCA 的结果完全一样(不考虑正负号和浮点数误差)。可以通过使用一个功能函数
   * 大致比较两个矩阵的向量数据来确定这一点:
   *
   * tolerance 公差 0.0000010
   */
  def approxEqual(array1: Array[Double], array2: Array[Double], tolerance: Double = 1e-6): Boolean = {
    // 注意这里略去了主成分的迹(Sign)
    val bools: Array[Boolean] = array1.zip(array2).map {
      case (v1, v2) =>
        if(math.abs(math.abs(v1) - math.abs(v2)) > tolerance)
          false
        else
          true
    }

    /**
     * fold 聚合操作
     *  初始值是true
     *    :fold()操作需要从一个初始值开始，并以该值作为上下文，处理集合中的每个元素。
     */
    bools.fold(true)((_: Boolean) & (_: Boolean))
  }


}
