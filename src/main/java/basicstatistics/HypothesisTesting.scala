package basicstatistics

import breeze.linalg.Matrix
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.{Matrices, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.mllib.stat.test.ChiSqTestResult
import org.apache.spark.rdd.RDD

/**
 *  HypothesisTesting  假设检验
 *
 *  Statistics 统计数据
 *
 *        Spark目前支持皮尔森卡方检测（Pearson’s chi-squared tests），
 *     包括“适配度检定”（Goodness of fit）以及“独立性检定”（independence）。
 *
 *
 *    一、基本原理

      在stat包中实现了皮尔逊卡方检验，它主要包含以下两类

      (1)适配度检验(Goodness of Fit test)：验证一组观察值的次数分配是否异于理论上的分配。

      (2)独立性检验(independence test) ：验证从两个变量抽出的配对观察值组是否互相独立
        (例如：每次都从A国和B国各抽一个人，看他们的反应是否与国籍无关)

 *
 */
object HypothesisTesting {
  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.ERROR)
    val sc = new SparkContext("local[*]", "")

    /**
     * 导入必要的包
     * import org.apache.spark.SparkContext
       import org.apache.spark.mllib.linalg._
       import org.apache.spark.mllib.regression.LabeledPoint
       import org.apache.spark.mllib.stat.Statistics._
     */

    /**
     *     接下来，我们从数据集中选择要分析的数据，比如说我们取出iris数据集中的前两条数据v1和v2。  5.1,  3.5,  1.4,0.2,Iris-setosa
     * 不同的输入类型决定了是做拟合度检验还是独立性检验。
     *  拟合度检验要求输入为 [Vector],
     *  独立性检验要求输入是 [Matrix]。
     */

      //iris数据集  take(2)
    val vectors: Array[linalg.Vector] = sc.textFile("/home/rjxy/IdeaProjects/spark/spark_mllib_course/src/main/resources/data/iris.data")
      .map(_.split(","))
      .map(p => Vectors.dense(p(0).toDouble, p(1).toDouble, p(2).toDouble, 20))
      .take(2)
    vectors.foreach(println)


    //1
    val v1: linalg.Vector = vectors.head
    //2
    val v2: linalg.Vector = vectors.last

    val vector: linalg.Vector = Vectors.dense(1, 2)


    /**
     * (一) 适合度检验 goodness of fit test
     *
     *       Goodness fo fit（适合度检验）：验证一组观察值的次数分配是否异于理论上的分配。
     *   其 H0假设（虚无假设，null hypothesis）为一个样本中已发生事件的次数分配会服从某个特定的理论分配。
     *   实际执行多项式试验而得到的观察次数，与虚无假设的期望次数相比较，检验二者接近的程度，
     *   利用样本数据以检验总体分布是否为某一特定分布的统计方法。通常情况下这个特定的理论分配指的是均匀分配，
     *   目前Spark默认的是均匀分配。
     *
     *    Statistics.chiSqTest(v1)  x平方分布检验
     */

    /**
     * Statistics.chiSqTest(:Vector)  对观察数据与均匀分布进行皮尔逊卡方拟合优度检验，每个类别的预期频率为1 /观察到的大小
     */
    val goodnessOfFitTest: ChiSqTestResult = Statistics.chiSqTest(v1)

    println("适合度检验 Goodness fo   fit\n\n"+goodnessOfFitTest)
    println("适合度检验 Goodness fo   fit"+BigDecimal.valueOf(goodnessOfFitTest.pValue))
    /**
     * 打印
     *     卡方检验摘要    Chi squared test summary:
     *
     *    method: pearson
          degrees of freedom = 3
          statistic = 5.588235294117647
          pValue = 0.1334553914430291
     *
     *     可以看到P值，自由度，检验统计量，所使用的方法，以及零假设等信息。我们先简单介绍下每个输出的意义：
          method: 方法。这里采用pearson方法。

          statistic： 检验统计量。简单来说就是用来决定是否可以拒绝原假设的证据。
            检验统计量的值是利用样本数据计算得到的，它代表了样本中的信息。
            检验统计量的绝对值越大，拒绝原假设的理由越充分，反之，不拒绝原假设的理由越充分。

          degrees of freedom：自由度。表示可自由变动的样本观测值的数目，

          pValue：统计学根据显著性检验方法所得到的P 值。一般以P < 0.05 为显著， P<0.01 为非常显著，
              其含义是样本间的差异由抽样误差所致的概率小于0.05 或0.01。
          一般来说，假设检验主要看P值就够了。在本例中pValue =0.133，说明两组的差别无显著意义。
          通过V1的观测值[5.1, 3.5, 1.4, 0.2]，无法拒绝其服从于期望分配（这里默认是均匀分配）的假设。

     *     No presumption against null hypothesis: observed follows the same distribution as expected..
     *     没有对零假设的假设:观察遵循与预期相同的分布。
     *
     */


    /**
     * （二）独立性检验 Indenpendence
     *
     *    卡方独立性检验是用来检验两个属性间是否独立。
     * 其中一个属性做为行，另外一个做为列，通过貌似相关的关系考察其是否真实存在相关性。
     * 比如天气温变化和肺炎发病率。
     * 首先，我们通过v1、v2构造一个举证Matrix，然后进行独立性检验：
     */

//    Statistics.chiSqTest(sc.parallelize(Seq(v1,v2)))
    println("\n ")
    val matrix: linalg.Matrix = Matrices.dense(2, 2, Array(v1(0),v1(1),v2(0),v2(1)))
    println("矩阵是按照列的顺序派列的")
    println(matrix)

    /**
     * Statistics.chiSqTest(: Matrix) 对输入列联矩阵进行Pearson 独立性检验，输入列联矩阵不能有负的项或累加为0的行或列。
     */
      println("\n独立性检验输出：")
    val Independence: ChiSqTestResult = Statistics.chiSqTest(matrix)
    println(Independence)
    println("\n独立性检验 Indenpendence  pvalue: \n"+BigDecimal.valueOf(Independence.pValue))

    /**
     *     本例中pValue =0.998，说明样本v1与期望值等于V2的数据分布并无显著差异。
     * 事实上，v1=[5.1,3.5,1.4,0.2]与v2= [4.9,3.0,1.4,0.2]很像，v1可以看做是从期望值为v2的数据分布中抽样出来的的。
     */


    /**
     * 同样的，键值对也可以进行独立性检验，这里我们取iris的数据组成键值对：
     *
     *
     */


    val value: RDD[LabeledPoint] = sc.textFile("/home/rjxy/IdeaProjects/spark/spark_mllib_course/src/main/resources/data/iris.data")
      .map(
        (item: String) => {
          val parts: Array[String] = item.split(",")
          //它表示数据点的特性和标签
          LabeledPoint(
            if (parts(4) == "Iris-setosa")
              0.toDouble
            else if (parts(4) == "Iris-versicolor")
              1.toDouble
            else
              2.toDouble
            ,
            Vectors.dense(
              parts(0).toDouble, parts(1).toDouble, parts(2).toDouble, parts(3).toDouble
            )
          )
        }
      )

    // 进行独立性检验，返回一个包含每个特征对于标签的卡方检验的数组：
    val independenceFeatureLabel: Array[ChiSqTestResult] = Statistics.chiSqTest(value)
    println("\n 进行独立性检验，返回一个包含每个特征对于标签的卡方检验的数组：")
    independenceFeatureLabel.foreach(i => println(BigDecimal.valueOf(i.pValue)))

    val value1: RDD[LabeledPoint] = sc.parallelize(Seq(
//      LabeledPoint(1, Vectors.dense(1, 1, 29)),
//      LabeledPoint(1, Vectors.dense(5, 5, 90)),
//      LabeledPoint(1, Vectors.dense(9, 29, 4)),
//      LabeledPoint(2, Vectors.dense(21, 0, 0)),
//      LabeledPoint(2, Vectors.dense(31, 11, 16)),
//      LabeledPoint(2, Vectors.dense(17, 19, 20))
      LabeledPoint(1, Vectors.dense(1, 1, 29)),
      LabeledPoint(1, Vectors.dense(1, 1, 29)),
      LabeledPoint(1, Vectors.dense(1, 1, 29)),
      LabeledPoint(1, Vectors.dense(2, 100, 18)),
      LabeledPoint(1, Vectors.dense(2, 100, 18)),
      LabeledPoint(1, Vectors.dense(2, 100, 18)),
      LabeledPoint(1, Vectors.dense(2, 100, 20))
    ))



    /**
     * ​ 这里实际上是把特征数据中的每一列都与标签进行独立性检验。可以看出，P值都非常小，说明可以拒绝“某列与标签列无关”的假设。也就是说，可以认为每一列的数据都与最后的标签有相关性。我们用foreach把完整结果打印出来：
     */
    println("\n")
    Statistics.chiSqTest(value1).foreach(i => println(BigDecimal.valueOf(i.pValue)))


    /**
     *    最后总结 pvalue
     *
     *  拟和度检验 类型 Vector
     *      pvalue < 0.05 拟和度概率底  < 0.01 拟和度概率底很低  书值越大证明 某列和 其他列 越拟合
     *
     *  独立性检验 类型 Matrix , (feature, label)
     *      pvalue < 0.05 独立性概率底  < 0.01 独立性概率很低    书值越大证明 某列和 矩阵列/特征列 不相关/独立的
     */

  }
}











