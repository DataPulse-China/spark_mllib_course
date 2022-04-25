package dimensionalityreduction

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.feature.{PCA, PCAModel}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.{Matrix, Vectors}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
 * PrincipalComponentAnalysis 主成分分析
 *
 *    单词
 *      Principal 主要
 *      Component 成分
 *      Analysis 分析
 *
 *    1、概念介绍
 *
 *         主成分分析（PCA） 是一种对数据进行旋转变换的统计学方法，其本质是在线性空间中进行一个基变换，
 *     使得变换后的数据投影在一组新的“坐标轴”上的方差最大化，随后，裁剪掉变换后方差很小的“坐标轴”，
 *     剩下的新“坐标轴”即被称为 主成分（Principal Component） ，它们可以在一个较低维度的子空间
 *     中尽可能地表示原有数据的性质。主成分分析被广泛应用在各种统计学、机器学习问题中，是最常见的
 *     降维方法之一。PCA有许多具体的实现方法，可以通过计算协方差矩阵，甚至是通过上文提到的SVD分解
 *     来进行PCA变换。
 *
 *
 *
 */
object PrincipalComponentAnalysis {
  def main(args: Array[String]): Unit = {
    /**
     * 2、PCA变换
     *
            MLlib提供了两种进行PCA变换的方法，第一种与上文提到的SVD分解类似，位于org.apache.spark.mllib.linalg
        包下的RowMatrix中，这里，我们同样读入上文中提到的a.mat文件，对其进行PCA变换：
     */
    Logger.getLogger("org").setLevel(Level.OFF)
    val sc = new SparkContext("local[*]", "li")

    val amat: RDD[String] = sc.textFile("/home/rjxy/IdeaProjects/spark/spark_mllib_course/src/main/resources/data/a.mat")
    val data: RDD[linalg.Vector] = amat
      .map(
        (_: String).split(" ")
          .map((_: String).toDouble)
      )
      .map(line => Vectors.dense(line))

    //通过RDD[Vectors]创建行矩阵
    val rowMatrix: RowMatrix = new RowMatrix(data)
    //computePrincipalComponents  主成分分析（PCA）
    val pc: Matrix = rowMatrix.computePrincipalComponents(3)
    println("\n/computePrincipalComponents  主成分分析（PCA）")
    println(pc)
    /**
    -0.41267731212833847  -0.3096216957951525    0.1822187433607524
    0.22357946922702987   -0.08150768817940773   0.5905947537762997
    -0.08813803143909382  -0.5339474873283436    -0.2258410886711858
    0.07580492185074224   -0.56869017430423      -0.28981327663106565
    0.4399389896865264    -0.23105821586820194   0.3185548657550075
    -0.08276152212493619  0.3798283369681188     -0.4216195003799105
    0.3952116027336311    -0.19598446496556066   -0.17237034054712738
    0.43580231831608096   -0.023441639969444372  -0.4151661847170216
    0.468703853681766     0.2288352748369381     0.04103087747663084

        可以看到，主成分矩阵是一个尺寸为(9,3)的矩阵，其中每一列代表一个主成分（新坐标轴），
    每一行代表原有的一个特征，而a.mat矩阵可以看成是一个有4个样本，9个特征的数据集，
    那么，主成分矩阵相当于把原有的9维特征空间投影到一个3维的空间中，从而达到降维的效果。

     */

    //可以通过矩阵乘法来完成对原矩阵的PCA变换，可以看到原有的(4,9)矩阵被变换成新的(4,3)矩阵。
    println("\n可以通过矩阵乘法来完成对原矩阵的PCA变换，可以看到原有的(4,9)矩阵被变换成新的(4,3)矩阵。")
    val matrix: RowMatrix = rowMatrix.multiply(pc)
    matrix.rows.foreach(println)
    //需要注意的是，MLlib提供的PCA变换方法最多只能处理65535维的数据。

    /**
        3、“模型式”的PCA变换实现
     *      除了矩阵类内置的PCA变换外，MLlib还提供了一种“模型式”的PCA变换实现，
        它位于org.apache.spark.mllib.feature包下的PCA类，它可以接受RDD[Vectors]作为参数，进行PCA变换。

            该方法特别适用于原始数据是LabeledPoint类型的情况，只需取出LabeledPoint的feature成员
        （它是RDD[Vector]类型），对其做PCA操作后再放回，即可在不影响原有标签情况下进行PCA变换。
     */

    //依然使用前文的a.mat矩阵，为了创造出LabeledPoint，我们为第一个样本标注标签为0.0，其他为1.0。
    val value: RDD[Array[Double]] = amat
      .map(_.split(" ").map(_.toDouble))
    val data2: RDD[LabeledPoint] = value
      .map(line => LabeledPoint(if (line(0) > 1.0) 1.toDouble else 0.toDouble, Vectors.dense(line)))

    //LabeledPoint，我们为第一个样本标注标签为0.0，其他为1.0。
      data2.foreach(println)

    //随后，创建一个PCA类的对象，在构造器中给定主成分个数为3，并调用其fit方法来生成一个PCAModel类的对象pca，该对象保存了对应的主成分矩阵：
    val pCAModel: PCAModel = new PCA(3).fit(data2.map(_.features))
    println("\n“模型式”的PCA变换实现")
    println("主成分的数量")
    println(pCAModel.k) //主成分的数量。
    println("主成分矩阵。每一列是一个主成分")
    println(pCAModel.pc) //主成分矩阵。每一列是一个主成分
    println("解释方差")
    pCAModel.explainedVariance.foreachActive((i,j)=>println(i+"  "+j)) //解释方差

    //对于LabeledPoint型的数据来说，可使用map算子对每一条数据进行处理，将features成员替换成PCA变换后的特征即可：
    val projected = data2.map(
      (p: LabeledPoint) => p.copy(features = pCAModel.transform(p.features))
    )

    projected.foreach(println)

  }
}
