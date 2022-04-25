package dimensionalityreduction

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.{Matrix, SingularValueDecomposition, Vectors}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.rdd.RDD

/**
 * Singular Value 奇异值 SVD
 *
 *     降维（Dimensionality Reduction） 是机器学习中的一种重要的特征处理手段，
 *  它可以减少计算过程中考虑到的随机变量（即特征）的个数，其被广泛应用于各种机器学习问题中，
 *  用于消除噪声、对抗数据稀疏问题。它在尽可能维持原始数据的内在结构的前提下，
 *  得到一组描述原数据的，低维度的隐式特征（或称主要特征）。
 *
 *       MLlib机器学习库提供了两个常用的降维方法：
 *   奇异值分解（Singular Value Decomposition，SVD）
 *   和
 *   主成分分析（Principal Component Analysis，PCA），
 *   下面我们将通过实例介绍其具体的使用方法。
 *
 */
object SingularValue {
  def main(args: Array[String]): Unit = {

    /**
     * 一、奇异值分解（SVD）
     *
     *  1、概念介绍
     *
     *      奇异值分解（SVD）** 来源于代数学中的矩阵分解问题，对于一个方阵来说，
     *    我们可以利用矩阵特征值和特征向量的特殊性质（矩阵点乘特征向量等于特征值数乘特征向量）
     *    ，通过求特征值与特征向量来达到矩阵分解的效果：
     *
     *    A = QΣQ^−1
     *
     *    这里，Q是由特征向量组成的矩阵，而Σ是特征值降序排列构成的一个对角矩阵（对角线上每个值是一个特征值
     *    ，按降序排列，其他值为0），特征值的数值表示对应的特征的重要性。
     *
     *      在很多情况下，最大的一小部分特征值的和即可以约等于所有特征值的和，而通过矩阵分解的降维就是通过在Q、Σ中
     *    删去那些比较小的特征值及其对应的特征向量，使用一小部分的特征值和特征向量来描述整个矩阵，从而达到降维的效果。
     *
     *        但是，实际问题中大多数矩阵是以奇异矩阵形式，而不是方阵的形式出现的，奇异值分解是特征值分解在奇异矩阵上的推广形式，
     *    它将一个维度为m×n奇异矩阵A分解成三个部分 :
     *    A=UΣV^T
     *
     *        其中U、V是两个正交矩阵，其中的每一行（每一列）分别被称为 左奇异向量 和 右奇异向量，他们和Σ中对角线上的奇异值相对应，
     *    通常情况下我们只需要取一个较小的值k，保留前k个奇异向量和奇异值即可，其中U的维度是m×k、V的维度是n×k、Σ是一个k×k的方阵，
     *    从而达到降维效果。
     *
     */

    /**
     * 2、SVD变换的例子
     *
     *
     * 准备好一个矩阵，这里我们采用一个简单的文件a.mat来存储一个尺寸为(4,9)的矩阵，其内容如下：
     *
      1 2 3 4 5 6 7 8 9
      5 6 7 8 9 0 8 6 7
      9 0 8 7 1 4 3 2 1
      6 4 2 1 3 4 2 1 5

        随后，将该文本文件读入成RDD[Vector]，并转换成RowMatrix，即可调用RowMatrix自带的computeSVD方法
      计算分解结果，这一结果保存在类型为SingularValueDecomposition的svd对象中：

    */

    Logger.getLogger("org").setLevel(Level.OFF)
    val sc = new SparkContext("local[*]", "li")
    val data: RDD[linalg.Vector] = sc.textFile("/home/rjxy/IdeaProjects/spark/spark_mllib_course/src/main/resources/data/a.mat")
      .map(
        (_: String).split(" ").map((_: String).toDouble)
      )
      .map((line: Array[Double]) => Vectors.dense(line))

    val matrix: RowMatrix = new RowMatrix(data)

    //保持领先的奇异值的数量(0 < k < = n)。它可能会返回小于k如果有数值零奇异值或没有足够的丽兹值聚合前达到Arnoldi更新迭代的最大数量(以防矩阵A是坏脾气的)。
    val value: SingularValueDecomposition[RowMatrix, Matrix] = matrix.computeSVD(3)
    println(value.s)
    println(value.V)
    println(value.U)

    /**
     * [28.741265581939565,10.847941223452608,7.089519467626695]
      -0.32908987300830383  0.6309429972945555    0.16077051991193514
      -0.2208243332000108   -0.1315794105679425   -0.2368641953308101
      -0.35540818799208057  0.39958899365222394   -0.147099615168733
      -0.37221718676772064  0.2541945113699779    -0.25918656625268804
      -0.3499773046239524   -0.24670052066546988  -0.34607608172732196
      -0.21080978995485605  0.036424486072344636  0.7867152486535043
      -0.38111806017302313  -0.1925222521055529   -0.09403561250768909
      -0.32751631238613577  -0.3056795887065441   0.09922623079118417
      -0.3982876638452927   -0.40941282445850646  0.26805622896042314
      null
     *
     *    这里可以看到，由于限定了取前三个奇异值，所以奇异值向量s包含有三个从大到小排列的奇异值，
     * 而右奇异矩阵V中的每一列都代表了对应的右奇异向量。U成员得到的是一个null值，这是因为在实际运用中，
     * 只需要V和S两个成员，即可通过矩阵计算达到降维的效果，其具体原理可以参看这篇博文：
     * 机器学习中的数学(5)-强大的矩阵奇异值分解(SVD)及其应用，这里不再赘述。如果需要获得U成员，
     * 可以在进行SVD分解时，指定computeU参数，令其等于True，即可在分解后的svd对象中拿到U成员，
     * 如下文所示：
     */

    val value1: SingularValueDecomposition[RowMatrix, Matrix] = matrix.computeSVD(3, computeU = true)

    println(value1.s)
    println(value1.V)
    val u: RowMatrix = value1.U
    val rows: RDD[linalg.Vector] = u.rows
    println(rows.foreach(println))

    PrincipalComponentAnalysis



  }
}
