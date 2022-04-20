package datatype

import breeze.linalg
import breeze.linalg.{DenseMatrix, diag}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.{DenseVector, Matrices, Matrix, SparseMatrix, Vector, Vectors}
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, CoordinateMatrix, IndexedRow, IndexedRowMatrix, MatrixEntry, RowMatrix}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object matrixDemo1 {
  def main(args: Array[String]): Unit = {

    val value: linalg.DenseVector[Double] = breeze.linalg.DenseVector.ones[Double](10)
    val value1: DenseMatrix[Double] = diag(value)
    println(value1)

    SetLogger
    val session: SparkSession = SparkSession.builder().master("local[1]").getOrCreate()

    /**
     * 局部矩阵
     *
     * 单词
     *
     *  Matrix: 矩阵
     *
     *
     *
     *  局部矩阵具有整数类型的行和列索引以及双精度类型的值，他们存储在单个计算机上。
     *  Mllib支持稠密矩阵(其条目值以列优先顺序存储在单个双精度数组中)和稀疏矩阵(其非零条目值以列优先顺序和压缩稀疏列格式存储)
     *
     *
     * 以矩阵大小(3,2)存储在一维数组[1.0,3.0,5.0,2.0,4.0,6.0]中.
     * 局部矩阵的基类是Matrix,提供了两个实现:DenseMatrix和SparseMatrix.
     * 建议使用在矩阵中实现的工厂方法创建局部矩阵。 （mllib 中的局部矩阵以列优先顺序存储）
     *
     */

    //创建稠密矩阵
    println("创建稠密矩阵")
    println(Matrices.dense(3, 2, Array(1.0, 4.0, 5.0, 6.0, 2.0, 3.0)).toString())

    //创建稀疏矩阵

    //Matrices.sparse
    //   * @param numRows number of rows  行数
    //   * @param numCols number of columns  列数
    //   * @param colPtrs the index corresponding to the start of a new column  对应于新列开头的索引
    //   * @param rowIndices the row index of the entry  行索引
    //   * @param values non-zero matrix entries in column major  非零矩阵在主列上
    val matris: Matrix = Matrices.sparse(3, 3, Array(0, 2, 1, 3), Array(0, 1, 2), Array(9, 6, 8))


    //强制转换
    val s: SparseMatrix = matris.asInstanceOf[SparseMatrix]
    //打印稀疏矩阵
    println("创建稀疏矩阵")
    println(s.toString())
    //打印稠密矩阵
    println("创建稀疏矩阵 转换打印稠密矩阵")
    println(s.toDense.toString())
    println()


    /**
     * 分布矩阵
     *
     * 单词
     *
     * RowMatrix 行矩阵
       IndexedRowMatrix 索引行矩阵
       CoordinateMatrix 坐标矩阵
       BlockMatrix 块矩阵
     *
     *    分布式矩阵具有长整形行和列索引以及双精度，他们以分布式方式存储在一个或多个RDD中。
     * 选择正确的格式存储大型的分布式矩阵非常重要。将分布式矩阵转换为不同的格式可能需要全局洗牌,相当耗费资源.
     * 到目前为止，已经实现了四种类型的分布矩阵。
     *    基本类型称为RowMatrix,是面向行的分布式矩阵，例如特征向量的集合,行索引不具有意义。
     * RowMatrix条目的保存格式为RDD，每行是一个局部变量。假设RowMatrix的列数并不是很大，
     * 因此如果单个局部向量可以合理的传递给驱动程序，也可以使用单个节点进行存储和操作。
     *    IndexedRowMatrix 与 RowMatrix类似，但具有行索引，可用于识别行和连接操作。
     *    CoordinateMatrix是以坐标列表格式存储的分布式矩阵，其条目的保存格式为RDD。
     *    BlockMatrix是分布式矩阵，其由包含 MatrixBlock的RDD组成。MatrixBlock是(Int,Int,Matrix)的元组
     *
     */


    /**
     * 1.RowMatrix 行矩阵
     * parallelize 并行化
     *
     * RowMatrix 就能将每行对应一个RDD,将矩阵的每行分布式存储，
     * 矩阵的每行是一个局部向量。由于每一行均由局部向量表示，
     * 因此列数受正数范围限制，但实际上应小得多
     *
     */
    //创建RDD[Vector]
    val rddVector: RDD[Vector] = session.sparkContext.parallelize(Seq(
      Vectors.dense(2.0, 3.0, 4.0),
      Vectors.dense(5.0, 5.0, 5.0),
      Vectors.dense(2.0, 3.0, 4.0)
    ))
    //从RDD[Vector]创建RDDMatrix
    val rowMatrix: RowMatrix = new RowMatrix(rddVector)

    println("创建RDDMatrix")
    println(rowMatrix.rows.foreach(println))
    println(rowMatrix.numRows()+" RowMatrix行数")
    println(rowMatrix.numCols()+" RowMatrix列数")
    println()

    /**
     * 2.IndexedRowMatrix 索引行矩阵
     * IndexedRow 索引的行
     *
     *    IndexedRowMatrix 类似于 RowMatrix,但但行索引有意义。它由带索引行的RDD存储，因此每行都由长整形索引和局部变量表示。
     *    IndexRowMatrix 可以用 RDD[IndexRow]实例创建，其中IndexRow是一个基于(long,Vector)的包装器。
     *    IndexRowMatrix 可以通过删除行索引转换为RowMatrix
     */

    //创建RDD[IndexedRow]
    val rddIndexedRow: RDD[IndexedRow] = session.sparkContext.parallelize(Seq(
      IndexedRow(0, Vectors.dense(1, 3)),
      IndexedRow(1, Vectors.dense(4, 5))
    ))

    //用RDD[IndexedRow]创建 IndexedRowMatrix
    val indexedRowMatrix = new IndexedRowMatrix(rddIndexedRow)

    //行数
    println("IndexedRowMatrix矩阵")
    println(indexedRowMatrix.rows.foreach(println))
    println(indexedRowMatrix.numRows()+" indexedRowMatrix行数")
    //indexedRowMatrix去掉行索引
    println("indexedRowMatrix去掉行索引")
    indexedRowMatrix.toRowMatrix().rows.foreach(println)
    println()

    /**
     * 3.CoordinateMatrix 坐标矩阵
     *
     *    CoordinateMatrix 也是分布式矩阵,每个条目由RDD保存。每个条目是(i:Long,j:Long,value:Double)
     * 的一个元组，其中i是行索引，j是列索引，value是条目值。
     * CoordinateMatrix 只有在矩阵的两个维度都很大且矩阵非常稀疏时才能使用。CoordinateMatrix 可以由RDD[MaxtrixEntry]
     * 实例创建，其中MatrixEntry是基于(Long,Long,Double)的包装器。可以通过调用 toIndexRowMatrix 将
     * CoordinateMatrix 转换为具有稀疏行的 IndexRowMatrix。目前还不支持 CoordinateMatrix 的其他计算
     */

    //创建RDD[MatrixEntry]
    val rddMatrixEntry: RDD[MatrixEntry] = session.sparkContext.parallelize(Seq(
        MatrixEntry(0, 1, 1), MatrixEntry(0, 2, 2), MatrixEntry(0, 3, 3),
        MatrixEntry(0, 4, 4), MatrixEntry(2, 3, 5), MatrixEntry(2, 4, 6),
        MatrixEntry(3, 4, 7), MatrixEntry(4, 5, 8)
    ))

    //用RDD[MatrixEntry]创建 CoordinateMatrix
    val coordinateMatrix: IndexedRowMatrix = new CoordinateMatrix(rddMatrixEntry).toIndexedRowMatrix()

    //转换成IndexRowmatrix,其中的行 为稀疏向量
    println("用RDD[MatrixEntry]创建  CoordinateMatrix  稀疏 >>\n")
    coordinateMatrix.rows.foreach(println)
    println("CoordinateMatrix  稠密矩阵 ")

    plintMaxtrix(coordinateMatrix)


    /**
     * 4.BlockMatrix 块矩阵
     *
     * 单词
     *
     *
     */
    //创建 RDD[MaxEntry]
    val rddblockMatrixEntry: RDD[MatrixEntry] = session.sparkContext.parallelize(Seq(
      MatrixEntry(0, 0, 1.2),MatrixEntry(0, 1, 1.3),MatrixEntry(0, 2, 1.4),
      MatrixEntry(1, 0, 2.1),MatrixEntry(1, 1, 1.5),MatrixEntry(1, 2, 1.6),
      MatrixEntry(6, 10, 3.7),MatrixEntry(2, 0, 1.7)
    ))

    //用RDD[MaxEntry]创建 CoordinateMatrix
    val blockCoordinateMatrix: CoordinateMatrix = new CoordinateMatrix(rddblockMatrixEntry)

    //将CoordinateMatrix 转换为 BlockMatrix
    val blockMatrix: BlockMatrix = blockCoordinateMatrix.toBlockMatrix().cache()

    //验证BlockMatrix的设置是否正确，当它是无效的，则抛出一个异常
    println(blockMatrix.validate())
    println("BlockMatrix toIndexedRowMatrix() 稀疏矩阵  打印")
    blockMatrix.toIndexedRowMatrix().rows.sortBy(_.index).foreach((item: IndexedRow) => {
      println(item.index +""+ item.vector)
    })
    println("BlockMatrix toIndexedRowMatrix() 稠密矩阵  打印")
    val denseBlockMatrix: IndexedRowMatrix = blockMatrix.toIndexedRowMatrix()

    plintMaxtrix(denseBlockMatrix)

    println()
    println(" 计算A^T A ")
    val multiplyBlockMatrix: IndexedRowMatrix = blockMatrix.transpose.multiply(blockMatrix).toIndexedRowMatrix()



    plintMaxtrix(multiplyBlockMatrix)

  }

  /**
   * 打印机 IndexedRowMatrix   "[ \t \t ]"
   *
   * @param IndexedRowMatrix
   */
  def plintMaxtrix(IndexedRowMatrix:IndexedRowMatrix){
    IndexedRowMatrix.rows.sortBy(_.index).foreach((item: IndexedRow) => {
      println(item.index + item.vector.toDense.values.mkString("[","\t\t","]"))
    })
  }

  def SetLogger = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("com").setLevel(Level.OFF)
    System.setProperty("spark.ui.showConsoleProgress", "false")
    Logger.getRootLogger().setLevel(Level.OFF);
  }
}
