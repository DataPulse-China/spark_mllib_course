package temp

import breeze.linalg._
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.{GradientDescent, LogisticGradient, SquaredL2Updater}
import org.apache.spark.rdd.RDD

import scala.util.Random

/**
 * 随机梯度下降(SGD)
 *
 */
object SparkSGD {
  def main(args: Array[String]): Unit = {
    var m = 4
    var n = 2000
    var sc = new SparkContext("local[*]","li")

    //
    val points: RDD[(Double, linalg.Vector)] = sc.parallelize(0 until (4), 2)
      .mapPartitionsWithIndex(
        (index: Int, item: Iterator[Int]) => {
          val random = new Random(index)
          item.map(
            (i: Int) => (1.0,
              //              所需元素的数量  元素的计算   大小为n的数组，其中每个元素包含计算结果
              //                                          【从这个随机数生成器序列中返回下一个伪随机、均匀分布的双精度值，
              //                                           该值介于0.0和1.0之间。】
              Vectors.dense(Array.fill(n)(random.nextDouble())))
          )
        }
      ).cache()

    /**
     * GradientDescent 随机梯度下降法
     *
     * 使用小批量并行运行随机梯度下降(SGD)。在每次迭代中，我们采样总数据的一个子集(fraction miniBatchFraction)，
     * 以便计算梯度估计。在每次迭代中使用一个标准的spark map-reduce来对这个子集上的子梯度进行采样和平均。
     *  参数:
     *    data -为SGD输入数据。RDD的数据集示例，每个表单(标签，[特征值])。
     *    gradient - gradient对象(用于计算单个数据示例的损失函数的梯度)
     *    updater -在给定方向上实际执行梯度步骤的更新函数。
     *    stepSize —第一步的初始步长
     *    numIterations — SGD应该运行的迭代次数。
     *    regParam -正则化参数
     *    miniBatchFraction — 输入数据集中应该用于SGD一次迭代的部分。默认值1.0
     *    SGD一次迭代的输入数据集的分数。默认值1.0。
     *    convergenceTol — 如果当前权值与前一个权值的相对差小于此值，则小批量迭代将在
     *    numIterations之前结束。在测量收敛性时，计算L2范数。默认值0.001。必须在0.0和1.0之间。
     *  返回: 包含两个元素的元组。
     *    第一个元素是包含每个特征权值的列矩阵，
     *    第二个元素是包含每次迭代计算的随机损失的数组。
     *
     *
     *   new LogisticGradient( ) 逻辑回归的梯度下降
     *   SquaredL2Updater 平方l2更新器
     */
    val tuple: (linalg.Vector, Array[Double]) = GradientDescent.runMiniBatchSGD(
      data = points, //SGD输入数据
      new LogisticGradient(), //gradient对象(用于计算单个数据示例的损失函数的梯度)
      new SquaredL2Updater, //在给定方向上实际执行梯度步骤的更新函数。
      0.1, //第一步的初始步长
      2, // SGD应该运行的迭代次数
      1.0, //正则化参数
      1.0, //输入数据集中应该用于SGD一次迭代的部分。默认值1.0
      initialWeights = Vectors.dense(new Array[Double](n)) //最初的重量
    )

    tuple._1.foreachActive((item1,item) => println(item1+" "+item))
    tuple._2.foreach(println)

//    val value: Vector[Double] = tuple._1.




  }
}
