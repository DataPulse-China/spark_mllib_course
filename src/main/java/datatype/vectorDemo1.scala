package datatype

import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg._
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
object vectorDemo1 {
  def main(args: Array[String]): Unit = {
    /**
     * 局部变量
     *
     * 单词
     *
     *  dense: 稠密
     *  sparse: 稀疏
     *
     *  indices: index array, must be strictly increasing. 索引 索引数组，必须严格递增
     *  elements: vector elements in (index, value) pairs  元素(索引，值)对中的向量元素
     *
     *  稠密向量
     *      由表示其输入值的双精度数组支持,而稀疏变量有两个并行数组支持：索引和值。
     *          例如：一个向量(1.0,0.0,3.0)可以用稠密格式表示为[1.0,0.0,3.0],或者以(3,[0,2],[1.0,3.0])的稀疏格式表示
     *  稀疏变量
     *      一个向量(1.0,0.0,3.0)表示为(3,[0,2],[1.0,3.0])的稀疏格式表示,其中第一个值3为向量的大小，第二个值表示向量中有值数据的索引，第三个值表示向量的值
     *
     *  局部变量的基类是Vector,提供了两个实现:DenseVector SparseVector 。 建议使用Vectors中实现的工厂方法创建局部变量。org.apache.spark.ml.linalg.Vectors
     *
     */

    //创建稠密向量（1.0,0.0,3.0）
    println("创建稠密向量（1.0,0.0,3.0）Vectors.dense(1.0,0.0,3.0)")
    //                    元素
    Vectors.dense(1.0,0.0,3.0).foreachActive((i,j)=>println(i+"  "+j))
    println("创建稀疏向量（1.0,0.0,3.0） Vectors.sparse(3,Array(0,2),Array(1.0,3.0))")

    //通过指定非零条目的索引和值，创建一个稀疏变量（1.0,0.0,3.0）
    //                       索引         值
    Vectors.sparse(3,Array(0,2),Array(1.0,3.0)).foreachActive((i,j)=>println(i+"  "+j))
    println("创建稀疏向量（1.0,0.0,3.0） Vectors.sparse(3,Seq((0,1.0),(2,3.0)))")

    //通过指定非零条目创建一个 稀疏变量 (1.0,0.0,3.0)
    //                       元素
    Vectors.sparse(3,Seq((0,1.0),(2,3.0))).foreachActive((i,j)=>println(i+"  "+j))
    println()

    /**
     * 标签向量
     *    标签向量是一个稠密或稀疏的局部变量,而且关联了标签.在MLlib中,标签向量用于有监督学习算法.
     *    因为使用双精度存储标签，所以可以在回归和分类中使用标签向量.
     *    对于二元分类，标签应该是0(负)或1(正).对于多雷分类，标签应该是从零开始的类索引：0,1,2,3。。。.标签向量由案例类Labeledoint表示
     *
     * 单词
     * LabeledPoint: 标记点
     * vector: 向量
     * Features: 特征
     */

    //使用正标签和稠密特征创建标签向量
    LabeledPoint(1.0,Vectors.dense(1.0,0.0,3.0))

    //使用负标签和稀疏特征向量创建标签向量
    LabeledPoint(0.0,Vectors.sparse(3,Array(0,2),Array(1.0,3.0)))

    /**
     * LIBSVM
     *
     * 在实践中,使用稀疏的训练数据是很常见的。saprkML支持以LIBSVM格式存储的阅读训练样本，
     * 这是LIBSVM和LIBLINEAR使用的默认格式,是一种文本格式,其中每行代表使用以下格式标记的稀疏特征向量：
     * label index1:value1 index2:value2...
     * 索引一开始就按升序排列。加载后，特征索引将转换为基于零的索引。libsvm包用于奖LIBSVM数据加载为DataFrame的数据元API.
     * 载的DataFrame有两列：包含作为双精度存储的标签和包含作为向量存储的特征。要使用LIBSVM格式的数据原，需要在DataFrameReader中将格式设置为libsvm,并可以指定option例如:
     * SparkSession.builder().getOrCreate().read.format("libsvm")
        *.option("numFeatures","780")
        *.load("/home/rjxy/IdeaProjects/spark/spark_mllib_course/src/main/resources/data/sample_libsvm_data.txt")
     *
     * numFeatures 特征数量
        *指定特征向量数量,如果为指定或不是正数，特征向量的数量将自动确定，但额外需要计算的代价
       *vectorType 特征向量类型 稀疏或稠密
        *特征向量类型，稀疏(默认)或稠密。
     *
     *    LIBSVM是台湾大学林智仁(Lin Chih-Jen)教授等开发设计的一个简单、易于使用和快速有效的SVM模式识别与回归的软件包，
     * 他不但提供了编译好的可在Windows系列系统的执行文件，还提供了源代码，方便改进、修改以及在其它操作系统上应用；
     * 该软件对SVM所涉及的参数调节相对比较少，提供了很多的默认参数，利用这些默认参数可以解决很多问题；并提供了交互检验(Cross Validation)
     * 的功能。该软件可以解决C-SVM、ν-SVM、ε-SVR和ν-SVR等问题，包括基于一对一算法的多类模式识别问题。
     */

    val session: SparkSession = SparkSession.builder().master("local[*]").getOrCreate()
      session.read.format("libsvm")
      .option("numFeatures","780")
      .load("/home/rjxy/IdeaProjects/spark/spark_mllib_course/src/main/resources/data/sample_libsvm_data.txt")
    //numFeatures 特征数量
      //指定特征向量数量,如果为指定或不是正数，特征向量的数量将自动确定，但额外需要计算的代价

    //vectorType 特征向量类型 稀疏或稠密
      //特征向量类型，稀疏(默认)或稠密。
  }
}
