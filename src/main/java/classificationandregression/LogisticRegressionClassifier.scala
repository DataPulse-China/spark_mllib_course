package classificationandregression

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS, LogisticRegressionWithSGD}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
 * Logistic regression classifier  逻辑回归分类器
 *
 *  单词
 *    Logistic  逻辑
 *    regression  回归
 *    classifier  分类器
 *
 *    randomSplit 随机分割
 *    prediction  预测
 *    training 训练
 *    accuracy 返回准确性
 *
 *      一、分类算法概述
 *
 *              分类是一种重要的机器学习和数据挖掘技术。分类的目的是根据数据集的特点构造一个分类函数或分类模型(也常常称作分类器)，
 *          该模型能把未知类别的样本映射到给定类别中的一种技术。
 *
 * 分类的具体规则可描述如下：给定一组训练数据的集合T(Training set)，T的每一条记录包含若干条属性（Features）组成一个特
 * 征向量，用矢量x=(x1,x2,..,xn)表示。xi可以有不同的值域，当一属性的值域为连续域时，该属性为连续属性(Numerical Attribute)，
 * 否则为离散属性(Discrete Attribute)。用C=c1,c2,..ck表示类别属性，即数据集有k个不同的类别。那么，T就隐含了一个从矢量X到
 * 类别属性C的映射函数：f(X)↦C。分类的目的就是分析输入数据，通过在训练集中的数据表现出来的特性，为每一个类找到一种准确的描述
 * 或者模型，采用该种方法(模型)将隐含函数表示出来。
 *
 * 构造分类模型的过程一般分为训练和测试两个阶段。在构造模型之前，将数据集随机地分为训练数据集和测试数据集。先使用训练数据
 * 集来构造分类模型，然后使用测试数据集来评估模型的分类准确率。如果认为模型的准确率可以接受，就可以用该模型对其它数据元组进分类。
 * 一般来说，测试阶段的代价远低于训练阶段。
 *
 * 二、spark.mllib分类算法
 *
 * 分类算法基于不同的思想，算法也不尽相同，例如支持向量机SVM、决策树算法、贝叶斯算法、KNN算法等。spark.mllib包支持各种分类方法，
 * 主要包含 二分类， 多分类和 回归分析。下表列出了每种类型的问题支持的算法。
 *
 * 问题类型	                        支持的方法
 *
 * 二分类	                  线性支持向量机，Logistic回归，决策树，随机森林，梯度上升树，朴素贝叶斯
 * 多类分类                	Logistic回归，决策树，随机森林，朴素贝叶斯
 * 回归分析	                线性最小二乘法，Lasso，岭回归，决策树，随机森林，梯度上升树， isotonic regression
 *
 * 我们介绍一些典型的分类和回归算法
 */
//逻辑斯蒂回归分类器--spark.mllib
object LogisticRegressionClassifier {
  def main(args: Array[String]): Unit = {
    /**
     * 方法简介
          逻辑斯蒂回归（logistic regression）是统计学习中的经典分类方法，属于对数线性模型。logistic回归的因变量可以是二分类的，也可以是多分类的。
     */

    /**
     * logistic分布
     *    设X是连续随机变量，X服从logistic分布是指X具有下列分布函数和密度函数：
     *
        F(x) = P(x≤x) = 1 / (1+e−(x−μ)/γ
        f(x) = F(x) = e−(x−μ)/γ / (γ(1+e−(x−μ)/γ)2))

     其中，μ为位置参数，γ为形状参数。

      f(x)与F(x)图像如下，其中分布函数是以(μ,1/2)为中心对阵，γ越小曲线变化越快。
     http://mocom.xmu.edu.cn/blog/58578f242b2730e00d70f9fb.jpg
     */

    /**
     * 二项logistic回归模型：
     *
     *     二项logistic回归模型如下：
     *     P(Y=1|x) = exp(w⋅x+b) / 1+exp(w⋅x+b)
     *     P(Y=0|x) = 1 / 1+exp(w⋅x+b)
     *
           其中，x∈Rn是输入，Y∈0,1是输出，w称为权值向量，b称为偏置，w⋅x为w和x的内积。
            参数估计
            假设：
            P(Y=1|x)=π(x),P(Y=0|x)=1−π(x)

            则采用“极大似然法”来估计w和b。似然函数为:
            N ∏ i=1[π(xi)]yi[1−π(xi)]1−yi

            为方便求解，对其“对数似然”进行估计：
            L(w)=N∑i=1[yilogπ(xi)+(1−yi)log(1−π(xi))]
            从而对L(w)求极大值，得到w的估计值。求极值的方法可以是梯度下降法，梯度上升法等。

     */

    //示例代码
    /**
     *      我们以iris数据集（https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data）为例进行分析。
     *  iris以鸢尾花的特征作为数据来源，数据集包含150个数据集，分为3类，每类50个数据，每个数据包含4个属性，
     *  是在数据挖掘、数据分类中非常常用的测试集、训练集。
     */

    //1. 导入需要的包： mllib
    //2. 读取数据：
    /**
     *        首先，读取文本文件；然后，通过map将每行的数据用“,”隔开，在我们的数据集中，每行被分成了5部分，前4部分是鸢尾花的4个特征，
     *    最后一部分是鸢尾花的分类。把这里我们用LabeledPoint来存储标签列和特征列。
     *
          LabeledPoint在监督学习中常用来存储标签和特征，其中要求标签的类型是double，特征的类型是Vector。这里，先把莺尾花的分类进行变换，
          "Iris-setosa"对应分类 0，
          "Iris-versicolor"对应分类 1，
          其余对应分类 2
          然后获取莺尾花的4个特征，存储在Vector中。
     */
    Logger.getLogger("org").setLevel(Level.OFF)
    val sc = new SparkContext("local[*]", "li")

    val data: RDD[String] = sc.textFile("/home/rjxy/IdeaProjects/spark/spark_mllib_course/src/main/resources/data/iris.data")

    val irisValue: RDD[LabeledPoint] = data.map(
      (item: String) => {
        val parts: Array[String] = item.split(",")
        LabeledPoint(
          if (parts(4) == "Iris-setosa") 0.toDouble else if (parts(4) == "Iris-versicolor") 1.toDouble else 2.toDouble
          ,
          Vectors.dense(parts(0).toDouble, parts(1).toDouble, parts(2).toDouble, parts(3).toDouble)
        )
      }
    )
    irisValue.foreach(println)

    //3. 构建模型
    // 接下来，首先进行数据集的划分，这里划分60%的训练集和40%的测试集：
    val array: Array[RDD[LabeledPoint]] = irisValue.randomSplit(Array(0.6, 0.4), seed = 11L)

    val training: RDD[LabeledPoint] = array(0).cache()
    val test: RDD[LabeledPoint] = array(1)

    // 然后，构建逻辑斯蒂模型，用set的方法设置参数，比如说分类的数目，这里可以实现多分类逻辑斯蒂模型：
    /**
     * LogisticRegressionWithLBFGS
     *
     *      利用有限记忆BFGS训练多项式/二元Logistic回归分类模型。默认使用标准特征缩放和L2正则化。
        LogisticRegressionWithLBFGS的早期实现对包括拦截在内的所有元素应用了正则化惩罚。
        如果使用一个标准更新器(L1Updater，或SquaredL2Updater)调用该函数，则会转换为对ml.LogisticRegression的调用，
        否则将使用现有的mllib GeneralizedLinearAlgorithm训练器，从而导致对拦截的正则化惩罚。

          setNumClasses
        设定多项Logistic回归中k类分类问题的可能结果数。默认情况下，它是二元逻辑回归，所以k将被设为2
     */
    val model: LogisticRegressionModel = new LogisticRegressionWithLBFGS().setNumClasses(3).run(training)

    /**
     *      接下来，调用多分类逻辑斯蒂模型用的predict方法对测试数据进行预测，并把结果保存在MulticlassMetrics中。
     * 这里的模型全名为LogisticRegressionWithLBFGS，加上了LBFGS，表示Limited-memory BFGS。其中，BFGS是求解
     * 非线性优化问题（L(w)求极大值）的方法，是一种秩-2更新，以其发明者Broyden, Fletcher, Goldfarb和Shanno的姓氏首字母命名。
     */

    //预测和标签
    val predictionAndLabels: RDD[(Double, Double)] = test.map(
      item => {
        val prediction: Double = model.predict(item.features)
        (prediction, item.label)
      }
    )

    predictionAndLabels.foreach(println)

    //4. 模型评估   最后，我们把模型预测的准确性打印出来：
    val multiclassMetrics = new MulticlassMetrics(predictionAndLabels)
    println("\n 模型评估   最后，我们把模型预测的准确性打印出来：")
    println(multiclassMetrics.precision)
    //accuracy 返回准确性(等于实例总数中正确分类的实例的总数)。
    println(multiclassMetrics.accuracy)


  }
}
