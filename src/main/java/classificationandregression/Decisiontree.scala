package classificationandregression

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD

/**
 * DecisionTree 决策树分类器--spark.mllib
 *
 *  单词
 *  Decision 决策
 *  Tree 树
 *  trainClassifier 训练分类器
 *
 *    一、方法简介
 *
 * 决策树（decision tree）是一种基本的分类与回归方法，这里主要介绍用于分类的决策树。决策树模式呈树形结构，
 * 其中每个内部节点表示一个属性上的测试，每个分支代表一个测试输出，每个叶节点代表一种类别。学习时利用训练数据，
 * 根据损失函数最小化的原则建立决策树模型；预测时，对新的数据，利用决策树模型进行分类。
 *
 * 二、基本原理
 * 决策树学习通常包括3个步骤：特征选择、决策树的生成和决策树的剪枝。
 *
 * （一）特征选择
 *
 * 特征选择在于选取对训练数据具有分类能力的特征，这样可以提高决策树学习的效率。通常特征选择的准则是
 * 信息增益（或信息增益比、基尼指数等），每次计算每个特征的信息增益，并比较它们的大小，选择信息增益最大
 * （信息增益比最大、基尼指数最小）的特征。下面我们重点介绍一下特征选择的准则：信息增益。
 *
 * 首先定义信息论中广泛使用的一个度量标准——熵（entropy），它是表示随机变量不确定性的度量。熵越大，
 * 随机变量的不确定性就越大。而信息增益（informational entropy）表示得知某一特征后使得信息的不确定性减
 * 少的程度。简单的说，一个属性的信息增益就是由于使用这个属性分割样例而导致的期望熵降低。信息增益、信息增益比
 * 和基尼指数的具体定义如下：
 *
 * 信息增益：特征A对训练数据集D的信息增益g(D,A)，定义为集合D的经验熵H(D)与特征A给定条件下D的经验条件熵H(D|A)之差，即
 * g(D,A)=H(D)−H(D|A)
 *
 * 信息增益比：特征A对训练数据集D的信息增益比gR(D,A)定义为其信息增益g(D,A)与训练数据集D关于特征A的值的熵HA(D)之比，即
 * gR(D,A)=g(D,A)HA(D)
 * 其中，HA(D)=−∑ni=1|Di||D|log2|Di||D|，n是特征A取值的个数。
 *
 * 基尼指数：分类问题中，假设有K个类，样本点属于第K类的概率为pk， 则概率分布的基尼指数定义为:
 * Gini(p)=K∑k=1pk(1−pk)=1−K∑k=1p2k
 *
 *
 * （二）决策树的生成
 *
 * 从根结点开始，对结点计算所有可能的特征的信息增益，选择信息增益最大的特征作为结点的特征，由该特征的不同取值建立子结点，
 * 再对子结点递归地调用以上方法，构建决策树；直到所有特征的信息增均很小或没有特征可以选择为止，最后得到一个决策树。
 *
 * 决策树需要有停止条件来终止其生长的过程。一般来说最低的条件是：当该节点下面的所有记录都属于同一类，或者当所有的记录属
 * 性都具有相同的值时。这两种条件是停止决策树的必要条件，也是最低的条件。在实际运用中一般希望决策树提前停止生长，限定叶节点
 * 包含的最低数据量，以防止由于过度生长造成的过拟合问题。
 *
 * （三）决策树的剪枝
 *
 * 决策树生成算法递归地产生决策树，直到不能继续下去为止。这样产生的树往往对训练数据的分类很准确，但对未知的测试数据的分
 * 类却没有那么准确，即出现过拟合现象。解决这个问题的办法是考虑决策树的复杂度，对已生成的决策树进行简化，这个过程称为剪枝。
 *
 * 决策树的剪枝往往通过极小化决策树整体的损失函数来实现。一般来说，损失函数可以进行如下的定义：
 * Ca(T) = C(T)+a|T|
 * 其中，T为任意子树，C(T) 为对训练数据的预测误差（如基尼指数），|T| 为子树的叶结点个数，a≥0为参数，Ca(T) 为参数
 * 是a时的子树T的整体损失，参数a权衡训练数据的拟合程度与模型的复杂度。对于固定的a，一定存在使损失函数Ca(T) 最小的子树，
 * 将其表示为Ta 。当a大的时候，最优子树Ta偏小；当a小的时候，最优子树Ta偏大。
 *
 * 三、示例代码
 *
 * 我们以iris数据集（https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data）为例进行分析。
 * iris以鸢尾花的特征作为数据来源，数据集包含150个数据集，分为3类，每类50个数据，每个数据包含4个属性，
 * 是在数据挖掘、数据分类中非常常用的测试集、训练集。
 *
 *
 *
 */
object Decisiontree {
  def main(args: Array[String]): Unit = {
    //读取数据：
    /**
     *      首先，读取文本文件；然后，通过map将每行的数据用“,”隔开，在我们的数据集中，每行被分成了5部分，前4部分是鸢尾花的4个特征，
     *   最后一部分是鸢尾花的分类。把这里我们用LabeledPoint来存储标签列和特征列。LabeledPoint在监督学习中常用来存储标签和特征，
     *   其中要求标签的类型是double，特征的类型是Vector。所以，我们把莺尾花的分类进行了一下改变，
     *   "Iris-setosa"对应分类0，
     *   "Iris-versicolor"对应分类1，
     *   其余对应分类2；
     *   然后获取莺尾花的4个特征，存储在Vector中。
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

    //3. 构建模型  接下来，首先进行数据集的划分，这里划分70%的训练集和30%的测试集：
    val splits: Array[RDD[LabeledPoint]] = irisValue.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    //然后，调用决策树的 trainClassifier 训练分类器 方法构建决策树模型，设置参数，比如分类数、信息增益的选择、树的最大深度等：
    //分类数 用于分类的类的数量。
    val numClasses = 3
    //categoricalFeaturesInfo 分类特征的映射存储能力。条目(n到k)表示特征n是分类的，有k个类别从0开始索引:{0,1，…, k - 1}。
    val categoricalFeaturesInfo = Map[Int,Int]()
    //impurity 用于信息增益计算的准则。 (推荐) "gini" (recommended) or "entropy"。
    val impurity = "gini"
    //树的最大深度(例如，深度0表示1个叶节点，深度1表示1个内部节点+ 2个叶节点)。(建议值:5)
    val maxDepth = 5
    //用于拆分特性的最大容器数。(建议值:32)
    val maxBins = 32
    //决策树分类模型
    val decisionTreeModel: DecisionTreeModel = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

    //标签和预测
    val labelAndPredicts: RDD[(Double, Double)] = testData.map(
      item => {
        val prediction: Double = decisionTreeModel.predict(item.features)
        (item.label, prediction)
      }
    )

    labelAndPredicts.foreach(println)
    //4. 模型评估
    val ErrRate: Double = labelAndPredicts.filter(r => r._1 != r._2).count().toDouble / testData.count()
    println("ErrRate  "+(ErrRate))
    println("rightRate  "+(1-ErrRate))



  }
}
