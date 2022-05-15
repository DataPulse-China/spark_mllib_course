package classificationmodel

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier, LogisticRegression, LogisticRegressionModel, NaiveBayes, NaiveBayesModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{LabeledPoint, VectorIndexer, VectorIndexerModel}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}

object Test1 {

  def main(args: Array[String]): Unit = {
    var conf: SparkConf = new SparkConf().setAppName("fenlei").setMaster("local[4]")
    conf.set("spark.testing.memory", "2147480000")
    var sc: SparkContext = new SparkContext(conf)

    val spark: SparkSession = SparkSession.builder().config(conf).appName("sda").getOrCreate()
    //    数据加载
    val rawData: RDD[String] = sc.textFile("D:/hadoop/spark/stumbleupon/train.tsv")
    rawData.take(2).foreach(println)
    val records: RDD[Array[String]] = rawData.map(line=>line.split("\t"))
    records.first().foreach(println)
    println(records.count())
    println(records.first().length)
    records.first().take(2).foreach(println)

    //  数据预处理
    //    去掉引号
    //    把标签列（即最后一列转换为整数）
    //    把第4列中的？ 转换为0.0
    //    使用lablepoint方法给数据打标签，即把标签及特征转换为LabeledPoint实例，同时把特征向量存储到ML的Vector中
    val data = records.map{r=> val trimmed = r.map(_.replaceAll("\"",""))
      val label = trimmed(r.size - 1).toInt
      val features = trimmed.slice(4, r.size - 1).map(d => if (d == "?") 0.0 else d.toDouble)
      //      给数据打标签
      LabeledPoint(label,Vectors.dense(features))
    }


    //    使用贝叶斯算法时，数据需要不小于0，故需要做些处理
    val nbData: RDD[LabeledPoint] = records.map { r =>
      val trimmed: Array[String] = r.map(_.replaceAll("\"",""))
      val label: Int = trimmed(r.size-1).toInt
      val features: Array[Double] =trimmed.slice(4, r.size - 1).map(d => if (d == "?") 0.0 else d.toDouble).map(d => if (d < 0) 0.0 else d)
      LabeledPoint(label,Vectors.dense(features))
    }

    data.take(2).foreach(print)

    //    通过RDD创建DataFrame
    val df: DataFrame = spark.createDataFrame(data)

    val nbDF: DataFrame = spark.createDataFrame(nbData)

    df.show(10)
    nbDF.show(10)
    df.printSchema()
    nbDF.printSchema()

    //    划分数据
    val Array(trainingData, testData) = df.randomSplit(Array(0.8,0.2),seed = 1234L)
    val Array(nbTrainingData, nbTestData) = nbDF.randomSplit(Array(0.8,0.2),seed = 1234L)
    println(trainingData.count())
    println(testData.count())

    //   缓存数据到内存
    trainingData.cache()
    testData.cache()
    nbTrainingData.cache()
    nbTestData.cache()

    //    创建贝叶斯模型
    val nb: NaiveBayes = new NaiveBayes().setLabelCol("label").setFeaturesCol("features")
    //    通过贝叶斯训练模型，对数据进行预测
    //    训练数据
    val nbModel: NaiveBayesModel = nb.fit(nbTrainingData)
    //    预测数据
    val nbPrediction: DataFrame = nbModel.transform(nbTestData)
    nbPrediction.show(10)



    //    朴素贝叶斯准确性统计
    //    t1存放预测值的数组，t2存放测试数据标签值，t3存放测试数据总行数
    val (t1,t2,t3) = (nbPrediction.select("prediction").collect(),
      nbTestData.select("label").collect(),nbTestData.count().toInt)
    //    t4为累加器
    var t4 = 0
    for (i <- 0 to t3 - 1){
      if (t1(i) == t2(i)){
        t4 += 1
      }
    }
    //    查看预测正确的个数
    println(t4)
    //    计算准确率
    val nbAccuracy: Double = 1.0 * t4 / t3
    println(nbAccuracy)

    //    组装
    //    建立特征索引
    val featureIndexer: VectorIndexerModel = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").fit(df)


    //    创建逻辑回归模型
    val lr: LogisticRegression = new LogisticRegression().setLabelCol("label").setFeaturesCol("indexedFeatures").setMaxIter(6).setRegParam(0.01)

    //    创建决策树模型
    //    setImpurity：指定信息熵，这里为entropy
    //    setMaxBins：离散化"连续特征"的最大划分数
    //    setMaxDepth：数的最大深度
    //    setMinInfoGain：一个节点分裂的最小信息增益
    val dt: DecisionTreeClassifier = new DecisionTreeClassifier()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")
      .setImpurity("entropy")
      .setMaxBins(100)
      .setMaxDepth(5)
      .setMinInfoGain(0.01)

    //    配置流水线
    val lrPipeline: Pipeline = new Pipeline().setStages(Array(featureIndexer,lr))

    val dtPipeline: Pipeline = new Pipeline().setStages(Array(featureIndexer,dt))

    //    模型优化
    //    分别配置网格参数
    val lrParamGrid: Array[ParamMap] = new ParamGridBuilder()
      .addGrid(lr.regParam,Array(0.1,0.3,0.5))
      .addGrid(lr.maxIter,Array(4,5,6)).build()

    val dtParamGrid: Array[ParamMap] = new ParamGridBuilder().addGrid(dt.maxDepth,Array(3,5,7)).build()

    //    分别实例化交叉验证模型,评估器
    val evaluator = new BinaryClassificationEvaluator()

    val lrCV: CrossValidator = new CrossValidator()
      .setEstimator(lrPipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(lrParamGrid)
      .setNumFolds(2)

    val dtCV: CrossValidator = new CrossValidator()
      .setEstimator(dtPipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(dtParamGrid)
      .setNumFolds(2)

    //    通过交叉验证模型，获取最优参数集，并测试模型
    val lrCvModel: CrossValidatorModel = lrCV.fit(trainingData)

    val dtCvModel: CrossValidatorModel = dtCV.fit(trainingData)

    val lrPrediction: DataFrame = lrCvModel.transform(testData)

    val dtPrediction: DataFrame = dtCvModel.transform(testData)

    //    查看数据
    lrPrediction.select("label","prediction").show(10)

    dtPrediction.select("label","prediction").show(10)


    //    查看逻辑回归匹配模型的参数
    val lrBestModel: PipelineModel = lrCvModel.bestModel.asInstanceOf[PipelineModel]

    val lrModel: LogisticRegressionModel = lrBestModel.stages(1).asInstanceOf[LogisticRegressionModel]
    println(lrModel.getRegParam +"  lrModel   " + lrModel.getMaxIter)

    //    查看决策树匹配模型的参数
    val dtBestModel: PipelineModel = dtCvModel.bestModel.asInstanceOf[PipelineModel]
    val dtModel: DecisionTreeClassificationModel = dtBestModel.stages(1).asInstanceOf[DecisionTreeClassificationModel]
    println(dtModel.getMaxDepth + "  dtModel " + dtModel.numFeatures)

    //    统计逻辑回归的预测正确率
    //    t_lr 为逻辑回归预测值的数组，t_dt为决策树预测值的数组
    //    t_label为测试集的标签值的数组
    val (t_lr, t_dt, t_label, t_count) = (lrPrediction.select("prediction").collect(),
      dtPrediction.select("prediction").collect(),
      testData.select("label").collect(),testData.count().toInt)

    //    c_lr为统计逻辑回归预测正确个数的累加器
    //    c_dt为统计决策树预测正确个数的累加器
    var Array(c_lr,c_dt) = Array(0,0)
    for (i <- 0 until t_count){
      if (t_lr(i) == t_label(i)){
        c_lr += 1
      }
    }
    //    统计逻辑回归正确率
    println(1.0 * c_lr / t_count)

    for (i <- 0 until t_count){
      if (t_dt(i) == t_label(i)){
        c_dt += 1
      }
    }

    //    统计决策树正确率
    println(1.0*c_dt / t_count)
  }
}
