package classificationandregression.bikesharing

import org.apache.log4j.Logger
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.regression.GeneralizedLinearRegression
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.{SparkSession, _}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.regression.DecisionTreeRegressor

/**
  * 决策树回归管道  Created by manpreet.singh on 22/11/16.
  */
object DecisionTreeRegressionPipeline {

  @transient lazy val logger = Logger.getLogger(getClass.getName)

  //决策树 向量格式的回归
  def decTreeRegressionWithVectorFormat(vectorAssembler: VectorAssembler, vectorIndexer: VectorIndexer, dataFrame: DataFrame) = {

    //决策树 回归器
    val lr: DecisionTreeRegressor = new DecisionTreeRegressor()
      //设置特征列
      .setFeaturesCol("features")
      //设置标签列
      .setLabelCol("label")

    //将 特征转换器 和 向量索引器 添加到管道流水线中
    val pipeline = new Pipeline().setStages(Array(vectorAssembler, vectorIndexer, lr))

    //
    val Array(training: Dataset[Row], test: Dataset[Row]) = dataFrame.randomSplit(Array(0.8, 0.2), seed = 12345)

    //
    val model: PipelineModel = pipeline.fit(training)

    // 预测数据  Make predictions.
    val predictions = model.transform(test)

    // 显示主要数据  Select example rows to display.
    predictions.select("prediction", "label", "features").show(5)

    // 预测误差  Select (prediction, true label) and compute test error.
    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    //绝方根误差
    val rmse = evaluator.evaluate(predictions)
    println("Root Mean Squared Error (RMSE) on test data = " + rmse)

    //
    val treeModel = model.stages(1).asInstanceOf[DecisionTreeRegressionModel]
    println("Learned regression tree model:\n" + treeModel.toDebugString)  }

  //决策树支持向量机格式的树回归
  def decTreeRegressionWithSVMFormat(spark: SparkSession) = {
    // Load training data
    val training = spark.read.format("libsvm")
      .load("/Users/manpreet.singh/Sandbox/codehub/github/machinelearning/spark-ml/Chapter_07/scala/2.0.0/scala-spark-app/src/main/scala/org/sparksamples/regression/dataset/BikeSharing/lsvmHours.txt")

    // Automatically identify categorical features, and index them.
    // Here, we treat features with > 4 distinct values as continuous.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(training)

    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = training.randomSplit(Array(0.7, 0.3))

    // Train a DecisionTree model.
    val dt = new DecisionTreeRegressor()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")

    // Chain indexer and tree in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(featureIndexer, dt))

    // Train model. This also runs the indexer.
    val model = pipeline.fit(trainingData)

    // Make predictions.
    val predictions = model.transform(testData)

    // Select example rows to display.
    predictions.select("prediction", "label", "features").show(5)

    // Select (prediction, true label) and compute test error.
    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println("Root Mean Squared Error (RMSE) on test data = " + rmse)

    val treeModel = model.stages(1).asInstanceOf[DecisionTreeRegressionModel]
    println("Learned regression tree model:\n" + treeModel.toDebugString)
  }

}
