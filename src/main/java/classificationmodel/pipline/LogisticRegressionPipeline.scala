package classificationmodel.pipline


import org.apache.log4j.Logger
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.DataFrame

/**
 * Created by manpreet.singh on 01/05/16.
 */
object LogisticRegressionPipeline {
  @transient lazy val logger = Logger.getLogger(getClass.getName)

  def logisticRegressionPipeline(vectorAssembler: VectorAssembler, dataFrame: DataFrame) = {

    //创建逻辑回归
    val lr = new LogisticRegression()

    //用于基于网格搜索的模型选择的参数网格的构建器。
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .addGrid(lr.fitIntercept)
      .addGrid(lr.elasticNetParam, Array(0.0, 0.25, 0.5, 0.75, 1.0))
      .build()

    //管道添加 特征转换  逻辑回归
    val pipeline = new Pipeline().setStages(Array(vectorAssembler, lr))

    //将输入数据集随机拆分为训练集和验证集，并使用验证集上的评估指标来选择最佳模型。类似于 [[CrossValidator]]，但只拆分集合一次。
    val trainValidationSplit = new TrainValidationSplit()
      //估计器
      .setEstimator(pipeline)
      //评估器
      .setEvaluator(new RegressionEvaluator)
      //设置 估计器参数映射
      .setEstimatorParamMaps(paramGrid)
      // 80% 的数据将用于训练，其余 20% 用于验证  80% of the data will be used for training and the remaining 20% for validation.
      .setTrainRatio(0.8)


    val Array(training, test) = dataFrame.randomSplit(Array(0.8, 0.2), seed = 12345)
    //val model = trainValidationSplit.fit(training)
    //输入机器拆分器 输入数据
    val model = trainValidationSplit.fit(dataFrame)

    //val holdout = model.transform(test).select("prediction","label")
    val holdout = model.transform(dataFrame).select("prediction","label")

    // have to do a type conversion for RegressionMetrics
    val rm = new RegressionMetrics(holdout.rdd.map(x => (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))

    logger.info("Test Metrics")
    logger.info("Test Explained Variance:")
    logger.info(rm.explainedVariance)
    logger.info("Test R^2 Coef:")
    logger.info(rm.r2)
    logger.info("Test MSE:")
    logger.info(rm.meanSquaredError)
    logger.info("Test RMSE:")
    logger.info(rm.rootMeanSquaredError)

    val totalPoints = dataFrame.count()
    val lrTotalCorrect = holdout.rdd.map(x => if (x(0).asInstanceOf[Double] == x(1).asInstanceOf[Double]) 1 else 0).sum()
    val accuracy = lrTotalCorrect/totalPoints
    println("Accuracy of LogisticRegression is: ", accuracy)

    holdout.rdd.map(x => x(0).asInstanceOf[Double]).repartition(1).saveAsTextFile("/home/ubuntu/work/ml-resources/spark-ml/results/LR.xls")
    holdout.rdd.map(x => x(1).asInstanceOf[Double]).repartition(1).saveAsTextFile("/home/ubuntu/work/ml-resources/spark-ml/results/Actual.xls")

    savePredictions(holdout, dataFrame, rm, "/home/ubuntu/work/ml-resources/spark-ml/results/LogisticRegression.csv")
  }

  def savePredictions(predictions:DataFrame, testRaw:DataFrame, regressionMetrics: RegressionMetrics, filePath:String) = {
    println("Mean Squared Error:", regressionMetrics.meanSquaredError)
    println("Root Mean Squared Error:", regressionMetrics.rootMeanSquaredError)

    predictions
      .coalesce(1)
      .write.format("com.databricks.spark.csv")
      .option("header", "true")
      .save(filePath)
  }
}
