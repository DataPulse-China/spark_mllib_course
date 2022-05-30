package classificationmodel.pipline


import org.apache.log4j.Logger
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.mllib.evaluation.{MulticlassMetrics, RegressionMetrics}
import org.apache.spark.sql.DataFrame

import scala.collection.mutable

/**
 * Created by manpreet.singh on 01/05/16.
 */
object NaiveBayesPipeline {
  @transient lazy val logger = Logger.getLogger(getClass.getName)

  def naiveBayesPipeline(vectorAssembler: VectorAssembler, dataFrame: DataFrame) = {
    val Array(training, test) = dataFrame.randomSplit(Array(0.9, 0.1), seed = 12345)

    // 建立管道 Set up Pipeline
    val stages = new mutable.ArrayBuffer[PipelineStage]()

    //建立索引器
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
    stages += labelIndexer

    //朴素贝叶斯
    val nb = new NaiveBayes()

    //特征转换器 + 管道
    stages += vectorAssembler
    //朴素贝叶斯 + 管道
    stages += nb

    //管道添加
    val pipeline = new Pipeline().setStages(stages.toArray)

    // 开始时间（毫秒）Fit the Pipeline
    val startTime = System.nanoTime()
    //锻炼管道 使用附加参数使管道适合输入数据集。 val model = pipeline.fit(training)
    val model: PipelineModel = pipeline.fit(dataFrame)
    //结束时间
    val elapsedTime = (System.nanoTime() - startTime) / 1e9
    println(s"Training time: $elapsedTime seconds")

    //最终训练数据
    val finalDataFrame = model.transform(dataFrame)

    //val holdout = model.transform(test).select("prediction","label")
    //管道模型转换 数据
    val holdout: DataFrame = finalDataFrame.select("prediction","label")

    println("liliyahui")
    finalDataFrame.show(false)

    //必须对回归指标进行类型转换  have to do a type conversion for Regression Metrics
    val rm: RegressionMetrics = new RegressionMetrics(
      holdout.rdd.map(
        //            预测数据                    实际数据
        x => (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])
      )
    )



    logger.info("Test Metrics测试指标")
    logger.info("Test Explained Variance解释方差:")
    logger.info(rm.explainedVariance)
    logger.info("Test R^2 Coef:")
    logger.info(rm.r2)
    logger.info("Test MSE均方误差:")
    logger.info(rm.meanSquaredError)
    logger.info("Test RMSE均方根误差:")
    logger.info(rm.rootMeanSquaredError)

    //
    val predictions = model.transform(test).select("prediction").rdd.map(_.getDouble(0))
    val labels = model.transform(test).select("label").rdd.map(_.getDouble(0))
    val accuracy = new MulticlassMetrics(predictions.zip(labels)).precision

    //测试数据准确率
    println(s"  Accuracy : $accuracy")

    //
    holdout.rdd.map(x => x(0).asInstanceOf[Double]).repartition(1).saveAsTextFile("/home/rjxy/work/ml-resources/spark-ml/results/NB.xls")

    savePredictions(holdout, test, rm, "/home/rjxy/work/ml-resources/spark-ml/results/NaiveBayes.csv")
  }

  def savePredictions(predictions:DataFrame, testRaw:DataFrame, regressionMetrics: RegressionMetrics, filePath:String) = {
    predictions
      .coalesce(1)
      .write.format("com.databricks.spark.csv")
      .option("header", "true")
      .save(filePath)
  }
}
