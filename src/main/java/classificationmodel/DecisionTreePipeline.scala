package classificationmodel


import org.apache.log4j.Logger
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.sql.DataFrame

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/**
 * Created by manpreet.singh on 01/05/16.
 */
object DecisionTreePipeline {
  @transient lazy val logger = Logger.getLogger(getClass.getName)

  //决策树管道
  def decisionTreePipeline(vectorAssembler: VectorAssembler, dataFrame: DataFrame) = {
    val Array(training, test) = dataFrame.randomSplit(Array(0.9, 0.1), seed = 12345)

    // 设置管道
    val stages: ArrayBuffer[PipelineStage] = new mutable.ArrayBuffer[PipelineStage]()

    val labelIndexer: StringIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")

    stages += labelIndexer

    //
    val dt: DecisionTreeClassifier = new DecisionTreeClassifier()
      .setFeaturesCol(vectorAssembler.getOutputCol)
      .setLabelCol("indexedLabel")
      .setMaxDepth(5)
      .setMaxBins(32)
      .setMinInstancesPerNode(1)
      .setMinInfoGain(0.0)
      .setCacheNodeIds(false)
      .setCheckpointInterval(10)

    stages += vectorAssembler
    stages += dt
    val pipeline: Pipeline = new Pipeline().setStages(stages.toArray)

    // Fit the Pipeline
    val startTime: Long = System.nanoTime()
    //val model = pipeline.fit(training)
    val model: PipelineModel = pipeline.fit(dataFrame)
    val elapsedTime: Double = (System.nanoTime() - startTime) / 1e9
    println(s"Training time: $elapsedTime seconds")

    //val holdout = model.transform(test).select("prediction","label")
    val holdout: DataFrame = model.transform(dataFrame).select("prediction","label")



    // 选择（预测，真实标签）并计算测试误差
    // Select (prediction, true label) and compute test error
    //评估者
    val evaluator: MulticlassClassificationEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    //m 准确度
    val mAccuracy: Double = evaluator.evaluate(holdout)
    println("Test set accuracy = " + mAccuracy)
  }
}

