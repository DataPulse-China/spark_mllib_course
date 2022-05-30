package classificationmodel

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.{ClassificationModel, LogisticRegressionModel, LogisticRegressionWithSGD, NaiveBayes, NaiveBayesModel, SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.optimization.{SimpleUpdater, SquaredL2Updater, Updater}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.{Entropy, Gini, Impurity}
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD

/**
	* @author manpreet.singh
	*/

object AllInOneClassification {

	def main(args: Array[String]) {
		val sc = new SparkContext("local[2]", "Classification")

		Logger.getLogger("org.apache.spark").setLevel(Level.OFF)

		// get StumbleUpon dataset 'https://www.kaggle.com/c/stumbleupon'
		val records = sc.textFile("/home/rjxy/IdeaProjects/spark/spark_mllib_course/src/main/resources/stumbleupon/train_noheader.tsv")
			.map(line => line.split("\t"))
			//.map(records => (records(0), records(1), records(2)))

		// records.foreach(println)


		// 线性模型
		val data: RDD[LabeledPoint] = records.map { r: Array[String] =>
			//修剪过的数据
			val trimmed: Array[String] = r.map((_: String).replaceAll("\"", ""))
			//标签
			val label: Int = trimmed(r.length - 1).toInt
			//特征 数组
			val features: Array[Double] = trimmed.slice(4, r.length - 1).map(d => if (d == "?") 0.0 else d.toDouble)
			LabeledPoint(label, Vectors.dense(features))
		}

		// for nb model
		val nbData: RDD[LabeledPoint] = records.map { r: Array[String] =>
			//修剪过的数据
			val trimmed: Array[String] = r.map((_: String).replaceAll("\"", ""))
			//标签
			val label: Int = trimmed(r.length - 1).toInt
			//特征 数组
			val features: Array[Double] = trimmed.slice(4, r.length - 1).map((d: String) => if (d == "?") 0.0 else d.toDouble).map(d => if (d < 0) 0.0 else d)
			LabeledPoint(label, Vectors.dense(features))
		}

		data.cache()

		val numData: Long = data.count()
		println(numData)

    // params for logistic regression and SVM
		val numIterations = 10
		// params for decision tree
		val maxTreeDepth = 5

		//逻辑回归模型
		val lrModel: LogisticRegressionModel = LogisticRegressionWithSGD.train(data, numIterations)

		//支持向量机模型
		val svmModel: SVMModel = SVMWithSGD.train(data, numIterations)

		//朴素贝叶斯模型
		val nbModel: NaiveBayesModel = NaiveBayes.train(nbData)

		//决策树模型
		val dtModel: DecisionTreeModel = DecisionTree.train(data, Algo.Classification, Entropy, maxTreeDepth)


		// 单数据点预测 single datapoint prediction
		val dataPoint = data.first()
		val prediction = lrModel.predict(dataPoint.features)
		println(prediction)
		println(dataPoint.label)

		// 批量预测 bulk prediction
		val predictions = lrModel.predict(data.map(lp => lp.features))
		predictions.foreach(println)

		// 逻辑回归 完全正确 准确性和预测误差 accuracy and prediction error
		val lrTotalCorrect = data.map { point =>
			if(lrModel.predict(point.features) == point.label) 1 else 0
		}.sum()

		//支持向量机 完全正确
		val svmTotalCorrect = data.map { point =>
			if(svmModel.predict(point.features) == point.label) 1 else 0
		}.sum()

		//朴素贝叶斯 完全正确
		val nbTotalCorrect = data.map { point =>
			if(nbModel.predict(point.features) == point.label) 1 else 0
		}.sum()

		// 决策树 完全正确
		val dtTotalCorrect = data.map { point =>
			val score = dtModel.predict(point.features)
			val predicted = if (score > 0.5) 1 else 0
			if(predicted == point.label) 1 else 0
		}.sum()

		//逻辑回归  准确性和预测误差 accuracy and prediction error
		val lrAccuracy = lrTotalCorrect / data.count()
		println(lrAccuracy)

		//支持向量机 准确性和预测误差 accuracy and prediction error
		val svmAccuracy = svmTotalCorrect / data.count()
		println(svmAccuracy)

		//朴素贝叶斯 准确性和预测误差 accuracy and prediction error
		val nbAccuracy = nbTotalCorrect / data.count()
		println(nbAccuracy)

		//决策树 准确性和预测误差 accuracy and prediction error
		val dtAccuracy = dtTotalCorrect / data.count()
		println(dtAccuracy)

    // ROC曲线和 AUC ROC curve and AUC
		val metrics: Seq[(String, Double, Double)] = Seq(lrModel, svmModel).map {
			model =>
			//预测（分数）和标签
			val scoreAndLabels: RDD[(Double, Double)] = data.map { point =>
				(model.predict(point.features), point.label)
			}
			// BinaryClassificationMetrics  分类指标 分类评估器
			val metrics = new BinaryClassificationMetrics(scoreAndLabels)

			( model.getClass.getSimpleName,
				metrics.areaUnderPR,
				metrics.areaUnderROC)
		}

		val nbMetrics = Seq(nbModel).map{ model =>
			val scoreAndLabels = nbData.map { point =>
				val score = model.predict(point.features)
				(if (score > 0.5) 1.0 else 0.0, point.label)
			}
			val metrics = new BinaryClassificationMetrics(scoreAndLabels)
			(model.getClass.getSimpleName, metrics.areaUnderPR,
				metrics.areaUnderROC)
		}

		val dtMetrics = Seq(dtModel).map{ model =>
			val scoreAndLabels = data.map { point =>
				val score = model.predict(point.features)
				(if (score > 0.5) 1.0 else 0.0, point.label)
			}
			val metrics = new BinaryClassificationMetrics(scoreAndLabels)
			(model.getClass.getSimpleName, metrics.areaUnderPR,
				metrics.areaUnderROC)
		}

		val allMetrics = metrics ++ nbMetrics ++ dtMetrics
		allMetrics.foreach{ case (m, pr, roc) =>
			println(f"$m, Area under PR: ${pr * 100.0}%2.4f%%, Area under ROC: ${roc * 100.0}%2.4f%%")
		}

    // 特征标准化 feature standardization
		val vectors: RDD[linalg.Vector] = data.map(lp => lp.features)
		val matrix: RowMatrix = new RowMatrix(vectors)
		val matrixSummary: MultivariateStatisticalSummary = matrix.computeColumnSummaryStatistics()
		println(matrixSummary.mean)
		println(matrixSummary.min)
		println(matrixSummary.max)
		println(matrixSummary.variance)
		println(matrixSummary.numNonzeros)

		val scaler = new StandardScaler(withMean = true, withStd =
			true).fit(vectors)
		val scaledData: RDD[LabeledPoint] = data.map(lp => LabeledPoint(lp.label,
			scaler.transform(lp.features)))
		println(data.first.features)
		println(scaledData.first.features)
		println((0.789131 - 0.41225805299526636)/ math.sqrt(0.1097424416755897))

		val lrModelScaled = LogisticRegressionWithSGD.train(scaledData, numIterations)
		val lrTotalCorrectScaled = scaledData.map { point =>
			if (lrModelScaled.predict(point.features) == point.label) 1 else
				0 }.sum
		val lrAccuracyScaled = lrTotalCorrectScaled / numData
		val lrPredictionsVsTrue = scaledData.map { point =>
			(lrModelScaled.predict(point.features), point.label)
		}
		val lrMetricsScaled = new BinaryClassificationMetrics(lrPredictionsVsTrue)
		val lrPr = lrMetricsScaled.areaUnderPR
		val lrRoc = lrMetricsScaled.areaUnderROC
		println(f"${lrModelScaled.getClass.getSimpleName}\nAccuracy: ${lrAccuracyScaled * 100}%2.4f%%\nArea under PR: ${lrPr * 100.0}%2.4f%%\nArea under ROC: ${lrRoc * 100.0}%2.4f%%")

		// additional features
		val categories = records.map(r => r(3)).distinct.collect.zipWithIndex.toMap
		val numCategories = categories.size
		println(categories)
		println(numCategories)

		val dataCategories = records.map { r =>
			val trimmed = r.map(_.replaceAll("\"", ""))
			val label = trimmed(r.size - 1).toInt
			val categoryIdx = categories(r(3))
			val categoryFeatures = Array.ofDim[Double](numCategories)
			categoryFeatures(categoryIdx) = 1.0
			val otherFeatures = trimmed.slice(4, r.size - 1).map(d => if
			(d == "?") 0.0 else d.toDouble)
			val features = categoryFeatures ++ otherFeatures
			LabeledPoint(label, Vectors.dense(features))
		}
		println(dataCategories.first)

		val scalerCats = new StandardScaler(withMean = true, withStd = true).fit(dataCategories.map(lp => lp.features))
		val scaledDataCats = dataCategories.map(lp => LabeledPoint(lp.label, scalerCats.transform(lp.features)))
		println(dataCategories.first.features)
		println(scaledDataCats.first.features)

		val lrModelScaledCats = LogisticRegressionWithSGD.
			train(scaledDataCats, numIterations)
		val lrTotalCorrectScaledCats = scaledDataCats.map { point =>
			if (lrModelScaledCats.predict(point.features) == point.label) 1 else
				0
		}.sum
		val lrAccuracyScaledCats = lrTotalCorrectScaledCats / numData
		val lrPredictionsVsTrueCats = scaledDataCats.map { point =>
			(lrModelScaledCats.predict(point.features), point.label)
		}
		val lrMetricsScaledCats = new BinaryClassificationMetrics(lrPredictionsVsTrueCats)
		val lrPrCats = lrMetricsScaledCats.areaUnderPR
		val lrRocCats = lrMetricsScaledCats.areaUnderROC
		println(f"${lrModelScaledCats.getClass.getSimpleName}\nAccuracy: ${lrAccuracyScaledCats * 100}%2.4f%%\nArea under PR: ${lrPrCats * 100.0}%2.4f%%\nArea under ROC: ${lrRocCats * 100.0}%2.4f%%")

		// using the correct form of data   k之一编码
		val dataNB = records.map { r =>
			val trimmed = r.map(_.replaceAll("\"", ""))
			val label = trimmed(r.size - 1).toInt
			val categoryIdx = categories(r(3))
			val categoryFeatures = Array.ofDim[Double](numCategories)
			categoryFeatures(categoryIdx) = 1.0
			LabeledPoint(label, Vectors.dense(categoryFeatures))
		}

		val nbModelCats = NaiveBayes.train(dataNB)
		val nbTotalCorrectCats = dataNB.map { point =>
			if (nbModelCats.predict(point.features) == point.label) 1 else 0
		}.sum
		val nbAccuracyCats = nbTotalCorrectCats / numData
		val nbPredictionsVsTrueCats = dataNB.map { point =>
			(nbModelCats.predict(point.features), point.label)
		}
		val nbMetricsCats = new BinaryClassificationMetrics(nbPredictionsVsTrueCats)
		val nbPrCats = nbMetricsCats.areaUnderPR
		val nbRocCats = nbMetricsCats.areaUnderROC
		println(f"${nbModelCats.getClass.getSimpleName}\nAccuracy: ${nbAccuracyCats * 100}%2.4f%%\nArea under PR: ${nbPrCats * 100.0}%2.4f%%\nArea under ROC: ${nbRocCats * 100.0}%2.4f%%")

		// 研究模型参数对性能辅助函数的影响以训练逻辑回归模型    investigate the impact of model parameters on performance
		// 训练逻辑回归模型的辅助函数   helper function to train a logistic regresson model
		def trainWithParams(input: RDD[LabeledPoint], regParam: Double, numIterations: Int, updater: Updater, stepSize: Double) = {
			val lr = new LogisticRegressionWithSGD
			lr.optimizer.setNumIterations(numIterations).setUpdater(updater).setRegParam(regParam).setStepSize(stepSize)
			lr.run(input)
		}
		// 创建 AUC 指标的辅助函数    helper function to create AUC metric
		def createMetrics(label: String, data: RDD[LabeledPoint], model: ClassificationModel) = {
			val scoreAndLabels = data.map { point =>
				(model.predict(point.features), point.label)
			}
			val metrics = new BinaryClassificationMetrics(scoreAndLabels)
			(label, metrics.areaUnderROC)
		}

		// 缓存数据以提高针对数据集的多次运行速度  cache the data to increase speed of multiple runs agains the dataset
		scaledDataCats.cache

		//不同的迭代次数 num iterations
		val iterResults = Seq(1, 5, 10, 50).map { param =>
			val model = trainWithParams(scaledDataCats, 0.0, param, new SimpleUpdater, 1.0)
			createMetrics(s"$param iterations", scaledDataCats, model)
		}
		iterResults.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }

		//不同的步长 step size
		val stepResults = Seq(0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
			val model = trainWithParams(scaledDataCats, 0.0, numIterations, new SimpleUpdater, param)
			createMetrics(s"$param step size", scaledDataCats, model)
		}
		stepResults.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }

		//不同的正则化参数 regularization
		val regResults = Seq(0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
			val model = trainWithParams(scaledDataCats, param, numIterations, new SquaredL2Updater, 1.0)
			createMetrics(s"$param L2 regularization parameter", scaledDataCats, model)
		}
		regResults.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }


		def trainDTWithParams(input: RDD[LabeledPoint], maxDepth: Int, impurity: Impurity) = {
			DecisionTree.train(input, Algo.Classification, impurity, maxDepth)
		}

		//研究熵杂质的树深度影响 investigate tree depth impact for Entropy impurity
		val dtResultsEntropy = Seq(1, 2, 3, 4, 5, 10, 20).map { param =>
			val model: DecisionTreeModel = trainDTWithParams(data, param, Entropy)
			val scoreAndLabels = data.map { point =>
				val score = model.predict(point.features)
				(if (score > 0.5) 1.0 else 0.0, point.label)
			}
			val metrics = new BinaryClassificationMetrics(scoreAndLabels)
			(s"$param tree depth", metrics.areaUnderROC)
		}

		dtResultsEntropy.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }

		// 研究 Gini(基尼) 杂质的树深度影响 investigate tree depth impact for Gini impurity
		val dtResultsGini = Seq(1, 2, 3, 4, 5, 10, 20).map { param =>
			val model = trainDTWithParams(data, param, Gini)
			val scoreAndLabels = data.map { point =>
				val score = model.predict(point.features)
				(if (score > 0.5) 1.0 else 0.0, point.label)
			}
			val metrics = new BinaryClassificationMetrics(scoreAndLabels)
			(s"$param tree depth", metrics.areaUnderROC)
		}
		dtResultsGini.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }

		// 研究朴素贝叶斯参数  investigate Naive Bayes parameters
		def trainNBWithParams(input: RDD[LabeledPoint], lambda: Double) = {
			val nb = new NaiveBayes()
			//设置平滑参数。默认值：1.0。
			nb.setLambda(lambda)
			nb.run(input)
		}
		//nb.setLambda(lambda) (0.001, 0.01, 0.1, 1.0, 10.0)
		val nbResults = Seq(0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
			val model = trainNBWithParams(dataNB, param)
			val scoreAndLabels = dataNB.map { point =>
				(model.predict(point.features), point.label)
			}
			val metrics = new BinaryClassificationMetrics(scoreAndLabels)
			(s"$param lambda", metrics.areaUnderROC)
		}
		nbResults.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }

		// 说明 交叉验证  illustrate cross-validation
		// 创建一个 60% 40% 的 traintest 数据拆分 create a 60% / 40% train/test data split
		val trainTestSplit = scaledDataCats.randomSplit(Array(0.6, 0.4), 123)
		val train = trainTestSplit(0)
		val test = trainTestSplit(1)
		// 现在我们使用“train”数据集训练我们的模型，并在看不见的“test”数据上计算预测  now we train our model using the 'train' dataset, and compute predictions on unseen 'test' data
		//此外，我们将评估正则化在训练和测试数据集上的不同性能 in addition, we will evaluate the differing performance of regularization on training and test datasets
		val regResultsTest = Seq(0.0, 0.001, 0.0025, 0.005, 0.01).map { param =>
			//逻辑回归模型
			val model = trainWithParams(train, param, numIterations, new SquaredL2Updater, 1.0)
			//创建 AUC 指标的辅助函数
			createMetrics(s"$param L2 regularization parameter", test, model)
		}
		regResultsTest.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.6f%%") }

		// training set results
		val regResultsTrain = Seq(0.0, 0.001, 0.0025, 0.005, 0.01).map { param =>
			val model = trainWithParams(train, param, numIterations, new SquaredL2Updater, 1.0)
			createMetrics(s"$param L2 regularization parameter", train, model)
		}
		regResultsTrain.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.6f%%") }
	}

}