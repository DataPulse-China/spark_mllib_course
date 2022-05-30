package stumbleupon

import org.apache.spark.SparkContext

object MOdelClassification {
  def main(args: Array[String]): Unit = {

    //这里以 logistic 回归模型为例(其他模型的处理方法类似):
    val sc = new SparkContext("local[*]", "")
    val records = sc.textFile("/Users/manpreet.singh/Sandbox/codehub/github/machinelearning/breeze.io/src/main/scala/sparkMLlib/AllInOneClassification/train_noheader.tsv").map(line => line.split("\t"))

    records.map(
      item => {

      }
    )


  }
}
