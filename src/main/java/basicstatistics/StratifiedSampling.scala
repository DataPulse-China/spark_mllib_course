package basicstatistics

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

/**
 * stratifiedSampling  分层抽样
 *  单词
 *      Stratified sampling 分层取样
 *      sampleByKey 样本通过键抽样
 *      sampleByKeyExact 样本通过键精确抽样
 *      fractions  片段
 *
 *
 *      分层取样（Stratified sampling）顾名思义，就是将数据根据不同的特征分成不同的组，然后按特定条件从不同的组中获取样本，
 *  并重新组成新的数组。分层取样算法是直接集成到键值对类型 RDD[(K, V)] 的 sampleByKey 和 sampleByKeyExact 方法，
 *  无需通过额外的 spark.mllib 库来支持。
 *
 */
object StratifiedSampling {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val sc = new SparkContext("local[*]", "")
    /**
     * sampleByKey 方法
     *
     *    sampleByKey 方法需要作用于一个键值对数组，其中 key 用于分类，value可以是任意数。
     * 然后通过 fractions 参数来定义分类条件和采样几率。fractions 参数被定义成一个 Map[K, Double] 类型，
     * Key是键值的分层条件，Double 是该满足条件的 Key 条件的采样比例，1.0 代表 100%。
     *
     * 首先，导入必要的包：
        import org.apache.spark.SparkContext
        import org.apache.spark.SparkContext._
        import org.apache.spark.rdd.PairRDDFunctions
     */

    //接下来，这里为了方便观察，没有从iris数据集中取数据，而是重新创建了一组数据，分成 “female” 女 和 “male” 男性 两类：
    val data: RDD[(String, String)] = sc.makeRDD(Array(
      ("female","Lily"),
      ("female","Lucy"),
      ("female","Emily"),
      ("female","Kate"),
      ("female","Alice"),
      ("male","Tom"),
      ("male","Roy"),
      ("male","David"),
      ("male","Frank"),
      ("male","Jack"),
      ("male","li")))

    /**
     * fractions  抽样率的特定关键字的映射 [小部分]
     *
     *    然后，我们通过  参数来定义分类条件和采样几率：这里，设置采取60%的female和40%的male，
     * 因为数据中female和male各有5个样本，所以理想中的抽样结果应该是有3个female和2个male。
     * 接下来用sampleByKey进行抽样：
     */

    val fractions: Map[String, Double] = Map("female" -> 0.6, "male" -> 0.4)

    /**
     * sampleByKey 样本通过键抽样方法
     *
     *   参数
     *     withReplacement  whether to sample with or without replacement
     *        样品是否更换  false
     *     fractions map of specific keys to sampling rates
     *        抽样率的特定关键字的映射  fractions Map("female" -> 0.6, "male" -> 0.4)
     *     seed seed for the random number generator
     *        为随机数生成器生成种子  1
     */

    val approxSample: RDD[(String, String)] = data.sampleByKey(false, fractions, 1)

    println("分层抽样 Stratified sampling  sampleByKey 方法")
    approxSample.foreach(println)

    /**
     * (male,li)
       (female,Lucy)
       (male,David)
       (male,Frank)
       (female,Alice)
       (female,Emily)
       (male,Roy)
       (female,Lily)
       (female,Kate)
     */


    /**
     * sampleByKeyExact 样本通过键精确抽样方法
     *
     *    sampleByKey 和 sampleByKeyExact 的区别在于 sampleByKey 每次都通过给定的概率以一种类似于掷硬币的方式来
     * 决定这个观察值是否被放入样本，因此一遍就可以过滤完所有数据，最后得到一个近似大小的样本，但往往不够准确。
     * 而 sampleByKeyExtra 会对全量数据做采样计算。对于每个类别，其都会产生 （fk⋅nk）个样本，其中fk是键为k的样本类别
     * 采样的比例；nk是键k所拥有的样本数。 sampleByKeyExtra 采样的结果会更准确，有99.99%的置信度，但耗费的计算资源也更多。
     *
     *  （fk⋅nk）
     *        其中fk是键为k的样本类别采样的比例
     *        nk是键k所拥有的样本数
     */

    println("sampleByKeyExact 样本通过键精确抽样方法")
    data.sampleByKeyExact(false,fractions,1).foreach(println)

    /**
     * 结果集
     *
     * (female,Lucy)
       (male,Roy)
       (female,Emily)
       (male,li)
       (male,David)
       (female,Alice)
     */








    /**
     *          参数              含义
     *    withReplacement    每次抽样是否有放回
     *
     *    fractions          控制不同key的抽样率
     *
     *    seed               随机数种子
     */



  }
}
