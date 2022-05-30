package xml

object Demo1 {
  def main(args: Array[String]): Unit = {
    val foo = <foo><bar type="greet">hi</bar><bar type="count">1</bar><bar type="color">yellow</bar></foo>

    foo.text.foreach(println)

    println(foo \ "bar".toString)



  }
}
