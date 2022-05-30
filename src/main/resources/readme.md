| 数据文件名称 | 说明                                                         |
| ------------ | ------------------------------------------------------------ |
| u.data       | 用户评分数据：包含4个字段:userid (用户🆔id) 、item id (项目id) 、 rating(评分)、timestamp(日期时间) |
| u.item       | 电影的数据：具有很多个字段，本书主要使用前2个字段：movie id（电影id） movie title (电影片名) |
|              |                                                              |

- 查看userid 字段的统计信息
> 这个方法可以了解每个字段的数据特性
> (count: 100000, mean: 462.484750, stdev: 266.613087, max: 943.000000, min: 1.000000)
- count 计数
- mean 平均
- stdev 标准偏差
- max 最大值
- min 最小值
  

**user** id[(count: 100000, mean: 462.484750, stdev: 266.613087, max: 943.000000, min: 1.000000)]
**item** id[(count: 100000, mean: 425.530130, stdev: 330.796702, max: 1682.000000, min: 1.000000)]



<table>
	<thead>
		<tr>
			<th>方法名</th>
			<th>方法含义</th>
			<th style="text-align:center">返回值类型</th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<td>count</td>
			<td>列的大小</td>
			<td style="text-align:center">long</td>
		</tr>
		<tr>
			<td>mean</td>
			<td>每列的均值</td>
			<td style="text-align:center">vector</td>
		</tr>
		<tr>
			<td>variance</td>
			<td>每列的方差</td>
			<td style="text-align:center">vector</td>
		</tr>
		<tr>
			<td>max</td>
			<td>每列的最大值</td>
			<td style="text-align:center">vector</td>
		</tr>
		<tr>
			<td>min</td>
			<td>每列的最小值</td>
			<td style="text-align:center">vector</td>
		</tr>
		<tr>
			<td>normL1</td>
			<td>每列的L1范数</td>
			<td style="text-align:center">vector</td>
		</tr>
		<tr>
			<td>normL2</td>
			<td>每列的L2范数</td>
			<td style="text-align:center">vector</td>
		</tr>
		<tr>
			<td>numNonzeros</td>
			<td>每列非零向量的个数</td>
			<td style="text-align:center">vector</td>
		</tr>
	</tbody>
</table>
