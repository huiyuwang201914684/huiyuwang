#                                                                                                  <center>作业1实验报告  </center>

## 一、    实验内容

​        利用K-Means、Affinity propagation、Mean-shift、Spectral clustering、Ward hierarchical clustering、Agglomerative clustering、DBSCAN、Gaussian mixtures八种聚类算法在load_digits、fetch_20newsgroups两个数据集上的进行聚类，并用评估这几种聚类方法的NMI、homogeneity、completeness。

## 二、    实验目的

（1）了解常用聚类算法的基本原理原理。
（2）了解K-Means、Affinity propagation、Mean-shift、Spectral clustering、Ward hierarchical clustering、Agglomerative clustering、DBSCAN、Gaussian mixtures这8种聚类算法的基本思想以及优缺点
（3）掌握不同聚类算法在不同数据集上的运行方式以及评估指标。

## 三、    实验原理

​        聚类就是按照某个特定准则把一个数据集分割成不同的类，最大化同一个类中的数据对象的相似性，同时最大化不在同一个类中的数据对象的差异性。

K-Means：按照样本之间的欧式距离大小，将样本集划分为k个簇，采用启发式迭代的方法最小化平方误差，最终得到k个类。
Affinity propagation：不需要先确定聚类的数目，而是把所有的数据点都看成潜在意义上的聚类中心，根据数据点之间的相似度来进行聚类。
Mean-shift：一般是指一个迭代的步骤,即先算出当前点的偏移均值,移动该点到其偏移均值,然后以此为新的起始点,继续移动,直到满足一定的条件结束。
Spectral clustering：是一种基于图论的聚类方法，将带权无向图划分为两个或两个以上的最优子图，使子图内部尽量相似，而子图间距离尽量距离较远，以达到常见的聚类的目的。
Hierarchical clustering：自下而上的方法，每个观察都在它自己的集群中开始，并且当一个集群向上移动时，它们将被合并；自上而下的方法，所有观察都在一个集群中开始，并且当一个集体向下移动时，递归地执行分割。
DBSCAN：是一种基于密度的聚类算法，这类密度聚类算法一般假定类别可以通过样本分布的紧密程度决定。
Gaussian mixtures：多个高斯分布的线性叠加能拟合非常复杂的密度函数；通过足够多的高斯分布叠加，并调节它们的均值，协方差矩阵，以及线性组合的系数，可以精确地逼近任意连续密度。

## 四、    实验步骤

### （1）  实验环境搭建：

​        在Windows系统下安装python3.7.3和集成开发环境pycharm,并导numpy、scipy、matplotlib、scikit-learn这4个工具包 。

### （2）  实现过程：

​        模块导入、加载数据集、数据集降维（使用sklearn内的TfidfVectorizer 将文本转化为tf-idf向量形式表示）、选择聚类算法、参数调整、结果输出。

## 五、    实验结果与分析

### （1）load_digits数据集上
![1](https://github.com/huiyuwang201914684/huiyuwang/blob/master/cluster/1.png)


### （2）fetch_20newsgroups数据集上
![2](.\2.png)


## 六、    结论

  
​        Sklearn对于一些聚类方法，直接调用就可以使用，非常方便。但是对于同一个文本集，不同的聚类方法效果不同，有好有坏，而一些聚类算法比如高斯方法，如果聚类个数增多，各项值也会提升，但是聚类个数过多就会报错无法聚类。

​        从结果上来看，效果比较好的是MeanShift和Spetral，如果样本集的密度不均匀、聚类间距差相差很大时，聚类质量较差，这时用DBSCAN聚类一般不适合。


