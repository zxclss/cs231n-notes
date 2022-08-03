> 原课程网址：https://cs231n.github.io/linear-classify/
> 翻译：Colopen  
> 校对：Colopen

# Linear Classification

在上一节中，我们介绍了图像分类问题，即从一组固定类别标签中选择一个分配给图像。此外，我们还介绍了k-Nearest Neighbor(kNN)分类器，该分类器通过将测试图像与训练集中的图像进行比较来进行分类。正如我们所见，kNN有着相当多的缺陷：

1. 分类器必须记住并存储所有的训练数据，以便未来与测试数据进行比较。空间效率十分低下，因为数据集的大小可能很容易打到千兆字节。  
2. 对一张测试图片进行分类的计算开销十分大，因为它需要与整个训练集中的图片一一进行比较。

**Overview.** 我们现在要实现一个更强大的图像分类器，并最终将自然地拓展到整个神经网络和卷积神经网络。该方法有两个主要组成部分：一个可以将原始数据映射到类别分数的**评分函数(score function)**，以及一个量化预测分类标签的得分与真实标签之间一致性的**损失函数(loss function)**。接下来我们会把该问题转化为优化问题，在这个优化问题中，我们将更新评分函数的参数值，以便能够最小化损失函数。

## Parameterized mapping from images to label scores

该方法的第一个重要组成部分是定义评分函数，评分函数用来将图片的像素值映射到各个分类标签的得分上（得分高低代表图像属于该类别的可能性高低）。我们将用一个具体例子来实现这个方法。正如之前一样，我们先假设有一个包含很多图片的训练集 $x_i\in R^D$，且每张图片都有一个分类标签 $y_i$，其中 $i=1 \cdots N,y_i\in 1 \cdots K$。这表示我们有N个图片样例（每个样例的维度为D）和K个独立的分类标签。例如CIFAR-10的训练集，有 N=50,000 张图片，每张图片有 D = 32 x 32 x 3 = 3072个像素，和K=10个独立的分类标签（狗，猫，车等等）。我们现在将定义一个把原始图像的像素值映射到各个分类标签得分的评分函数 $f: R^D \mapsto R^K$。

**Linear classifier.** 在这个模型中，我们将从最简单的函数开始，线性映射：

$$
f(x_i, W, b) = Wx_i + b
$$

在上述方程中，$x_i$ 是将图像 i 像素拉成 [D x 1] 形状而形成的列向量，而矩阵 $\mathbf W_{K \times D}$ 和向量 $\mathbf b_{K\times1}$ 则是函数的**参数(parameters)**。在CIFAR-10数据集中，$x_i$包含第i张图片的所有像素信息，且这些信息被拉成一个 [3072 x 1] 的列向量。$\mathbf W$ 大小为 [10 x 3072]，$\mathbf b$ 的大小为 [10 x 1]。所以，这3072个数字（原始像素值）输入函数后，就会输出10个数字（分类标签得分）。参数 $\mathbf W$ 常被称作 **权重(weights)**，$\mathbf b$ 常被称作 **偏置向量(bias vector)**，因为他不和原始数据 $x_i$ 产生联系，但会影响输出的数值。然而，你经常会听到人们混用 *权重* 和 *参数* 这两个术语。

有几件事需要注意：

- 

## Interpreting a linear classifier
## Loss function
### Multiclass Support Vector Machine loss
## Practical Considerations
## Softmax classifier
## SVM vs. Softmax
## Interactive web demo
## Summary
## Further Reading