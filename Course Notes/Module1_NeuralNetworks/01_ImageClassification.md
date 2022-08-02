> 原文网址：https://cs231n.github.io/classification/   
> 译：Colopen

# Image Classification

**Motivation**. 在本节中，我们将介绍**图像分类**(**image classification**)问题。图像分类问题的主要任务是，为**输入图**像(**input image**)从一组已有固定的分类标签集合中，选择一个作为该图像的**分类标签**(**label**)。这也是**计算机视觉**(**Computer Vision**)中的核心问题之一。虽然图像分类问题看上去很简单，但是其却具有大量的实际应用价值。而且，计算机视觉的许多看似不同的任务(e.g. 对象检测、切割)，都是可以归结为图像分类任务的。

**Example**. 例如在下图中，图像分类模型读取该图片，并生成该图片属于集合 {cat, dog, hat, mug} 中各个标签的概率。需要注意的是，对于计算机来说，图像是一个由数字组成的巨大的3维数组。在本例中，猫的图像宽 248 pixels，高 400 pixels，具有红色、绿色、蓝色（或简称 RGB）三个颜色通道。 因此，图像由 248 x 400 x 3 个数字组成，或总共 297,600 个数字。 每个数字都是一个整数，范围从 0（黑色）到 255（白色）。 我们的任务是将这 25 万个数字变成一个标签，例如“猫”。

![](https://cs231n.github.io/assets/classify.png)

图像分类中的任务是预测给定图像的单个标签（或者给出属于一系列不同标签的可能性分布）。图像是由 0 到 255 的整数构成的 3 维数组，数组大小为 Width x Height x 3。（其中 3 代表红、绿、蓝三个颜色通道）

----------------

**Challenges**. 对于人来说，识别一个视觉概念（例如猫）的任务是十分简单的，然而从计算机视觉算法的角度来看就值得深思了。现在图像不再是一个图像，而是一个原始表示为亮度值的 3-D 数组，然后思考如下问题：

- 视角变化（Viewpoint variation）：对于同一个目标物体，摄像机可以从多个角度来呈现。  
- 大小变化（Scale variation）：物体可视的大小是会变化的（不仅是在图片中，在真实世界中大小也是变化的）。  
- 形变（Deformation）：很多目标物体并非刚体，可能会发生扭曲变形。  
- 遮挡（Occlusion）：目标物体可能被挡住，有时候可能只有很小的一部分（可以小到几个像素）是可见的。  
- 光照条件（Illumination conditions）：在像素层面上，光照的影响非常大。  
- 背景干扰（Background clutter）：目标物体可能混入他的背景之中，导致很难分辨出来。  
- 类内差异（Intra-class variation）：目标物体的类别数可能也很多，比如椅子。这一类物体有许多不同的对象，每个类别都有自己特定的外形。

![](https://pic2.zhimg.com/80/1ee9457872f773d671dd5b225647ef45_1440w.jpg)

对上述所有变化及其组合，一个好的图像分类模型必须能够维持住分类的稳定，同时对类与类之间的差异足够敏感。

-----------------

**Data-driven approach**. 如何写出一个图像分类算法？这个不像写一个排序算法，对于怎样写出一个识别猫猫的算法，不是那么易于上手。因此，比起直接在代码中指明各个类别具体是什么样的，应该更倾向于像教小孩看图识物的方法一样：我们将给计算机提供各个类别的大量实例，然后实现一个学习算法，让计算机通过这些实例，学习到各个类别的视觉特征。因为该方法第一步是要先收集大量已做好标签的图片作为训练集，所以该方法被称为**数据驱动方法(Data-driven approach)**。下图是简要展现了该数据集的模样。

![](https://cs231n.github.io/assets/trainset.jpg)

上图是一个设有 4 个视觉类别的训练集的例子。但在实际中，我们可能有上万个分类，每个分类可能有上百万张图片。

-----------------

**The image classification pipeline**. 图像分类的任务是输入一个元素为像素的数组，然后给它分配一个分类标签，完整流程如下：

- **输入(Input)**：我们的输入是由一个 N 张图片构成，每张图片的标签是K种分类标签中的一种。该数据称为**训练集**(training data)。  
- **学习(Learning)**：我们的任务是用训练集学习，得知每个分类标签是怎样的。该步骤称为**训练一个分类器或学习一个模型**(training a classifier or learning a model)  
- **评估(Evaluation)**：最后，我们通过让分类器作用在一个由此前未使用于训练的新图像构成的数据集进行预测，来评估分类器的质量。接着我们将这些图像的真正分类标签和分类器通过输入预测到的分类标签进行比较。显然，我们希望尽可能多的预测结果可以匹配上真正的答案(ground truth)。

## Nearest Neighbor Classifier

作为要介绍的第一个方法，我们将实现一个 **Nearest Neighbor 分类器**。这个分类器与**卷积神经网络**(**Convolutional Neural Networks, CNN**)毫无关系，并且在实际中用的也极少。但是通过学习该分类器，可以帮助我们对于图像分类问题的方法有个基本的认识。

**Example image classification dataset: CIFAR-10.** CIFAR-10数据集是一个非常流行的图像分类数据集。该数据集包含了 60,000 张大小为 32 x 32 的图片。这些图片共分为 10 个类别，每张图片都从属于唯一一个类别（例如“飞机，手机，鸟”等）。这 60,000 张图片被划分为 50,000 张图片的训练集和 10,000 张图片的测试集。在下面的图片中，你可以看到分别来自 10 个类别的 10 张图片：

![](https://cs231n.github.io/assets/nn.jpg)

左图：一些来自数据集 CIFAR-10 的图片；右图：第一列展示了一些测试集中的图片，后面紧接的是训练集中与该测试图像(逐像素差)最相邻的10张图像。

------------------

假设现在我们已经有了 CIFAR-10 50,000 张图片的训练集（每个分类标签下有 5,000张图片），我们希望对剩下的 10,000 张测试图片进行分类。Nearest Neighbor 分类器将会拿一张测试图片与整个训练集中的所有图片进行比较，然后将他认为最相似的那个训练集图片的标签赋给这张测试图片。上述图片的右侧部分就展示了这样的结果。注意到 10 张测试图片里，只有 3 张图片被正确分类。比如第 8 行的马头被分类为一辆红色跑车，可能是因为黑色背景图过于突出。因此这张马的图片被错误的分类为了车子。

那么应该如何比较两张图片呢？在本例中，两张图片是以两个 32 x 32 x 3 的方格表示的。一个十分简单的方法是一个像素一个像素进行比较，然后将所有像素的差异做一个总和，表示为整张图片的差异。话句话说，就是将两张图片先转化为两个向量 $I_1, I_2$，然后计算他们的 **L1 距离**

$$
d_1(I_1, I_2) = \sum_p |I_1^p - I_2^p|
$$

这里的求和是针对所有像素的。下图是比较流程的可视化：

![](https://cs231n.github.io/assets/nneg.jpeg)

上图用**逐像素差**和**L1距离**来比较两张图片（在这个例子中只有一个颜色）。两张图片逐像素相减，然后对所有差求和。如果两张图片是相同的，则结果为0。但要是两张图片差异过大，那结果也会变得很大。

-------------------

考虑一下我们应该如何用代码实现这个分类器。首先，把CIFAR-10数据集，用四个数组分别加载进内存中：训练数据的图像/标签 以及 测试数据的图像/标签。在如下代码中，`Xtr` (50,000 x 32 x 32 x 3) 存储着所有训练集中的图像，`Ytr` 是对应的长度为 50,000 的1维数组，存储着所有的训练集中的标签。

```py
Xtr, Ytr, Xte, Yte = load_CIFAR10('data/cifar10/') # a magic function we provide
# flatten out all images to be one-dimensional
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 3072
```
现在我们已经将所有的图片拉升成行向量，下面将展示如何训练和评估模型：

```py
nn = NearestNeighbor() # create a Nearest Neighbor classifier class
nn.train(Xtr_rows, Ytr) # train the classifier on the training images and labels
Yte_predict = nn.predict(Xte_rows) # predict labels on the test images
# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
print 'accuracy: %f' % ( np.mean(Yte_predict == Yte) )
```
我们常常用衡量正确预测得分的**准确率**作为评价准则。请注意，我们构建的所有分类器都应当满足这一公共 API：有 `train(X,y)` 函数，用于学习数据和标签。从内部来看，类应该实现的是一些关于标签和标签如何被预测的模型。然后，有一个 `perdict(X)` 函数，用于接受新的数据，然后输出预测的分类标签。当然，我们忘了一个核心的事情——分类器自身的实现。下面是一个简单的 L1距离的 Nearnest Neighbor分类器 模板

```py
import numpy as np

class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y

  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

    # loop over all test rows
    for i in range(num_test):
      # find the nearest training image to the i'th test image
      # using the L1 distance (sum of absolute value differences)
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
      min_index = np.argmin(distances) # get the index with smallest distance
      Ypred[i] = self.ytr[min_index] # predict the label of the nearest example

    return Ypred
```

如果你运行上述代码，会发现该分类器在 CIFAR-10 数据集上的精确度只达到 **38.6%** 。虽然这样的精确度比起随机猜测（只有10%的精确度）要高上不少，但与人工分类的精确度（[据估计有94%](https://karpathy.github.io/2011/04/27/manually-classifying-cifar10/)）相比，或者与CNN卷积神经网络能达到的95%还是相差甚远。(查看基于CIFAR-10数据的Kaggle算法竞赛[排行榜](https://www.kaggle.com/c/cifar-10/leaderboard))

**The choice of distance.** 计算向量间距离的方法还有许多。另一个常用于替代L1距离的方法是 **L2距离**。在几何角度来看，L2距离是在计算两个向量间的**欧氏距离**。L2距离的计算公式如下：

$$
d_2(I_1,I_2) = \sqrt{
  \sum_p (I_1^p - I_2^p)^2
}
$$

换句话说，我们仍在像之前一样计算像素间的差值，但是这次我们对差值进行了平方才加起来了，最后还开根号了。在numpy里，使用上述的代码，我们仅仅只需要替换简单的一行即可。计算距离的代码部分修改如下：

```py
distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))
```

注意到上面使用了 `np.sqrt` 函数，但在实际 nearest neighbor 应用中，我们往往不会开根号，因为平方根函数是一个 *单调函数*。这就意味着，他虽然改变了L2距离的绝对大小，但是保留了L2距离之间的相对大小，因此 nearest neighbor 有或没有他都是一样的。如果你在CIFAR-10数据集上运行 Nearest Neighbor 分类器，你将会得到 **35.4%** 的准确度（就比L1距离生成的结果少一点点）。

**L1 vs. L2**. 比较两者度量方式是很有意义的。当面对两个向量间的差异时，L2距离比起L1距离更加不能容忍这些差异。那就是说，L2距离比起一个巨大程度的差异，更加偏爱多个中等程度的差异。L1/L2距离（也称为一对图像之间差异的L1/L2范数）是 [p范数](https://planetmath.org/vectorpnorm) 最常用的特殊形式。

## k-Nearnest Neighbor Classifier

你可能已经注意到了，当我们想要做出预测时，仅仅只用最相似的图片的标签作为预测的标签，这个行为非常奇怪。事实上，使用k-Nearnest Neighbor分类器，往往就能做的更好。它的思想很简单：我们会找到前k个最相似的图片，而不是像之前一样只找到唯一最相似的图片，然后让他们针对测试图片进行投票，把票数最高的作为本次预测的标签。特殊情况，当 *k=1* 时，k-NN分类器就会退化成NN分类器。从直观感受上来看，k的值越高，分类的效果就越平滑，这样就会使分类器对于异常值更加具有抵抗力。

![](https://cs231n.github.io/assets/knn.jpeg)

上图是举了一个数据集的例子，来展现同一个数据集在NN分类器和5-NN分类器下区别。该例子使用了2维的点坐标和3个分类标签（红，绿，蓝）。对于不同区域进行染色，就可以看到用L2距离的分类器的决策边界了。白色区域的点是分类模糊的例子（图像与两个以上的分类标签绑定）。注意到 NN分类器 的例子中，异常数据点（例如在蓝色点中间的绿色点）会创造出一个非正确预测的小岛屿，然而 5-NN分类器 将这些不规则都平滑了，使得该模型对于训练集的数据有更好的 **泛化(generalization)** 能力。同样还要注意到，在5-NN分类器中的灰色区域是由于近邻标签的最高票数相同导致的（e.g. 邻居有2个红色，2个蓝色，1个绿色）。

-------------------

事实上，你总是会用到k-NN分类器。但是k值应该取多少呢？我们接下来回来讨论这个问题。

## Validation sets for Hyperparameter tuning

k-NN分类器需要设定k值。但是k取何值时最佳？此外，我们还可以选择不同的距离函数：L1/L2范数。除此之外还有很多我们没考虑到的选择（e.g. 点积）。这些选择被称为 **超参数(hyperparameter)**，他们在基于数据学习的机器学习算法的设计中，出现的非常频繁。如何选择和设置这些参数，往往不是那么显而易见。

你可能会想尝试许多不同的参数，然后看看哪个表现最优。这是一个不错的注意，我们也确实是这么做的。但是这必须格外小心仔细。尤其要注意，**不能用测试集来调整超参**。无论你何时设计机器学习算法，都应该把测试集当做一个非常珍惜的资源，除非到项目的最终阶段，否则永远不要碰它。否则，你利用测试集来调优超参，使得算法的效果看起来不错，但真正的危险是：如果你部署改模型，那最终模型的表现可能会远低于预期。这种情况，我们称为算法对测试集 **过拟合(overfit)** 了。从另一个角度来看待该问题，如果你在测试集上对超参进行调优，你就会将测试集当做训练集来使用，因此算法在测试集上的表现将会远远优于当你部署该模型时呈现的效果。但是如果你最后只使用测试集一次，那么该模型仍然可以很好的度量你所设计的分类器的**泛化(generalization)**能力（我们会后续课程中看到更多围绕泛化能力的讨论）。

> 在测试集上的评估一般是在最后的最后，并且只需进行一次评估即可。

幸运的是，有一个正确的超参调优方法，他自始至终都不会用到测试集。这个方法是从我们的测试集中取出一部分来对超参调优，我们称该子集为 **验证集(validation set)**。以CIFAR-10作为一个例子，我们可以用训练集中的49,000个图像用作训练，剩下的1,000张图像用作验证。验证集其实就是作为假的训练集来调优的。

下方是以CIFAR-10为例子的超参调优代码：

```py
# assume we have Xtr_rows, Ytr, Xte_rows, Yte as before
# recall Xtr_rows is 50,000 x 3072 matrix
Xval_rows = Xtr_rows[:1000, :] # take first 1000 for validation
Yval = Ytr[:1000]
Xtr_rows = Xtr_rows[1000:, :] # keep last 49,000 for train
Ytr = Ytr[1000:]

# find hyperparameters that work best on the validation set
validation_accuracies = []
for k in [1, 3, 5, 10, 20, 50, 100]:

  # use a particular value of k and evaluation on validation data
  nn = NearestNeighbor()
  nn.train(Xtr_rows, Ytr)
  # here we assume a modified NearestNeighbor class that can take a k as input
  Yval_predict = nn.predict(Xval_rows, k = k)
  acc = np.mean(Yval_predict == Yval)
  print 'accuracy: %f' % (acc,)

  # keep track of what works on the validation set
  validation_accuracies.append((k, acc))
```

在程序的最后，我们可以作图分析出哪个k值表现最好。接着我们会用这个k值进行训练，并最后在测试集上完成一次模型评估。

> 吧训练集划分成训练集和验证集。用验证集来对超参进行调优。最后在测试集上跑一次并报告模型的性能。

**Cross-validation**. 在有些时候，你的训练集（包括验证集在内）可能很小，人们会使用一个更加复杂的技术来完成对超参的调优，这种方法称为 **交叉检验(cross-validation)**。继续我们先前的例子，交叉检验的思想是：通过迭代不同的验证集，并平均这些验证集的性能，从而估计出一个效果更好，噪声更小的k值，而不是把前1,000个数据点当做验证集，剩下的作为训练集这么简单。例如，在5份交叉检验中，我们会把训练集分成5等分，用其中的4用作训练，1个用作验证。接着，我们循环选择哪一个用作验证，评估性能，并最后取5次验证得到的性能的平均值作为算法的验证结果。

![](https://pic1.zhimg.com/80/6a3ceec60cc0a379b4939c37ee3e89e8_1440w.png)

上图是一个5份交叉检验对超参k调优的例子。对于k的每一个取值，我们会在4份上训练，并在第5份上验证。因此，对于每个k值，我们可以得到5个准确度（y轴表示准确度，每个结果是一个点）。对于每个k值所得的5个准确度，绘制出他们的平均值作为趋势线，绘制出他们的标准差作为误差线。注意到本例中，k=7时算法在数据集上的表现最好（对应图中的准确度峰值）。如果我们把训练集额外多分5份，则会看到更加平滑的曲线（即噪声更少）。

--------------------

**In practice.** 在实际应用中，人们不是很喜欢用交叉检验，一般只会简单地划分验证集，因为交叉检验很耗费计算资源。一般来讲，会将训练集的50%-90%作为训练集，剩余部分作为验证集。具体多少是根据情况而定的：例如如果超参的数量很多，那你肯定希望验证集尽可能大。如果验证集小的话（或许只有几百条数据这么多），那还是使用交叉检验更好。关于交叉检验应该几等分，一般而言会用3/5/10等分交叉检验。

![](https://cs231n.github.io/assets/crossval.jpeg)

上图是常用的数据分割模式。一开始只有一个训练集和一个测试集。训练集被均分（例如这里的五等分）。1-4份变成训练集，剩下一份（黄色的那一份）用作验证集来调优超参。而在交叉检验中，各份会轮流作为验证集，具体参考5份交叉检验。最后模型训练完毕，最优超参选定完毕，模型才会在测试集（红色）上进行评估。

---------------------

### Pros and Cons of Nearest Neighbor classifier

现在值得我们去思考Nearest Neighbor分类器的优劣了。显然，一个好处是Nearest Neighbor分类器实现简单，易于理解。此外，分类器不需要时间去训练，因为训练期间唯一要做的就是把训练数据存储和索引起来。然而，我们在测试期间需要耗费大量的计算资源，因为每对一个测试样例分类，就需要拿它与整个训练集作比较。这显然是一个缺点，因为在实际应用中，我们更希望测试效率远高于训练效率。事实上，在之后课程中会提到的深度神经网络，就将这个权衡走向了另一个极端：他们会消耗大量计算资源去训练，但一旦训练完毕，对新的测试样例进行分类就会很快。这种操作模式更加符合实际使用的需求。

顺带一提，Nearest Neighbor分类器的计算复杂度是一个活跃的研究领域，有若干个加速数据集中Nearnest Neighbor查找的 **Approximate Nearest Neighbor (ANN)** 算法和库（e.g. FLANN）。这些算法可以在检索过程对Nearest Neighbor的正确性及其空间/时间复杂度进行权衡，并且通常依赖构造一棵kdtree或跑一次k-means算法的预处理/索引操作。

Nearest Neighbor分类器在一些场景中（尤其是当数据是低维的）有时也是一个不错的选择，但在图像分类的实际运用中，并没有适当的应用场景。一大原因是由于图像是高维度的数据（通常他们包含了许多像素），并且高维空间的向量间距离非常反直觉。下面图片说明了这一点，我们先前实现的基于像素的L2范数相似度与人能感知到的相似度非常不同：

![](https://pic3.zhimg.com/80/fd42d369eebdc5d81c89593ec1082e32_1440w.png)

在高纬度数据中（尤其是图像），基于像素的距离非常反直觉。左1是原始图像，后面3张与左1在L2像素距离上是相等的（人为构造的）。显然逐像素相似与感官上和语义上的相似并不完全有关。

------------------

这里还有一个视觉化应用可以向你说明，用像素差异去比较图片是远远不够的。这是一个被叫做[t-SNE](https://lvdmaaten.github.io/tsne/)的可视化技术，它才用的是CIFAR-10的图片，将这些图片放入一个二维坐标，这就可以很好的保留住图片与图片之间的距离（即差异值）。在这个视觉应用中，越相邻图片在逐像素L2距离上就越小。

![](https://cs231n.github.io/assets/pixels_embed_cifar10.jpg)

CIFAR-10图片用t-SNE技术放入了一个二维坐标中。图片中相邻的图像被认为在逐像素L2距离上也相近。注意到，这些图像的背景而不是他们的语义，对他们在坐标中的位置影响很大。点击[这里](https://cs231n.github.io/assets/pixels_embed_cifar10_big.jpg)查看更大图片。

------------------

尤其是可以注意到，这些图片的排布更像是一种颜色分布函数，或者说是一种基于背景的，而不是基于语义主体。比如说，因为背景都非常亮，一只狗可能就离一只青蛙很近。我们理想情况下是希望同类图片可以聚集在一起，而不用担心被不相关的特征和变化所影响（例如背景）。然而，为了打到这个目的，我们将不得不止步于原始像素的比较。

## Summary

总结一下：

- 介绍了 **图像分类(Image Classification)** 问题，即给我们一个被标注了分类标签的图像构成的数据集，要求算法能预测没有标签的图像的分类标签，并度量预测的准确性。  
- 介绍了一个简单地分类器 **最近邻分类器(Nearest Neighbor classifier)**。分类器中存在许多**超参数(hyper-parameters)**（例如k值，比较样例的距离范数），他们都与这个分类器息息相关，并且选择起来不是一件轻而易举的事。  
- 正确设置超参的方法是：把原始训练集分成训练集和验证集，使用不同的超参值训练和验证，最后保留下在验证集上效果最好的值。  
- 如果训练数据很少，可以使用**交叉检验(cross-validation)**方法，该方法可以帮助减少在我们在选取最优超参时的噪声。  
- 一旦最优超参找到了，我们就会固定下超参的值，然后再真正的测试集上进行一轮评估。  
- Nearest Neighbor 在CIFAR-10上有40%的准确度。虽然实现很容易，但是需要我们存储下整个数据集，并且进行一个样例的评估也会相当消耗时间。
- 最后，我们发现仅仅对行向量使用L1或L2距离是远远无法分析出两张图片的相似度的，因为距离比起图片的真是语义，更容易收到图片的背景和颜色分布的影响。

在下一节中，我们会着重于解决这些问题，并最终把准确率提到90%，这样我们一旦学习完毕就可以舍弃掉训练集了，同时可以在一毫秒内完成一张图片的分类。

### Summary：Applying kNN in practice

如果你想要把kNN运用于实际（最好别用到图像上，若是仅仅作为练手还可以接受）那就按照下述流程即可：

1. 数据预处理：对数据中的特征进行**归一化(normalize)**（e.g. 图像中的每一个像素），让其拥有**零平均值(zero mean)**和**单位标准差(unit variance)**。在之后的章节中，会具体讲解这些细节，本小结不会细讲。因为图像中的像素都是同质的，并且不会表现出较大的差异分布，因此也就不需要数据归一化操作。  
2. 如果你的数据是高维数据，考虑使用降维方法，比如主成分分析PCA（[wiki ref](https://en.wikipedia.org/wiki/Principal_component_analysis), [CS229ref](http://cs229.stanford.edu/notes/cs229-notes10.pdf), [blog ref](https://web.archive.org/web/20150503165118/http://www.bigdataexaminer.com:80/understanding-dimensionality-reduction-principal-component-analysis-and-singular-value-decomposition/)），NCA（[wiki ref](https://en.wikipedia.org/wiki/Neighbourhood_components_analysis), [blog ref](https://kevinzakka.github.io/2020/02/10/nca/)），或者甚至随机投影([Random Projections](https://scikit-learn.org/stable/modules/random_projection.html))  
3. 将数据随机地分成训练集和验证集。按照经验，70%-90%数据作为训练集。这个比例取决于算法中超参的个数，以及超参对于算法的预期影响。如果有非常多的超参要确定，你应该选择更大的验证集来有效确定下它们。如果你的计算资源足够丰富，用检查检验方法总是更加安全的。（均分的份数越多，效果越好，但是计算开销越大）  
4. 在验证集上（如果是交叉检验就是所有数据）进行超参调优，尝试足够多的k值（越多越好），尝试不同范数的距离（L1/L2是不错的选择）。  
5. 如果你的kNN分类器运行时间过长，考虑使用Approximate Nearest Neighbor库（e.g. [FLANN](https://github.com/flann-lib/flann)）来加速检索过程（当然这会以牺牲掉一些准确性为代价）。  
6. 记录下最优的超参取值。有一个问题是是否应该用最优超参再在整个训练集上训练一遍呢？因为如果把验证集放回到训练集后，超参达到的效果可能会改变（因为训练数据变大了）。在实践中，不会再使用到验证集的数据，就假象他在估计出超参后消失不见了即可。就直接用测试集来评估我们最佳模型的性能。报告测试集的准确度，并将结果声明为kNN分类器在数据上的性能。

## Further Reading

Here are some (optional) links you may find interesting for further reading:

- [A Few Useful Things to Know about Machine Learning](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf), where especially section 6 is related but the whole paper is a warmly recommended reading.  
- [Recognizing and Learning Object Categories](https://people.csail.mit.edu/torralba/shortCourseRLOC/index.html), a short course of object categorization at ICCV 2005.