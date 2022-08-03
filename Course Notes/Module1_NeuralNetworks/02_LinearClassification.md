> 原课程网址：https://cs231n.github.io/linear-classify/
> 翻译：Colopen  
> 校对：Colopen

# Linear Classification

在上一节中，我们介绍了图像分类问题，即从一组固定类别中为图像分配单个标签的任务。此外，我们还介绍了k-Nearest Neighbor(kNN)分类器，该分类器通过将图像与训练集中的图像进行比较来进行分类。正如我们所见，kNN有着相当多的缺陷：

1. 分类器必须记住并存储所有的训练数据，以便未来与测试数据进行比较。空间效率十分低下，因为数据集的大小可能很容易打到千兆字节。  
2. 对一张测试图片进行分类的开销十分大，因为它需要与整个训练集中的图片一一进行比较。



## Parameterized mapping from images to label scores
## Interpreting a linear classifier
## Loss function
### Multiclass Support Vector Machine loss
## Practical Considerations
## Softmax classifier
## SVM vs. Softmax
## Interactive web demo
## Summary
## Further Reading