权重的“质量”通常以损失函数来衡量，损失函数值越小，表示权重越理想

SVM作为一个分类函数 ，简单考虑，其损失可以错误分类的次数来衡量

但是再考虑一步，同样对于两个错误分类的cases，一个是刚过了决策边界，令一个则超过了很多，显然这两种错误的程度是不同的

因此 ，基于Score的损失函数更靠谱

SVM Loss，也称Hinge Loss，因为其图像长得类似于一个合叶

SVM被称为大间隔分类，它“希望”有个安全边际

对于第 $i$ 个样本的合叶损失 $L_i$ 是所有非标签类( $j\ne y_i$ , $y_i$ 是标签)的Loss之和：

$$
\begin{aligned}
L_i
&= \sum_{j\ne y_i} \max(0, s_j - s_{y_i} + \Delta) \\
&= \sum_{j\ne y_i} \max(0, x_i w_j^T - x_i w_{y_i}^T + \Delta)
\end{aligned}
$$

，其中

- $i$ ：样本编号，Iterate over N samples  
- $j$ : 类别编号，Iterate over C classes  
- $L_i$ ：第 $i$ 个样本的损失，是在C个分类上的loss之和  
- $w_j^T$ : 计算第 $j$ 个分类时对应的权重  
- $y_i$ : $x_i$ 的正确分类的标签  
- $\Delta$ ：Margin  

直觉上理解：SVM希望样本 $x_i$ 正确分类 $y_i$ 的score $x_i w_{y_i}^T$ 比任何其他错误分类的score $x_i w_j^T$ 都要高一个安全边际 $Delta$

**梯度推导：**

$$
\begin{aligned}
\nabla_wL_i
&= [\dfrac{\partial L_i}{\partial w_1}, \dfrac{\partial L_i}{\partial w_2}, \cdots, \dfrac{\partial L_i}{\partial w_C}] \\\\
&= \begin{pmatrix}
\dfrac{\partial L_i}{\partial w_{11}} & \cdots & \dfrac{\partial L_i}{\partial w_{1y_i}} & \cdots & \dfrac{\partial L_i}{\partial w_{1C}} \\\\
\dfrac{\partial L_i}{\partial w_{21}} & \cdots & \dfrac{\partial L_i}{\partial w_{2y_i}} & \cdots & \dfrac{\partial L_i}{\partial w_{2C}} \\\\
\cdots & \cdots & \cdots & \cdots & \cdots \\\\
\dfrac{\partial L_i}{\partial w_{D1}} & \cdots & \dfrac{\partial L_i}{\partial w_{Dy_i}} & \cdots & \dfrac{\partial L_i}{\partial w_{DC}}
\end{pmatrix}
\end{aligned}
$$

对于所有 $v \ne y_i$ 的项 $\dfrac{\partial L_i}{\partial w_{uv}}$ 有

$$
\dfrac{\partial L_i}{\partial w_{uv}} = 
\dfrac{\partial \sum\limits_{j\ne y_i} \max(0, x_i w_j^T - x_i w_{y_i}^T + \Delta)}{\partial w_{uv}}
$$

若 $x_i w_j^T - x_i w_{y_i}^T + \Delta > 0$，则有

$$
\dfrac{\partial L_i}{\partial w_{uv}} =
\dfrac{\partial \sum\limits_{j\ne y_i} \sum\limits_{k=1}^D x_{ik} w_{kj} - x_{ik} w_{k, y_i}}{\partial w_{uv}}
$$

由于 $v \ne y_i$，故

$$
\dfrac{\partial L_i}{\partial w_{uv}} =
\dfrac{\partial \sum\limits_{j\ne y_i} \sum\limits_{k=1}^D x_{ik} w_{kj}}{\partial w_{uv}} =
\dfrac{\partial \sum\limits_{k=1}^D x_{ik} w_{kv}}{\partial w_{uv}} =
\dfrac{\partial x_{iu} w_{uv}}{\partial w_{uv}} = x_{iu}
$$

因此对于所有 $v \ne y_i$ 的项 $\dfrac{\partial L_i}{\partial w_{uv}}$ 有 $\dfrac{\partial L_i}{\partial w_{uv}} = 1[x_i w_v^T - x_i w_{y_i}^T + \Delta > 0]x_{iu}$

同理，对于特殊的 $v=y_i$ 的项 $\dfrac{\partial L_i}{\partial w_{uy_i}}$ 有 $\dfrac{\partial L_i}{\partial w_{uy_i}} = -(\sum\limits_{j\ne y_i}1[x_i w_j^T - x_i w_{y_i}^T + \Delta > 0])x_i$

由此可得：

$$
\dfrac{\partial L_i}{\partial w_{v}^T} = 1[x_i w_v^T - x_i w_{y_i}^T + \Delta > 0]x_{i}
$$

$$
\dfrac{\partial L_i}{\partial w_{y_i}^T} = -(\sum\limits_{j\ne y_i}1[x_i w_j^T - x_i w_{y_i}^T + \Delta > 0])x_i
$$