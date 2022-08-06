score function：

$$f(x_i) = \mathbf W x_i$$

$$
f_j = w^T_j x_i
$$

cost function:

$$
L_i
= -\log \bigg(
    \dfrac{e^{f_{y_i}}}{\sum\limits_j  e^{f_j}}
\bigg)
= -f_{y_i} + \log \bigg(\sum\limits_{j} e^{f_j}\bigg)
$$

$$
L = - \sum_{i} \log \bigg(
    \dfrac{e^{f_{y_i}}}{\sum\limits_j  e^{f_j}}
\bigg) =
-f_y + \log \bigg(
\sum_j e^{f_j}
\bigg)
$$

derivative：

when $j \ne y_i$

$$
\begin{aligned}
\dfrac{\partial L_i}{\partial w_j^T} &= 
\dfrac{e^{f_j}}{\sum\limits_j  e^{f_j}} \cdot
\dfrac{\partial (x_iw_j^T)}{\partial w_j^T} =
x_i^T \cdot \dfrac{e^{f_j}}{\sum\limits_j  e^{f_j}} \\\\
\dfrac{\partial L}{\partial w_j^T} &=
X^T \cdot \dfrac{e^{f_j}}{\sum\limits_j  e^{f_j}}
\end{aligned}
$$

when $j = y_i$

$$
\begin{aligned}
\dfrac{\partial L_i}{\partial w_j^T} &=
-x_i^T  + \dfrac{e^{f_j}}{\sum\limits_j  e^{f_j}} \cdot
x_i^T =
x_i^T \cdot (\dfrac{e^{f_j}}{\sum\limits_j  e^{f_j}} - 1) \\\\
\dfrac{\partial L}{\partial w_j^T} &=
X^T \cdot (\dfrac{e^{f_j}}{\sum\limits_j  e^{f_j}} - 1) \\\\
\end{aligned}
$$
