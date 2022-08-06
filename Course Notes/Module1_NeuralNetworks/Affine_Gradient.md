Hypothesis:

$$
Y = XW + b
$$

，其中

$$
X_{N\times D} = \begin{pmatrix}
x_{11} & x_{12} & \cdots & x_{1D} \\\\
x_{21} & x_{22} & \cdots & x_{2D} \\\\
\cdots & \cdots & \cdots & \cdots \\\\
x_{N1} & x_{N2} & \cdots & x_{ND}
\end{pmatrix}, \quad
W_{D\times C} = \begin{pmatrix}
w_{11} & w_{12} & \cdots & w_{1C} \\\\
w_{21} & w_{22} & \cdots & w_{2C} \\\\
\cdots & \cdots & \cdots & \cdots \\\\
w_{D1} & w_{D2} & \cdots & w_{DC}
\end{pmatrix}, \quad
b_{1\times C} = \begin{pmatrix}
b_{1} & b_{2} & \cdots & b_C
\end{pmatrix}
$$

$$
Y_{N \times C} = \begin{pmatrix}
y_{11} & y_{12} & \cdots & y_{1C} \\\\
y_{21} & y_{22} & \cdots & y_{2C} \\\\
\cdots & \cdots & \cdots & \cdots \\\\
y_{N1} & y_{N2} & \cdots & y_{NC}
\end{pmatrix}, \quad
y_{ij} = \sum_{k=1}^D x_{ik}w_{kj} + b_j =
x_{i1}w_{1j} + x_{i2}w_{2j} + \cdots + x_{iD}w_{Dj} + b_j
$$

$$
\dfrac{\partial L}{\partial Y} = \begin{pmatrix}
\dfrac{\partial L}{\partial y_{11}} & \dfrac{\partial L}{\partial y_{12}} & \cdots &
\dfrac{\partial L}{\partial y_{1C}} \\\\
\dfrac{\partial L}{\partial y_{21}} & \dfrac{\partial L}{\partial y_{22}} & \cdots &
\dfrac{\partial L}{\partial y_{2C}} \\\\
\cdots & \cdots & \cdots & \cdots   \\\\
\dfrac{\partial L}{\partial y_{N1}} & \dfrac{\partial L}{\partial y_{N2}} & \cdots &
\dfrac{\partial L}{\partial y_{NC}}
\end{pmatrix}_{N \times C}
$$

derivative with respect to X:

$$
\dfrac{\partial y_{pq}}{\partial x_{ij}} = 1_{p=i} w_{jq} \quad\Rightarrow\quad
\dfrac{\partial L}{\partial x_{ij}} =
\sum_{k}^{N} \dfrac{\partial L}{\partial y_{ik}} \dfrac{\partial y_{ik}}{\partial x_{ij}} =
\sum_{k}^{N} \dfrac{\partial L}{\partial y_{ik}} w_{jk}
$$

$$
\begin{aligned}
\dfrac{\partial L}{\partial X} &=
\begin{pmatrix}
\sum_{k}^{N} \dfrac{\partial L}{\partial y_{1k}} w_{1k} & 
\sum_{k}^{N} \dfrac{\partial L}{\partial y_{1k}} w_{2k} & \cdots &
\sum_{k}^{N} \dfrac{\partial L}{\partial y_{1k}} w_{Dk} \\\\
\sum_{k}^{N} \dfrac{\partial L}{\partial y_{2k}} w_{1k} & 
\sum_{k}^{N} \dfrac{\partial L}{\partial y_{2k}} w_{2k} & \cdots &
\sum_{k}^{N} \dfrac{\partial L}{\partial y_{2k}} w_{Dk} \\\\
\cdots & \cdots & \cdots & \cdots \\\\
\sum_{k}^{N} \dfrac{\partial L}{\partial y_{Nk}} w_{1k} & 
\sum_{k}^{N} \dfrac{\partial L}{\partial y_{Nk}} w_{2k} & \cdots &
\sum_{k}^{N} \dfrac{\partial L}{\partial y_{Nk}} w_{Dk}
\end{pmatrix} \\\\ &=
\begin{pmatrix}
\dfrac{\partial L}{\partial y_{11}} & \dfrac{\partial L}{\partial y_{12}} & \cdots &
\dfrac{\partial L}{\partial y_{1C}} \\\\
\dfrac{\partial L}{\partial y_{21}} & \dfrac{\partial L}{\partial y_{22}} & \cdots &
\dfrac{\partial L}{\partial y_{2C}} \\\\
\cdots & \cdots & \cdots & \cdots   \\\\
\dfrac{\partial L}{\partial y_{N1}} & \dfrac{\partial L}{\partial y_{N2}} & \cdots &
\dfrac{\partial L}{\partial y_{NC}}
\end{pmatrix}
\begin{pmatrix}
w_{11} & w_{21} & \cdots & w_{D1} \\\\
w_{12} & w_{22} & \cdots & w_{D2} \\\\
\cdots & \cdots & \cdots & \cdots \\\\
w_{1C} & w_{2C} & \cdots & w_{DC}
\end{pmatrix} \\\\ &=
\dfrac{\partial L}{\partial Y} W^{\top}
\end{aligned}
$$

derivative with respect to W:

$$
\dfrac{\partial y_{pq}}{\partial w_{ij}} = 1_{q=j} x_{pi} \quad\Rightarrow\quad
\dfrac{\partial L}{\partial w_{ij}} =
\sum_k^C \dfrac{\partial L}{\partial y_{kj}} \dfrac{\partial y_{kj}}{\partial w_{ij}}
$$

$$
\begin{aligned}
\dfrac{\partial L}{\partial W} &=
\begin{pmatrix}
\sum_k^C \dfrac{\partial L}{\partial y_{k1}}x_{k1} & \sum_k^C \dfrac{\partial L}{\partial y_{k2}}x_{k1} & \cdots & \sum_k^C \dfrac{\partial L}{\partial y_{kC}}x_{k1} \\\\
\sum_k^C \dfrac{\partial L}{\partial y_{k1}}x_{k2} & \sum_k^C \dfrac{\partial L}{\partial y_{k2}}x_{k2} & \cdots & \sum_k^C \dfrac{\partial L}{\partial y_{kC}}x_{k2} \\\\
\cdots & \cdots & \cdots & \cdots \\\\
\sum_k^C \dfrac{\partial L}{\partial y_{k1}}x_{kD} & \sum_k^C \dfrac{\partial L}{\partial y_{k2}}x_{kD} & \cdots & \sum_k^C \dfrac{\partial L}{\partial y_{kC}}x_{kD}
\end{pmatrix} \\\\ &=
\begin{pmatrix}
x_{11} & x_{21} & \cdots & x_{N1} \\\\
x_{12} & x_{22} & \cdots & x_{N2} \\\\
\cdots & \cdots & \cdots & \cdots \\\\
x_{1C} & x_{2C} & \cdots & x_{NC}
\end{pmatrix}
\begin{pmatrix}
\dfrac{\partial L}{\partial y_{11}} & \dfrac{\partial L}{\partial y_{12}} & \cdots &
\dfrac{\partial L}{\partial y_{1C}} \\\\
\dfrac{\partial L}{\partial y_{21}} & \dfrac{\partial L}{\partial y_{22}} & \cdots &
\dfrac{\partial L}{\partial y_{2C}} \\\\
\cdots & \cdots & \cdots & \cdots   \\\\
\dfrac{\partial L}{\partial y_{N1}} & \dfrac{\partial L}{\partial y_{N2}} & \cdots &
\dfrac{\partial L}{\partial y_{NC}}
\end{pmatrix} \\\\ &=
X^\top \dfrac{\partial L}{\partial Y}
\end{aligned}
$$

derative with respect to b:

$$
\dfrac{\partial y_{pq}}{\partial b_i} = 1_{q=i} \quad\Rightarrow\quad
\dfrac{\partial L}{\partial b_i} = 
\sum_{k}^{C} \dfrac{\partial L}{\partial y_{ki}} \dfrac{\partial y_{ki}}{b_i} =
\sum_{k}^{C} \dfrac{\partial L}{\partial y_{ki}}
$$

$$
\begin{aligned}
\dfrac{\partial L}{\partial b} &=
\begin{pmatrix}
\sum_{k}^{C} \dfrac{\partial L}{\partial y_{k1}} &
\sum_{k}^{C} \dfrac{\partial L}{\partial y_{k2}} & \cdots &
\sum_{k}^{C} \dfrac{\partial L}{\partial y_{kC}}
\end{pmatrix} \\\\ &=
\begin{pmatrix}
1 & 1 & \cdots & 1
\end{pmatrix}
\begin{pmatrix}
\dfrac{\partial L}{\partial y_{11}} & \dfrac{\partial L}{\partial y_{12}} & \cdots &
\dfrac{\partial L}{\partial y_{1C}} \\\\
\dfrac{\partial L}{\partial y_{21}} & \dfrac{\partial L}{\partial y_{22}} & \cdots &
\dfrac{\partial L}{\partial y_{2C}} \\\\
\cdots & \cdots & \cdots & \cdots   \\\\
\dfrac{\partial L}{\partial y_{N1}} & \dfrac{\partial L}{\partial y_{N2}} & \cdots &
\dfrac{\partial L}{\partial y_{NC}}
\end{pmatrix} \\\\ &=
\begin{pmatrix}
1 & 1 & \cdots & 1
\end{pmatrix} \dfrac{\partial L}{\partial Y} = \text{sum}(\dfrac{\partial L}{\partial Y}, \text{axis=0})
\end{aligned}
$$