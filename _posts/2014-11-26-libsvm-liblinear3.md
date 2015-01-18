---
layout: post-no-feature
title: "LIBSVM与LIBLINEAR（三）"
category: tech
tags:  [MachineLearning]
---

##调节参数
LIBSVM和LIBLINEAR工具包都包含很多需要调节的参数，参数的调节既需要足够的耐心，也有着很多的技巧。当然，还需要对参数本身的意义和对模型的影响了如指掌。下面主要讨论一些对模型影响较大的参数

###参数C
参数$C$是在LIBLINEAR和LIBSVM的求解中都要用到的一个参数。前面说到的各种模型，可以写成统一的形式：

$$
\large
\begin{aligned}
\underset{w}{\operatorname{argmin}}  \quad \Omega(\phi(w))  + C \sum_{i=1}^l \ell(y_i, \phi(w)^T\phi(x_i))
\end{aligned}
$$

其中右边的一项是模型的损失项，其大小表明了分类器对样本的拟合程度。而左边的一项，则是人为加上的损失，与训练样本无关，被称作正则化项(Regularizer)，反映了对训练模型额外增加的一些约束。而参数$C$则负责调整两者之间的权重。$C$越大，则要求模型能够更好地拟合训练样本数据，反之，则要求模型更多的满足正则化项的约束。以LIBLINEAR为例，下面先讨论LIBLINEAR下$\ell-2$norm的情况：

$$
\large
\begin{aligned}
\underset{w}{\operatorname{argmin}}  \quad \parallel w \parallel_2^2  + C \sum_{i=1}^l \ell(y_i, w^Tx_i)
\end{aligned}
$$

之所以要增加正则化项，是因为在设计模型的时候，我们对于样本的质量以及模型的泛化能力没有充分的自信，认为在没有其他约束的情况下，训练得到的模型会因为过于迁就已有的样本数据而无法对新的数据达到同样的效果。在这个时候，就必须在模型中增加人类的一些经验知识。比如上面对$\phi(w)$增加$\ell_2$norm的约束就是如此。如果上面公式中的损失函数对应一个回归问题，那么这个问题就被称作Ridge Regression，中文叫做脊回归或者岭回归。

我们可以站在不同的角度来理解$\ell_2$norm正则化项的意义。如果把学习分类函数中$w$看作是一个参数估计的问题，那么不带正则化项的目标函数对应的就是对$w$进行最大似然估计的问题。为了使$w$的估计更加接近真实的情况，我们可以根据经验对$w$制定一个先验分布。当我们假设$w$先验分布是一个多元高斯分布，且不同维度之间是没有关联的(即协方差矩阵非对角线元素为$0$)，而每一个维度特征的方差为某一固定制，那么推导出来的最大后验概率就是上面的带正则化项的目标函数。而$C$与$w$先验分布的方差相关。$C$越大，就意味着正则化的效果偏弱，$w$的波动范围可以更大，先验的方差也更大；而$C$越小，则意味着正则化的效果更强，$w$的波动范围变小，先验的方差也变小。通过减小$C$的值，可以控制$w$的波动不至于过大，以免受一些数据的影响，造成模型的过拟合（overfitting）。　	 
另外也有一种更直观的解释，上面regularized形式的目标函数也可以根据KKT条件转为constraint形式，也就是：

$$
\large
\begin{aligned}
\underset{w}{\operatorname{argmin}} \quad &  \sum_{i=1}^l \ell(y_i, w^Tx_i) \\
s.t. \quad & \parallel w \parallel_2^2  < r^2
\end{aligned}
$$

通过参数$s$限制$w$的大小，而$r$与$C$也存在着一定正向相关的关系。因此，当$C$较小时，$w$的取值也被限制在的一个很小的范围内。
![L2Norm]{/images/11/l2.svg)
