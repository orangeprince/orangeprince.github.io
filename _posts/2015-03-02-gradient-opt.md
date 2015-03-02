---
layout: post-no-feature
title: "梯度下降、牛顿法与逻辑斯蒂回归"
category: tech
tags: [MachineLearning]
---

逻辑斯蒂回归是在机器学习领域应用非常广泛的算法，而其优化问题与求解方法也非常具有代表性。这边文章简单对这一类优化问题的求解进行以下总结。

一般来说，我们可以把逻辑回归的目标函数写成如下形式：

$$
\large
\begin{aligned}
\ell(\theta) = \sum_{i=1}^m \left( y_i log h(x_i) + (1-y_i)log(1-h(x_i)) \right)
\end{aligned}
$$

其中

$$
\large
\begin{aligned}
h(x) = \frac{1}{1+e^{-\theta^Tx}}
\end{aligned}
$$
