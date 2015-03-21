---
layout: post-no-feature
title: "梯度下降、牛顿法与逻辑斯蒂回归"
category: tech
tags: [MachineLearning]
---

逻辑斯蒂回归是在机器学习领域应用非常广泛的算法，而其优化问题与求解方法也非常具有代表性。这边文章简单对这一类优化问题的求解进行一个总结。

一般来说，我们可以把逻辑回归的对数似然函数写成如下形式：

$$
\large
\begin{aligned}
\ell(\theta) = \sum_{i=1}^m \left( y_i\, log \, h(x_i) + (1-y_i)\,log\,(1-h(x_i)) \right)
\end{aligned}
$$

其中

$$
\large
\begin{aligned}
h(x) = \frac{1}{1+e^{-\theta^Tx}}
\end{aligned}
$$

为了求解参数$\theta$，我们需要找到合适的$\theta$以最大化对数似然函数$\ell(\theta)$。换句话说，也就是需要找到$\theta$以使得$\ell(\theta)$能够取到最大值。到了这里，很多教程就开始介绍各种优化算法了，不过我觉得有必要先思考一些更加基本的问题。

###理论基础
1. 上面的目标函数是否一定能取到最大值？

并不是所有的目标函数都能取到最大值，比如函数$f(x)=x$就不行，如果是这样，那我们对参数的求解就没有意义了。但是上面的目标函数显然是可以的，我们可以做一个简单地数学推导