---
layout: post-no-feature
tags:  [MachineLearning]
title: LIBSVM与LIBLINEAR（二）
---

##模型选择
LIBSVM和LIBLINEAR都提供了多种不同的模型供使用者选择，不同的模型有各自适用的场景。下面分别介绍LIBSVM和LIBLINEAR所提供的各种模型。
###libsvm
下面是LIBSVM帮助内容提供的介绍，给出了LIBSVM支持的5种模型。其中模型0和1对应的都是SVM的分类模型，2对应的是one-class分类器，也就是只需要标注一个标签，模型3和4对应的是SVM的回归模型。
{% highlight bash %}
-s svm_type : set type of SVM (default 0)
	0 -- C-SVC		(multi-class classification)
	1 -- nu-SVC		(multi-class classification)
	2 -- one-class SVM
	3 -- epsilon-SVR	(regression)
	4 -- nu-SVR		(regression)
{% endhighlight %}
首先来看最基础的C-SVC模型。SVM可以写成如下的优化目标函数（这里不详细介绍推导算法了）：

$$
\begin{aligned}
\underset{w, b, \xi}{\operatorname{argmin}}  \quad &\frac{1}{1}  w^Tw  + C \sum_{i=1}^l \xi_i \\\
subject\,to \quad & y_i(w ^T \phi(x_i)- b)  \geq 1 - \xi_i, \\\
& \xi_i \leq 0, i = 1, \ldots, l
\end{aligned}
$$

当模型使用linear kernel，也就是$\phi(x) = x$时，上面的问题一个标准的二次凸优化问题，可以比较方便的对每一个变量进行求导。求解这样的问题是有很多快速的优化方法的，这些方法在LIBLINEAR中都有应用。但是如果是引入kernel的SVM，情况就大不一样了。因为很多时候我们既不能得到核函数的具体形式，又无法得到特征在核空间中新的表达。这个时候，之前用在线性SVM上的的求解思路就完全不work了。为了解决这个问题，就必须采用标准的SVM求解思路，首先把原问题转化为对偶问题，得到下面的目标函数（具体过程可以参考任何介绍SVM的资料）：	

$$
\begin{aligned}
\underset{\mathbf{\alpha}}{\operatorname{argmin}} \quad & f(\mathbf{\alpha}) =
\frac{1}{2} \mathbf{\alpha}^T Q \mathbf{\alpha} - e^T \mathbf{\alpha} \\\
subject\,to \quad & 0 \le \alpha_i \le C, i= 1,\ldots,l, \\\
& \mathbf{y}^T \mathbf{\alpha}= 0
\end{aligned}
$$

通过对偶变化，上面的目标函数变成了一个关于变量$\alpha$的二次型。很显然，上面目标函数中最重要的常亮是矩阵$Q$，既训练样本的Kernel Matrix，满足$Q_{i.j}=\phi(x_i)^T\phi(x_j)$。先看好的一方面，根据核函数的定义，能够保证$Q$是一个正定的矩阵。也就是说，上面的目标函数还是一个凸函数，优化收敛后能保证得到的解是全局最优解， 这也是SVM的重要优势之一。但是问题也随之而来，使用常用的核函数，只要任意给出两个向量，总是能够计算出一个非0的距离。这也就意味着矩阵$Q$将会是一个非常稠密的矩阵，如果训练样本足够多，那么矩阵$Q$的存储和计算将成为很大的问题，这也是SVM的优化算法中的最大挑战。

由于矩阵$Q$过大，所以想一次性优化整个$\alpha$是比较困难的。所以常用的方法都是先把$Q$大卸八块，每次选择一部分的$Q$，然后update与这部分$Q$相关的$\alpha$的值。这其中最著名的算法就是1998由John C. Platt提出的[SMO算法](http://research.microsoft.com/pubs/68391/smo-book.pdf)，而LIBSVM的优化过程也是基于SMO算法进行的。SMO算法的每一步迭代都选择最小的优化单元，也就是固定其他的$\alpha$，只挑选两个$\alpha$的值进行优化。之所以不选择一个，是因为有$\mathbf{y}^T \mathbf{\alpha}= 0$的约束，至少选择两个$\alpha$的坐标才有可能进行更新。本文主要目的是介绍LIBSVM，所以就不详细讨论SMO的细节了。至于LIBSVM中的具体算法实现，在[LIBSVM的官方论文](http://140.112.30.28/~cjlin/papers/libsvm.pdf)中介绍的很详细，这里总结一些关键问题：

*	sdfs
* 	sdf  	
