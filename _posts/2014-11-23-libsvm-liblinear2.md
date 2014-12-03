---
layout: post-no-feature
tags:  [MachineLearning]
title: LIBSVM与LIBLINEAR（二）
---

##模型与优化
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
\underset{w, b, \xi}{\operatorname{argmin}}  \quad &\frac{1}{2}  w^Tw  + C \sum_{i=1}^l \xi_i \\\
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

由于矩阵$Q$过大，所以想一次性优化整个$\alpha$是比较困难的。所以常用的方法都是先把$Q$大卸八块，每次选择一部分的$Q$，然后update与这部分$Q$相关的$\alpha$的值。这其中最著名的算法就是1998由John C. Platt提出的[SMO算法](http://research.microsoft.com/pubs/68391/smo-book.pdf)，而LIBSVM的优化过程也是基于SMO算法进行的。SMO算法的每一步迭代都选择最小的优化单元，也就是固定其他的$\alpha$，只挑选两个$\alpha$的值进行优化。之所以不选择一个，是因为有$\mathbf{y}^T \mathbf{\alpha}= 0$的约束，至少选择两个$\alpha$的坐标才有可能进行更新。本文主要目的是介绍LIBSVM，所以就不详细讨论SMO的细节了。至于LIBSVM中的具体算法实现，在[LIBSVM的官方论文](http://140.112.30.28/~cjlin/papers/libsvm.pdf)中介绍的很详细，这里总结部分关键问题：

*	Working Set，也就是需要优化的$\alpha$部分的选取
* 	迭代停止条件的设置
* 	$\alpha$的更新算法，也就是每一步子问题的求解方法
* 	Shrinking，即移除一些已经满足条件的$\alpha$，加快收敛速度
* 	Cache，当$Q$矩阵过大时，需要对矩阵进行缓存。

上面的每个问题，处理起来都不简单。作为使用者，或许也没有必要深谙里面的所有细节。我觉得最需要认识的两个问题是：1) SVM的目标函数看起来好像是一个标准的优化问题，但实际求解却要复杂得多。为了提高求解的速度，既要做算法上的优化，也需要做工程上的改进。所以如果是我们简简单单按照教科书的方法，甚至直接调用一些优化的工具包来实现的SVM算法，只能起到demo的作用。要能够真正写一个高效稳定、能处理大规模数据的SVM工具还是非常不容易的。所以用LIBSVM还是比自己实现算法要简单靠谱不少。2)SVM的求解之所以要优化，就是因为这个问题本身计算和存储比较麻烦。所以虽然做了这么多的优化，整个算法求解的效率仍然较低。所以我们在使用时还要注意各种程序的细节，提高运行的效率。另外，样本量过大时，有时候为了充分利用数据，也不得不忍痛割爱，放弃kernel的使用。	

除了标准的$C$-SVM，LIBSVM也提供了对其他一些SVM方法的支持。其中$\nu$-SVM与$C$-SVM的算法与应用场景基本是相同的，唯一的区别是原本的参数$C$变成了参数$\nu$。$C$-SVM中参数$C$调整范围在$[0,+\infty)$，而$\nu$-SVM中与之对应的参数$\nu$的调整范围变成了 $(0,1]$。这样的设置使得$\nu$-SVM更具解释性，有时在参数设置上也能提供一定的方便。但$\nu$-SVM与$C$-SVM并不存在本质上的差别，通过参数的调节，两者可以达到完全相同的效果。所以在使用LIBSVM处理分类问题是，选择上面任何一种方法都是OK的，只需要遵循自己的习惯就好了。

One-Class SVM也是LIBSVM所支持的一种分类方法。顾名思义，使用One Class时，只需要提供一类样本，算法会学习一个尽量小的超球面包裹所有的训练样本。One-Class SVM看起来很有诱惑力，因为我们经常会遇到有一类样本而需要学习分类器的情况。但事实上，一方面很多时候我们得到的正样本在采样过程中存在很大的偏差，导致学习出的One Class分类器不一定考虑到了所有正样本的情形；另一方面，大部分问题还是存在很多构造人工负样本的办法。根据我的经验，采用普通的SVM效果通常还是会好过One-Class SVM，而One-Class SVM在真实场景中的使用也并算不上多。因此在使用这个方法前也需要对问题进行更深入的研究。

最后，LIBSVM也支持基于SVM的回归模型，即SVR。与分类模型类似，SVR也分为$C$-SVR和$\nu$-SVR。SVR的目标函数与SVM的分类模型稍有区别。由于回归问题预测值与目标值的偏差可大可小，因此SVR使用了两个slack variable用来刻画预测的误差边界。虽然存在这样的差别，但是两者的基本思路和优化算法与还是基本一致的。

在LIBSVM的实现中，上面五种模型，即$C$-SVM，$\nu$-SVM，One-class SVM，$C$-SVR，$\nu$-SVR，最终都可以转化为一个更通用的优化框架，然后用同样的策略进行求解，这也是LIBSVM所实现的主要功能。在实际使用中，最常用到的方法还是$C$-SVM，这是最传统的SVM分类模型。

###LIBLINEAR

接下来讨论LIBLINEAR的求解方法。LIBLINEAR的出现较LIBSVM晚，而解决的问题反倒更简单，那么LIBLINEAR又有着什么样的过人之处呢？

对线性的SVM，目标函数可以写成如下的形式：
$$
\begin{aligned}
\underset{w}{\operatorname{argmin}}  \quad \frac{1}{2}  w^Tw  + C \sum_{i=1}^l (max(0, 1-y_iw^Tx_i)) 
$$

	