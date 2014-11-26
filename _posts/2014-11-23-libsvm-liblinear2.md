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
subject\,to \quad & y_i(w ^T x_i - b)  \geq 1 - \xi_i, \\\
& \xi_i \leq 0, i = 1, \ldots, l
\end{aligned}
$$

上面这个目标函数里面，是一个标准的二次凸优化问题，可以比较方便的对每一个变量进行求导，所以是有很多快速的优化算法的。这些算法在LIBLINEAR中都有应用。但是如果是引入kernel的SVM，情况就大不一样了。当引入核函数$\phi$之后，我们可以把上面目标函数中所有的$x_i$都换成$\phi(x_i)$。 如果我们能够知道$\phi$的具体形式，还是可以预先计算出$\phi(x_i)$，然后还按照线性模型的思路进行求解。可实际上很多核函数并无法得到函数的具体形式，也同样无法得到在和空间的特征表达，这个时候，之前的求解思路就完全不work了。所以这里就必须采用标准的SVM求解思路，首先把原问题转化为对偶问题，得到下面的目标函数（具体过程可以参考任何介绍SVM的资料）：	

$$
\begin{aligned}
\underset{\mathbf{\alpha}}{\operatorname{argmin}} \quad & f(\mathbf{\alpha}) =
\frac{1}{2} \mathbf{\alpha}^T Q \mathbf{\alpha} - e^T \mathbf{\alpha} \\\
subject\,to \quad & 0 \le \alpha_i \le C, i= 1,\ldots,l, \\\
& \mathbf{y}^T \mathbf{\alpha}= 0
\end{aligned}
$$

 
