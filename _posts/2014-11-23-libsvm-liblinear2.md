---
layout: post-no-feature
tags:  [MachineLearning]
title: LIBSVM与LIBLINEAR（二）
---

##模型选择
LIBSVM和LIBLINEAR都提供了多种不同的模型供使用者选择，不同的模型有各自适用的场景。下面分别介绍LIBSVM和LIBLINEAR所提供的各种模型。
###libsvm
下面是LIBSVM帮助内容提供的介绍，给出了LIBSVM支持的5种模型。其中模型0和1对应的都是普通的SVM分类器，2对应的是one-class分类器，也就是只需要标注一个标签，模型3和4对应的是基于SVM的回归模型。
{% highlight bash %}
-s svm_type : set type of SVM (default 0)
	0 -- C-SVC		(multi-class classification)
	1 -- nu-SVC		(multi-class classification)
	2 -- one-class SVM
	3 -- epsilon-SVR	(regression)
	4 -- nu-SVR		(regression)
{% endhighlight %}
首先来看最基础的C-SVC模型。线性的SVM可以写成如下的优化目标函数（这里不详细介绍推导算法了）：

$$
\begin{aligned}
\underset{w, b, \ksi}{\operatorname{argmin}}  \quad &\frac{1}{1} \parallel w \parallel ^2 + C \sum_{i=1}^l \xi_i
subject\,to \quad & y_i(w \cdot x_i - b) + \xi_i \leq 1
& \xi_i \leq 0, i = 1, \ldot, m
\end{aligned}
$$

$$
\begin{aligned}
\underset{\mathbf{\alpha}}{\operatorname{argmin}} \quad & f(\mathbf{\alpha}) =
\frac{1}{2} \mathbf{\alpha}^T Q \mathbf{\alpha} - e^T \mathbf{\alpha} \\\
subject\,to \quad & 0 \le \alpha_i \le C, i= 1,\ldots,l, \\\
& \mathbf{y}^T \mathbf{\alpha}= 0
\end{aligned}
$$

 
