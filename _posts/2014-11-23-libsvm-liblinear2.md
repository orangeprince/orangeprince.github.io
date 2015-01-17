---
layout: post-no-feature
category: tech
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
\large
\begin{aligned}
\underset{w, b, \xi}{\operatorname{argmin}}  \quad &\frac{1}{2}  w^Tw  + C \sum_{i=1}^l \xi_i \\\
subject\,to \quad & y_i(w ^T \phi(x_i)- b)  \geq 1 - \xi_i, \\\
& \xi_i \leq 0, i = 1, \ldots, l
\end{aligned}
$$

当模型使用linear kernel，也就是$\phi(x) = x$时，上面的问题一个标准的二次凸优化问题，可以比较方便的对每一个变量进行求导。求解这样的问题是有很多快速的优化方法的，这些方法在LIBLINEAR中都有应用。但是如果是引入kernel的SVM，情况就大不一样了。因为很多时候我们既不能得到核函数的具体形式，又无法得到特征在核空间中新的表达。这个时候，之前用在线性SVM上的的求解思路就完全不work了。为了解决这个问题，就必须采用标准的SVM求解思路，首先把原问题转化为对偶问题，得到下面的目标函数（具体过程可以参考任何介绍SVM的资料）：

$$
\large
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

上面的每个问题，处理起来都不简单。作为使用者，或许也没有必要深谙里面的所有细节。我觉得最需要认识的两个问题是：1) SVM的目标函数看起来好像是一个标准的优化问题，但实际求解却要复杂得多。为了提高求解的速度，既要做算法上的优化，也需要做工程上的改进。如果只是简简单单按照教科书的方法，甚至直接调用一些优化的工具包来实现的SVM算法，最多也就算个demo。要能够真正写一个高效稳定、能处理大规模数据的SVM工具还是非常不容易的。所以用LIBSVM还是比自己实现算法要简单靠谱不少。2)SVM的求解之所以要优化，就是因为这个问题本身计算和存储比较麻烦。所以虽然做了这么多的优化，整个算法求解的效率仍然较低。所以我们在使用时还要注意各种程序的细节，提高运行的效率。另外，样本量过大时，有时候为了充分利用数据，也不得不忍痛割爱，放弃kernel的使用。	

除了标准的$C$-SVM，LIBSVM也提供了对其他一些SVM方法的支持。其中$\nu$-SVM与$C$-SVM的算法与应用场景基本是相同的，唯一的区别是原本的参数$C$变成了参数$\nu$。$C$-SVM中参数$C$调整范围在$[0,+\infty)$，而$\nu$-SVM中与之对应的参数$\nu$的调整范围变成了 $(0,1]$。这样的设置使得$\nu$-SVM更具解释性，有时在参数设置上也能提供一定的方便。但$\nu$-SVM与$C$-SVM并不存在本质上的差别，通过参数的调节，两者可以达到完全相同的效果。所以在使用LIBSVM处理分类问题是，选择上面任何一种方法都是OK的，只需要遵循自己的习惯就好了。

One-Class SVM也是LIBSVM所支持的一种分类方法。顾名思义，使用One Class时，只需要提供一类样本，算法会学习一个尽量小的超球面包裹所有的训练样本。One-Class SVM看起来很有诱惑力，因为我们经常会遇到有一类样本而需要学习分类器的情况。但事实上，一方面很多时候我们得到的正样本在采样过程中存在很大的偏差，导致学习出的One Class分类器不一定考虑到了所有正样本的情形；另一方面，大部分问题还是存在很多构造人工负样本的办法。根据我的经验，采用普通的SVM效果通常还是会好过One-Class SVM，而One-Class SVM在真实场景中的使用也并算不上多。因此在使用这个方法前也需要对问题进行更深入的研究。

最后，LIBSVM也支持基于SVM的回归模型，即SVR。与分类模型类似，SVR也分为$C$-SVR和$\nu$-SVR。SVR的目标函数与SVM的分类模型稍有区别。由于回归问题预测值与目标值的偏差可大可小，因此SVR使用了两个slack variable用来刻画预测的误差边界。虽然存在这样的差别，但是两者的基本思路和优化算法与还是基本一致的。

在LIBSVM的实现中，上面五种模型，即$C$-SVM，$\nu$-SVM，One-class SVM，$C$-SVR，$\nu$-SVR，最终都可以转化为一个更通用的优化框架，然后用同样的策略进行求解，这也是LIBSVM所实现的主要功能。在实际使用中，最常用到的方法还是$C$-SVM，这是最传统的SVM分类模型。

###LIBLINEAR

LIBLINEAR是在LIBSVM流行多年后才开发的，要解决的问题本质上也比LIBSVM更简单，其优势主要在于效率与scalablility。之所以存在这样的优势，是因为线性SVM的求解要比kernel SVM简单许多。

还从上面的对偶问题说起，之前SVM的求解很大程度上受到$ \mathbf{y}^T \mathbf{\alpha}= 0$的困扰，因此每次必须选择一组 $\alpha$进行优化。如果对这一约束项追根述源，可以发现这一项是通过令模型的常数项$b$导数为$0$而得到的。而在线性模型中，我们可以通过一个简单地trick，令$x = [x, 1]$和$w = [w, b]$，这样，在模型中的常数项就不存在了。当然，这样的trick只能在线性模型中才适用。没有了上面的约束，优化的目标函数变成了：

$$
\large
\begin{aligned}
\underset{\mathbf{\alpha}}{\operatorname{argmin}} \quad & f(\mathbf{\alpha}) =
\frac{1}{2} \mathbf{\alpha}^T Q \mathbf{\alpha} - e^T \mathbf{\alpha} \\\
subject\,to \quad & 0 \le \alpha_i \le C, i= 1,\ldots,l
\end{aligned}
$$

这个时候，就可以每次只选择一个$\alpha_i$进行优化，每一轮遍历$\alpha$的所有维度，多轮迭代，直至最后收敛。这样的优化算法叫做coordinate descent（坐标下降法）。利用线性函数的特殊性，直接根据$\alpha$就可以计算出$w$的向量表示，于是大大提高了算法的效率。具体的优化算法可以参考文献 [A Dual Coordinate Descent Method for Large-scale Linear SVM](www.csie.ntu.edu.tw/~cjlin/papers/cddual.pdf)。

换一个看问题的角度，线性SVM的目标函数可以写成下面的形式：
	
$$
\large
\begin{aligned}
\underset{w}{\operatorname{argmin}}  \quad \frac{1}{2}  w^Tw  + C \sum_{i=1}^l (max(0, 1-y_iw^Tx_i)) 
\end{aligned}
$$

进一步对问题进行抽象，可以把一类分类问题写成下面的形式：

$$
\large
\begin{aligned}
\underset{w}{\operatorname{argmin}}  \quad  \Omega(w)  + C \sum_{i=1}^l \ell(y_i, w^Tx_i)
\end{aligned}
$$

其中的$\ell$作为误差函数，用来度量预测值与目标值的损失。在上面的线性SVM的情形中，有

$$
\large
\begin{aligned}
\ell(y_i, w^Tx_i) = max(0, 1-y_iw^Tx_i)
\end{aligned}
$$

这里的$\ell$称为Hinge Loss。

又如在Logistic Regression中，loss function $\ell$被定义为
$$
\large
\begin{aligned}
\ell(y_i, w^Tx_i) = log(1+e^{-y_iw_i^Tx_i})
\end{aligned}
$$
$\Omega$一般被称为正则化项(Regularizer)，最常使用的就是前面出现的$\ell_2$-norm，写作$w^Tw$，也可以写作$\parallel w \parallel_2^2$，即向量$w$中所有元素的平方和。除$\ell_2$-norm之外，$\ell_1$-norm也是经常使用regularizer，而且会带来一些特别的效果（后面会进行讨论）。大量的监督学习模型都可以写成loss function + regularizer的形式，而参数C则控制了两者在最终损失函数中所占的比重。不同loss function与regularizer的选取以及两者之间的平衡，是机器学习的最重要主题之一。
	
对于上面的问题，有很多成熟的算法可以进行模型的求解，比如最速梯度法，牛顿法等，对于样本量较大时，也可以采用随机梯度的方法进行训练。	一般来说，由于考虑了二阶导数，牛顿法本身的优化效率要高于只考虑一阶导数的最速梯度法。但由于牛顿法本身在计算量和收敛性上存在很多局限性，所以很少直接使用，而是在牛顿法思想基础上进行一定的改进。其中普遍使用的算法有BFGS和L-BFGS等。具体到liblinear软件包，作者采用的是Trust Region Newton (TRON) method对模型对传统牛顿法进行了改进，该方法被证明比L-BFGS训练更加高效。

LIB LINEAR中实现了基于TRON方法的L-2 SVM和Logistical Regression模型训练。其中的L2-loss SVM是标准SVM的变种，loss function变成了：
$$
\large
\begin{aligned}
\ell(y_i, w^Tx_i) = \left( max(0, 1-y_iw^Tx_i) \right) 	^2
\end{aligned}
$$

从实际效果来说，L2-loss SVM与标准的L1-loss SVM并没有太大的区别。但是在计算上，前者的求导形式更加简单，便于梯度的计算与优化。LIBLINEAR并没有实现Trust Region Newton法的标准L1-loss SVM实现，一方面是因为直接对hinge loss求导需要分段讨论比较复杂，另一方面L2-loss SVM基本可以直接替代L1-loss SVM。不过在其他的一些软件包中，如[SVMLIN](http://vikas.sindhwani.org/svmlin.html)中，则实现了L1-loss SVM的原问题求解，但使用的优化算法是L-BGFS而不是TRON。

### 总结
前面介绍了LIBSVM和LIBLINEAR的优化算法，下面简单总结一下不同算法的应用场景吧：

* 所有线性问题都是用LIBLINEAR，而不要使用LIBSVM。
* LIBSVM中的不同算法，如C-SVM和$nu$-SVM在模型和求解上并没有本质的区别，只是做了一个参数的变换，所以选择自己习惯的就好。
* LIBLINEAR的优化算法主要分为两大类，即求解原问题(primal problem)和对偶问题(dual problem)。求解原问题使用的是TRON的优化算法，对偶问题使用的是Coordinate Descent优化算法。总的来说，两个算法的优化效率都较高，但还是有各自更加擅长的场景。对于样本量不大，但是维度特别高的场景，如文本分类，更适合对偶问题求解，因为由于样本量小，计算出来的Kernel Matrix也不大，后面的优化也比较方便。而如果求解原问题，则求导的过程中要频繁对高维的特征矩阵进行计算，如果特征比较稀疏的话，那么就会多做很多无意义的计算，影响优化的效率。相反，当样本数非常多，而特征维度不高时，如果采用求解对偶问题，则由于Kernel Matrix过大，求解并不方便。反倒是求解原问题更加容易。

##多分类问题

LIBSVM和LIBLINEAR都支持多分类（Multi-class classification）问题。所谓多分类问题，就是说每一个样本的类别标签可以超过2个，但是最终预测的结果只能是一个类别。比如经典的手写数字识别问题，输入是一幅图像，最后输出的是0-9这十个数字中的某一个。

LIBSVM与LIBLINEAR但实现方式却完全不同。LIBSVM采取的one vs one的策略，也就是所有的分类两两之间都要训练一个分类器。这样一来，如果存在$k$个class，理论上就需要训练 $k(k-1)/2$个分类器。实际上，libsvm在这一步也进行了一定的优化，利用已有分类的关系，减少分类器的个数。尽管如此，LIBSVM在多分类问题上还是要多次训练分类器。但是，考虑到前面说的LIBSVM的优化方法，随着样本数量的增加，训练的复杂度会非线性的增加。而通过1VS1的策略，可以保证每一个子分类问题的样本量不至于太多，其实反倒是方便了整个模型的训练。

而LIBLINEAR则采取了另一种训练策略，即one vs all。每一个class对应一个分类器，副样本就是其他类别的所有样本。由于LIBLINEAR能够和需要处理的训练规模比LIBSVM大得多，因此这种方式要比one vs one更加高效。此外，LIBLINEAR还实现了基于Crammer and Singer方法的SVM多分类算法，在一个统一的目标函数中学习每一个class对应的分类器。

##输出文件
一般来说，我们使用LIBLINEAR或者LIBSVM，可以直接调用系统的训练与预测函数，不需要直接去接触训练得到的模型文件。但有时候我们也可能需要在自己的平台实现预测的算法，这时候就不可避免的要对模型文件进行解析。

由于LIBLINEAR与LIBSVM的训练模型不同，因此他们对应的模型文件格式也不同。LIBLINEAR训练结果的格式相对简单，例如：
{% highlight bash %}
solver_type L2R_L2LOSS_SVC_DUAL
nr_class 3
label 1 2 3
nr_feature 5
bias -1
w
-0.4021097293855418 0.1002472498884907 -0.1619908595357437
0.008699468444669581 0.2310109611908343 -0.2295723940247394
-0.6814324057724231 0.4263611607497726 -0.4190714505083906
-0.1505088594898125 0.2709227166451816 -0.1929294695905781
2.14656708009991 -0.007495770268046003 -0.1880325536062815
{% endhighlight%}
上面的<code>solver_type</code>表示求解类型

	
