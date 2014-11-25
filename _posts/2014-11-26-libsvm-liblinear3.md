---
layout: post-no-feature
title: "LIBSVM与LIBLINEAR（三）"
category: "机器学习"
tags:  [MachineLearning]
---

##模型选择
{% highlight bash %}
-s svm_type : set type of SVM (default 0)
	0 -- C-SVC		(multi-class classification)
	1 -- nu-SVC		(multi-class classification)
	2 -- one-class SVM
	3 -- epsilon-SVR	(regression)
	4 -- nu-SVR		(regression)
{% endhighlight %}
###libsvm
 