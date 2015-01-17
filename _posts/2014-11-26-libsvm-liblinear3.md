---
layout: post-no-feature
title: "LIBSVM与LIBLINEAR（三）"
category: tech
tags:  [MachineLearning]
---

##调节参数
LIBSVM和LIBLINEAR工具包都包含很多需要调节的参数，参数的调节既需要足够的耐心，也有着很多的技巧。当然，还需要对参数本身的意义和对模型的影响了如指掌。下面主要讨论一些对模型影响较大的参数

###参数$C$
参数$C$是在LIBLINEAR和LIBSVM的求解中都要用到的一个参数。