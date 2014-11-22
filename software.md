---
layout: post
title : Software
header : Software
group: navigation
---

浙江大学计算机学院研究生论文Latex模板 
---


针对浙江大学计算机学院的毕业论文格式制作的Latex模板，适用于硕士和博士毕业论文的撰写（和浙大其他学院模板有区别）。   
使用xelatex编译，在mac和windows下均可正常使用(mac下需安装[中文字体包](http://linux-wiki.cn/wiki/zh-hans/LaTeX%E4%B8%AD%E6%96%87%E6%8E%92%E7%89%88%EF%BC%88%E4%BD%BF%E7%94%A8XeTeX%EF%BC%89 "中文字体包"))
    
本人和其他一些同学的博士学位论文都使用该软件包完成，顺利通过学院答辩，所以请放心使用。软件包参考了一些[latex论文模板](https://code.google.com/p/zjuthesistex/),在此对这些软件包的作者表示感谢。

使用方法：

{% highlight ruby %}
#compile the bib file twice if needed, e.g., 
bibtex zjulib
bibtex zjulib
#complile the tex file
xelatex main.tex
{% endhighlight %}

下载地址：[点击链接下载](assets/files/zjucs_thesis.zip)
    
    
更多的软件会被陆续添加，敬请期待:-)
---
