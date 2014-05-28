---
layout: page
title: Hello World!
tagline: Supporting tagline
---
{% include JB/setup %}

Read [Jekyll Quick Start](http://jekyllbootstrap.com/usage/jekyll-quick-start.html)

{% for post in paginator.posts %}
    <a href="{{ post.url }}">{{ post.title }}</a>
{% endfor %}

We need to clean up the themes, make theme usage guides with theme-specific markup examples.
