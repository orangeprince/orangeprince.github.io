---
layout: page
title: 富贵闲人
---
{% include JB/setup %}

<ul class="posts">
{% for post in site.posts %}
  <li><span>{{ post.date | date_to_string }}</span> &raquo; <a href="{{ post.url }}">{{ post.title }}</a></li>
{% endfor %}
</ul>

      <section style="width:250px;">
        <h3>Recently Visitors</h3>
          <ul class="ds-recent-visitors" data-num-items="4" data-avatar-size="45" style="margin-top:10px;"></ul>
      </section>