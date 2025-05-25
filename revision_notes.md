---
layout: content
title: Revision Notes
---

{% assign rev_posts = site.categories["Revision Notes"] %}
{% if rev_posts %}
  {% assign rev_posts = rev_posts | sort: "date" %}
  <ul class="icon-list">
  {% for post in rev_posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
      <small>— {{ post.date | date: "%b %d, %Y" }}</small>
    </li>
  {% endfor %}
  </ul>
{% endif %}
