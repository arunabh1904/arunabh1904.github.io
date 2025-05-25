---
layout: content
title: Ponderings
---

{% assign ponder_posts = site.categories["Ponderings"] %}
{% if ponder_posts %}
  {% assign ponder_posts = ponder_posts | sort: "date" %}
  <ul class="icon-list">
  {% for post in ponder_posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
      <small>— {{ post.date | date: "%b %d, %Y" }}</small>
    </li>
  {% endfor %}
  </ul>
{% endif %}
