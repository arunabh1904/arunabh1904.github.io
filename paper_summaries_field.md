---
layout: page
title: Paper Summaries by Field
---

{% assign paper_posts = site.categories["Paper Shorts"] %}
{% if paper_posts %}
  {% assign paper_posts = paper_posts | sort: "date" %}
  {% assign posts_by_field = paper_posts | group_by: "field" %}
  {% for field in posts_by_field %}
  <h2>{{ field.name }}</h2>
  <ul>
    {% for post in field.items %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
      <small>— {{ post.date | date: "%b %d, %Y" }}</small>
    </li>
    {% endfor %}
  </ul>
  {% endfor %}
{% endif %}

