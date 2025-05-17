---
layout: post
title: Paper Summaries by Year
---

{% assign paper_posts = site.categories["Paper Shorts"] %}
{% if paper_posts %}
  {% assign paper_posts = paper_posts | sort: "date" %}
  {% assign posts_by_year = paper_posts | group_by_exp: "post", "post.date | date: '%Y'" %}
  {% for year in posts_by_year %}
  <h2>{{ year.name }}</h2>
  <ul>
    {% for post in year.items %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
      <small>— {{ post.date | date: "%b %d, %Y" }}</small>
    </li>
    {% endfor %}
  </ul>
  {% endfor %}
{% endif %}

