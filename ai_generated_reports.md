---
layout: content
title: AI Generated Reports
---

## Audio 🎧 summaries

{% assign ai_posts = site.categories["Machine Learning Deep-Dives"] %}
{% if ai_posts %}
  {% assign ai_posts = ai_posts | sort: "date" %}
  <ul class="icon-list">
  {% for post in ai_posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
      <small>— {{ post.date | date: "%b %d, %Y" }}</small>
    </li>
  {% endfor %}
  </ul>
{% endif %}

## Deep research summaries

{% assign ai_pdfs = site.static_files | where: "extname", ".pdf" | where_exp: "f", "f.path contains '/assets/pdfs/'" %}
{% if ai_pdfs %}
  <ul class="icon-list">
  {% for file in ai_pdfs %}
    {% if file.name == 'VLM Research Summary.pdf' %}
      <li><a href="{{ file.path }}">A survey of VLMs. - May 17, 2025 generated at</a></li>
    {% else %}
      <li><a href="{{ file.path }}">{{ file.name }}</a></li>
    {% endif %}
  {% endfor %}
  </ul>
{% endif %}
