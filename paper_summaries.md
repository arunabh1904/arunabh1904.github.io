---
layout: page
title: Paper Summaries
---

<details>
  <summary><h2>Organize by Year</h2></summary>
  {% assign paper_posts = site.categories["Paper Shorts"] %}
  {% if paper_posts %}
    {% assign paper_posts = paper_posts | sort: "date" %}
    {% assign posts_by_year = paper_posts | group_by_exp: "post", "post.date | date: '%Y'" %}
    {% for year in posts_by_year %}
      <h3>{{ year.name }}</h3>
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
</details>

<details>
  <summary><h2>Organize by Field</h2></summary>
  {% assign paper_posts = site.categories["Paper Shorts"] %}
  {% if paper_posts %}
    {% assign paper_posts = paper_posts | sort: "date" %}
    {% assign posts_by_field = paper_posts | group_by: "field" %}
    {% for field in posts_by_field %}
      <h3>{{ field.name }}</h3>
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
</details>

