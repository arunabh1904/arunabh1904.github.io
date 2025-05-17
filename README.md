# Personal Jekyll Blog

This repository contains the source for a static website built with [Jekyll](https://jekyllrb.com). The site uses the Minima theme with custom CSS for dark/light mode and some JavaScript to add navigation features.

## Repository Overview

Site settings and metadata (author, description, theme and plugins) are configured in `_config.yml`:

```
author: "Arunabh Mishra"
description: "A personal blog on computer vision, robotics, and more."
theme: "minima"
plugins:
  - jekyll-feed
  - jekyll-sitemap
  - jekyll-seo-tag
```

Posts use the default layout `post` and are automatically tagged via the configuration’s `defaults` section. The site excludes build files and paginates posts in sets of five.

## Structure

- `_layouts/` – Contains page templates. `home.html` defines the landing page layout with a full-page background and a pinned "Contact Me" panel at the bottom right. `post.html` defines post pages with light/dark theme toggling and a "Go Home" button implemented in JavaScript.
- `_includes/` – Partial templates. `head.html` loads Font Awesome icons and Highlight.js for code syntax highlighting. `navlinks.html` and `sharelinks.html` provide previous/next navigation and social-sharing buttons respectively.
- `css/override.css` – Custom styles, including variables for dark/light themes, styling for buttons, and a pinned links panel.
- `index.html` – Home page content listing posts from two categories ("My Journey So Far" and "ML Deep research reports").
- `archive.md` – Generates an archive of posts grouped by tag.
- `_posts/` – Blog posts written in Markdown with front matter specifying layout, title, date and categories.

Assets such as images are stored in `assets/images/`.
PDFs can be added to `assets/pdfs/` and will appear in the ML Deep research reports section.

## Getting Started

1. Install Jekyll and dependencies (Ruby, Bundler) if you want to build locally.
2. Run `jekyll serve` from the repository root to start a local server.
3. Posts are created in `_posts/` following the `YYYY-MM-DD-title.md` naming scheme and include front matter (layout, title, date, categories).
4. Modify `_config.yml` to change site settings like social usernames or pagination.
5. Customize layouts or include files to alter page structure or metadata.

## Staging Environment

1. Create a `staging` branch from `main`.
2. Push updates to this branch to run the workflow.
3. Enable GitHub Pages on `staging` to preview changes.

## What to Learn Next

- **Jekyll basics** – Understanding layouts, includes, front matter and Liquid templating will help you modify or expand the site.
- **Custom styling** – Explore how `css/override.css` interacts with the Minima theme to provide dark mode and responsive elements.
- **Writing and organizing posts** – Posts can be grouped by categories or tags; see how `index.html` and `archive.md` loop over categories.
- **GitHub Pages** – If hosting on GitHub, familiarize yourself with its Jekyll integration, which can build the site automatically.

This README adapts the repository overview from a conversation with the repository maintainers to help new contributors quickly understand the project.
