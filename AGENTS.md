# Repository Agent Guidelines

This file collects style guidelines for agents contributing to this repository.
Update it in pull requests as new conventions are adopted.

## Repository overview
- `_posts/` contains the blog posts written in Markdown.
- `_layouts/` stores the page templates used across the site.
- `_includes/` holds partial templates that are shared between layouts.
- `assets/` contains site assets such as images and scripts.
- `css/` houses custom styles that override the Minima theme.
- `index.html` defines the home page layout and lists recent posts.
- `archive.md` generates an archive page grouped by tag.

## Setup
- Install Ruby 3.1 and Bundler.
- Run `bundle install` to install dependencies.
- Use `bundle exec jekyll build` if you have a Gemfile.

## Style
- Use two spaces for indentation in YAML, HTML, and CSS files.
- Keep lines under 120 characters when possible.
- End files with a single newline.
- Write comments as full sentences and end them with a period.

## Local workflow
- Run `jekyll build` to verify the site compiles before committing changes that affect the structure or content.
- Use `jekyll serve` to preview the site locally while developing.

## Git
- Write commit messages in the present tense and keep the summary under 50 characters.
- Use the imperative mood for commit messages.
- Ensure the site builds by running `jekyll build` before committing changes that affect the site structure or content.
- When a pull request introduces a new style convention, add it to this file.

## Pull requests
- Provide a clear summary of your changes and include any build or preview steps you followed.

## PR Instructions
- Title format: `[Fix] Short description`.
- Include a one-line summary and a "Testing Done" section.

## Markdown
- Use `##` for second-level headings.
- Wrap code blocks in triple backticks with an optional language identifier.

## Test workflow
- `.github/workflows/jekyll.yml` runs on pushes to `main` and `staging` and on pull requests.
- The job installs Jekyll and plugins using `ruby/setup-ruby`.
- It then runs `jekyll build` to verify the site compiles.

