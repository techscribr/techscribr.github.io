# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

TechScribr is a Jekyll blog built on the [Chirpy](https://github.com/cotes2020/jekyll-theme-chirpy) theme (installed as a gem, `jekyll-theme-chirpy ~> 7.4`). It is a content repo: the bulk of the work is writing Markdown posts in [_posts/](_posts/), not editing site code. Site theme files (layouts, includes, sass) live inside the gem — use `bundle info --path jekyll-theme-chirpy` to locate them when overriding is necessary.

## Commands

- **Serve locally (livereload):** `bash tools/run.sh` — runs `bundle exec jekyll s -l -H 127.0.0.1`. Add `-p` for production mode, `-H <host>` to change bind address.
- **Build + link-check:** `bash tools/test.sh` — builds to `_site/` with `JEKYLL_ENV=production` and runs `htmlproofer` (external URLs disabled, localhost ignored).
- **Install deps:** `bundle install`.

## Content architecture

- **Posts:** [_posts/](_posts/) is organized into topic subdirectories (`llm`, `adtech`, `deep-learning`, `feature-engineering`, `fundamental-ml-concepts`, `ml-algos`, `ml-engineering`, `probability-&-statistics`). Jekyll flattens these at build time — the subfolder is purely for authoring organization and does **not** affect the URL. Permalink is `/posts/:title/` (set in [_config.yml](_config.yml)).
- **Filename convention:** `YYYY-MM-DD-slug.md`. The date in the filename must match the `date:` front matter; both drive ordering.
- **Front matter conventions** (see existing posts for reference): `title`, `date` with timezone `+0530`, `categories: [<Category>]`, `tags: [<Tag>]`, `math: true` when LaTeX is used. Defaults in [_config.yml](_config.yml) already set `layout: post`, `comments: true`, `toc: true` — don't re-specify.
- **Last-modified dates** are auto-derived from git history by [_plugins/posts-lastmod-hook.rb](_plugins/posts-lastmod-hook.rb) (reads `git log` for each post path). This means `last_modified_at` appears only after a post has more than one commit touching it — local uncommitted edits won't update it.
- **Tabs:** [_tabs/](_tabs/) holds the top-level pages (about, archives, categories, tags). Permalink is `/:title/`.
- **Assets:** images and static files go in [assets/](assets/) (subfolders `img/`, `html/`, `lib/`).

## Site configuration

[_config.yml](_config.yml) is the single source of truth for site metadata, permalinks, kramdown/rouge highlighting (line numbers enabled on block code), and the `jekyll-archives` setup for category/tag pages. Timezone is `Asia/Kolkata`. The `tools/` directory and `README.md` are excluded from the build.

## Deployment

GitHub Pages CD is wired via [.github/](.github/) workflows (Chirpy's standard CD). Pushing to `main` publishes the site — treat post dates and content accordingly before committing.
