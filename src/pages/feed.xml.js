import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';
import { sortPostsDescending } from '../lib/content';
import { SITE_DESCRIPTION, SITE_TITLE } from '../lib/site';

export async function GET(context) {
  const posts = sortPostsDescending(await getCollection('posts'));

  return rss({
    title: SITE_TITLE,
    description: SITE_DESCRIPTION,
    site: context.site,
    items: posts.map((post) => ({
      title: post.data.title,
      description: post.data.summary ?? SITE_DESCRIPTION,
      pubDate: post.data.date,
      link: post.data.legacyPath,
    })),
  });
}
