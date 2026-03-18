import rss from '@astrojs/rss';
import { getAllPosts } from '../lib/content';
import { SITE_DESCRIPTION, SITE_TITLE } from '../lib/site';

export async function GET(context) {
  const posts = await getAllPosts();

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
