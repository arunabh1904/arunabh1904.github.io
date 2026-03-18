import { getCollection, type CollectionEntry } from 'astro:content';

export type PostEntry = CollectionEntry<'posts'>;
export type Section = PostEntry['data']['section'];

export function sortPostsAscending(posts: PostEntry[]) {
  return [...posts].sort(
    (left, right) => left.data.date.getTime() - right.data.date.getTime(),
  );
}

export function sortPostsDescending(posts: PostEntry[]) {
  return [...posts].sort(
    (left, right) => right.data.date.getTime() - left.data.date.getTime(),
  );
}

export async function getAllPosts() {
  return sortPostsDescending(await getCollection('posts'));
}

export async function getPostsBySection(
  section: Section,
  order: 'asc' | 'desc' = 'desc',
) {
  const posts = (await getCollection('posts')).filter(
    (post) => post.data.section === section,
  );
  return order === 'asc' ? sortPostsAscending(posts) : sortPostsDescending(posts);
}

export async function getPostsByField() {
  const posts = await getPostsBySection('paper-shorts', 'asc');
  const grouped = new Map<string, PostEntry[]>();

  for (const post of posts) {
    const key = post.data.field ?? 'Other';
    if (!grouped.has(key)) {
      grouped.set(key, []);
    }
    grouped.get(key)?.push(post);
  }

  return [...grouped.entries()].map(([field, items]) => ({ field, items }));
}

export function groupPostsByTag(posts: PostEntry[]) {
  const grouped = new Map<string, PostEntry[]>();

  for (const post of posts) {
    const tags = post.data.tags.length > 0 ? post.data.tags : ['Other'];
    for (const tag of tags) {
      if (!grouped.has(tag)) {
        grouped.set(tag, []);
      }
      grouped.get(tag)?.push(post);
    }
  }

  return [...grouped.entries()]
    .map(([tag, items]) => ({
      tag,
      items: sortPostsAscending(items),
    }))
    .sort((left, right) => left.tag.localeCompare(right.tag));
}

export function trimLeadingSlash(value: string) {
  return value.startsWith('/') ? value.slice(1) : value;
}
