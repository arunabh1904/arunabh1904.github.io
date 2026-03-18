import { getCollection, type CollectionEntry } from 'astro:content';
import {
  groupPostsByField,
  groupPostsByTag,
  sortPosts,
  sortPostsAscending,
  sortPostsDescending,
  toStaticPostSlug,
  type SortOrder,
} from './post-utils';

export type PostEntry = CollectionEntry<'posts'>;
export type Section = PostEntry['data']['section'];
export type { SortOrder };
export {
  groupPostsByField,
  groupPostsByTag,
  sortPosts,
  sortPostsAscending,
  sortPostsDescending,
  toStaticPostSlug,
};

export async function getAllPosts(order: SortOrder = 'desc') {
  return sortPosts(await getCollection('posts'), order);
}

export async function getPostsBySection(section: Section, order: SortOrder = 'desc') {
  return sortPosts(
    (await getCollection('posts')).filter((post) => post.data.section === section),
    order,
  );
}

export async function getPostsByField(order: SortOrder = 'asc') {
  return groupPostsByField(await getPostsBySection('paper-shorts', order)).map(
    ({ field, posts }) => ({ field, items: posts }),
  );
}
