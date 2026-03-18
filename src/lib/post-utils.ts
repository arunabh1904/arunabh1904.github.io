export type SortOrder = 'asc' | 'desc';

interface PostLike {
  data: {
    date: Date;
    legacyPath: string;
    tags: string[];
    field?: string;
  };
}

export function sortPosts<TPost extends PostLike>(
  posts: readonly TPost[],
  order: SortOrder = 'desc',
) {
  const direction = order === 'asc' ? 1 : -1;
  return [...posts].sort(
    (left, right) => (left.data.date.getTime() - right.data.date.getTime()) * direction,
  );
}

export function sortPostsAscending<TPost extends PostLike>(posts: readonly TPost[]) {
  return sortPosts(posts, 'asc');
}

export function sortPostsDescending<TPost extends PostLike>(posts: readonly TPost[]) {
  return sortPosts(posts, 'desc');
}

export function groupPostsByField<TPost extends PostLike>(posts: readonly TPost[]) {
  const grouped = new Map<string, TPost[]>();

  for (const post of posts) {
    const field = post.data.field ?? 'Other';
    if (!grouped.has(field)) {
      grouped.set(field, []);
    }
    grouped.get(field)?.push(post);
  }

  return [...grouped.entries()].map(([field, items]) => ({
    field,
    posts: sortPostsAscending(items),
  }));
}

export function groupPostsByTag<TPost extends PostLike>(posts: readonly TPost[]) {
  const grouped = new Map<string, TPost[]>();

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

export function toStaticPostSlug(legacyPath: string) {
  return legacyPath.replace(/^\//, '').replace(/\.html$/, '');
}
