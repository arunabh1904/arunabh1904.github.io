export const BLOG_GROUPS = [
  {
    id: 'projects',
    title: 'Projects & systems',
    description:
      'The useful part starts after the demo: turning a local model into a system I actually rely on.',
  },
  {
    id: 'local-ai-lab',
    title: 'Local AI lab',
    description:
      'What actually fits, how fast it runs, and where a 64 GB Mac stops being enough.',
  },
  {
    id: 'research-guides',
    title: 'Long-form research guides',
    description:
      'I follow the mechanism until a stack of papers becomes a mental model I can use.',
  },
  {
    id: 'essays',
    title: 'Essays & career',
    description:
      'The harder questions behind the work: what to learn, how to choose, and what remains valuable.',
  },
] as const;

export type BlogGroupId = (typeof BLOG_GROUPS)[number]['id'];

interface BlogPostLike {
  data: {
    blogGroup?: BlogGroupId;
  };
}

export function groupBlogPosts<TPost extends BlogPostLike>(posts: readonly TPost[]) {
  const groups = BLOG_GROUPS.map((group) => ({ ...group, posts: [] as TPost[] }));
  const groupById = new Map(groups.map((group) => [group.id, group]));
  const ungrouped: TPost[] = [];

  for (const post of posts) {
    const group = post.data.blogGroup ? groupById.get(post.data.blogGroup) : undefined;
    if (group) {
      group.posts.push(post);
    } else {
      ungrouped.push(post);
    }
  }

  return [
    ...groups.filter((group) => group.posts.length > 0),
    ...(ungrouped.length > 0
      ? [
          {
            id: 'other',
            title: 'Other writing',
            description: 'Writing that has not found a better shelf yet.',
            posts: ungrouped,
          },
        ]
      : []),
  ];
}
