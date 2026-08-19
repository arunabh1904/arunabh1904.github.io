export const BLOG_GROUPS = [
  {
    id: 'research-guides',
    title: 'Long-form research guides',
    description:
      'Deep dives into perception, multimodal models, reinforcement learning, and scaling.',
  },
  {
    id: 'essays',
    title: 'Essays & career',
    description:
      'Notes on research, careers, and what remains valuable as AI changes the work.',
  },
  {
    id: 'projects',
    title: 'Projects & systems',
    description:
      'Local agents, Blog audio, and the systems I built around open models.',
  },
  {
    id: 'local-ai-lab',
    title: 'Local AI lab',
    description:
      'Benchmarks and fit checks for running open-weight models on a 64 GB Mac.',
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
