export const BLOG_GROUPS = [
  {
    id: 'projects',
    title: 'Projects & systems',
    description:
      'End-to-end builds, including the local Qwen3-TTS blog narrator and a local-weight agent.',
  },
  {
    id: 'local-ai-lab',
    title: 'Local AI lab',
    description:
      'Measured model-fit and inference decisions for running open weights on a 64 GB MacBook Pro.',
  },
  {
    id: 'research-guides',
    title: 'Long-form research guides',
    description:
      'Mechanism-first explanations of attention, scaling, multimodal learning, robot policies, and perception.',
  },
  {
    id: 'essays',
    title: 'Essays & career',
    description:
      'Personal experience and working principles for research, engineering, and a career in robotics.',
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
            description: 'Posts awaiting a more specific reading path.',
            posts: ungrouped,
          },
        ]
      : []),
  ];
}
