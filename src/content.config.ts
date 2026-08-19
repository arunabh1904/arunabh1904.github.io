import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

const posts = defineCollection({
  loader: glob({
    base: './src/content/posts',
    pattern: '**/*.{md,mdx}',
  }),
  schema: z.object({
    title: z.string(),
    date: z.coerce.date(),
    section: z.enum(['paper-shorts', 'blog', 'revision-notes']),
    blogGroup: z
      .enum(['projects', 'local-ai-lab', 'research-guides', 'essays'])
      .optional(),
    postSlug: z.string(),
    legacyPath: z.string(),
    tags: z.array(z.string()).default(['Other']),
    field: z.string().optional(),
    topics: z
      .array(
        z.enum([
          'multimodal',
          'embodied',
          'autonomy',
          'learning',
          'generation',
          'language-systems',
        ]),
      )
      .default([]),
    summary: z.string().optional(),
  }).refine((post) => post.section !== 'blog' || post.blogGroup !== undefined, {
    message: 'Blog posts must declare a blogGroup so they remain findable in the Blog index.',
    path: ['blogGroup'],
  }),
});

export const collections = { posts };
