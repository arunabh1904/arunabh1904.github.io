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
    section: z.enum([
      'paper-shorts',
      'blog',
      'build-intuition',
      'revision-notes',
      'ai-generated-reports',
    ]),
    postSlug: z.string(),
    legacyPath: z.string(),
    tags: z.array(z.string()).default(['Other']),
    field: z.string().optional(),
    summary: z.string().optional(),
  }),
});

export const collections = { posts };
