export type ReadingListGroupId = 'writing-craft' | 'research-blogs';

export interface ReadingListItem {
  title: string;
  author: string;
  href: string;
  dateLabel: string;
  summary: string;
  group: ReadingListGroupId;
}

export interface ReadingListGroup {
  id: ReadingListGroupId;
  title: string;
  description: string;
}

export const READING_LIST_GROUPS: ReadingListGroup[] = [
  {
    id: 'writing-craft',
    title: 'Writing & craft',
    description: 'How to think clearly, explain precisely, and use AI without flattening your voice.',
  },
  {
    id: 'research-blogs',
    title: 'Research blogs',
    description: 'Writers who turn fast-moving research into durable technical understanding.',
  },
];

export const READING_LIST_ITEMS: ReadingListItem[] = [
  {
    title: 'Writing in the Age of LLMs',
    author: 'Shreya Shankar',
    href: 'https://www.sh-reya.com/blog/ai-writing/',
    dateLabel: 'Jun 16, 2025',
    summary:
      'A practical essay on writing with LLMs, avoiding low-density prose, and revising for clarity, rhythm, and judgment.',
    group: 'writing-craft',
  },
  {
    title: "Lil'Log",
    author: 'Lilian Weng',
    href: 'https://lilianweng.github.io/',
    dateLabel: 'Since 2017',
    summary:
      'Deep learning and AI research notes with careful long-form explainers on agents, alignment, diffusion, scaling laws, and related systems.',
    group: 'research-blogs',
  },
];
