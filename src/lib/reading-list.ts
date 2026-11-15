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
    title: 'No AI Slop: A Manifesto',
    author: 'Speakers and attendees of AIE Code Summit 2026',
    href: 'https://www.slopmanifesto.com/',
    dateLabel: '2026',
    summary:
      'A shared standard for AI-assisted work: own what you ship, respect the reader’s attention, and fix recurring mistakes in the process that produces them.',
    group: 'writing-craft',
  },
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
    title: 'Impact, agency, and taste',
    author: 'Ben Kuhn',
    href: 'https://www.benkuhn.net/impact/',
    dateLabel: 'Apr 2025',
    summary:
      'How agency and taste help researchers and engineers choose valuable problems and carry them through.',
    group: 'writing-craft',
  },
  {
    title: 'End-to-End 4-Bit Training',
    author: 'Jack Cook and Thinking Machines Lab',
    href: 'https://thinkingmachines.ai/blog/end-to-end-4-bit-training/',
    dateLabel: 'Nov 13, 2026',
    summary:
      'The NVFP4 training recipe for Inkling 2 Small: MXFP8 backward passes, stochastic rounding to control gradient bias, and quantized MoE communication to turn faster arithmetic into training speedups.',
    group: 'research-blogs',
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
