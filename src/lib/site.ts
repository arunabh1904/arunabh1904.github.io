export const SITE_TITLE = 'Arunabh.BLOG';
export const SITE_DESCRIPTION =
  'A personal blog on computer vision, robotics, and more.';

export const CONTACT_LINKS = [
  { href: 'mailto:arunabh1904@gmail.com', iconName: 'email', label: 'Email' },
  { href: 'https://github.com/arunabh1904', iconName: 'github', label: 'GitHub' },
  {
    href: 'https://www.linkedin.com/in/arunabh-mishra',
    iconName: 'linkedin',
    label: 'LinkedIn',
  },
  {
    href: 'https://twitter.com/ArunabhMishra8',
    iconName: 'x',
    label: 'X',
  },
] as const;

export const PDF_REPORTS = [
  {
    href: '/assets/pdfs/VLM Research Summary.pdf',
    label: 'A survey of VLMs.',
    dateLabel: 'May 17, 2025',
  },
] as const;
