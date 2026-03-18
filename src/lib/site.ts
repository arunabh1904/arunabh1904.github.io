export const SITE_TITLE = 'Arunabh.BLOG';
export const SITE_DESCRIPTION =
  'A personal blog on computer vision, robotics, and more.';

export const CONTACT_LINKS = [
  { href: 'mailto:arunabh1904@gmail.com', iconClass: 'fas fa-envelope fa-lg', label: 'Email' },
  { href: 'https://github.com/arunabh1904', iconClass: 'fab fa-github fa-lg', label: 'GitHub' },
  {
    href: 'https://www.linkedin.com/in/arunabh-mishra',
    iconClass: 'fab fa-linkedin fa-lg',
    label: 'LinkedIn',
  },
  {
    href: 'https://twitter.com/ArunabhMishra8',
    iconClass: 'fab fa-twitter fa-lg',
    label: 'Twitter',
  },
] as const;

export const PDF_REPORTS = [
  {
    href: '/assets/pdfs/VLM Research Summary.pdf',
    label: 'A survey of VLMs.',
    dateLabel: 'May 17, 2025',
  },
] as const;
