const THEME_STORAGE_KEY = 'theme';

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem(THEME_STORAGE_KEY, theme);
}

function slugifyHeading(text) {
  return text
    .toLowerCase()
    .trim()
    .replace(/['".,!?()[\]{}:;]+/g, '')
    .replace(/&/g, ' and ')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

function ensureHeadingIds(headings) {
  const usedIds = new Set(
    Array.from(document.querySelectorAll('[id]'))
      .map((element) => element.id)
      .filter(Boolean),
  );

  headings.forEach((heading) => {
    if (heading.id) {
      usedIds.add(heading.id);
      return;
    }

    const baseId = slugifyHeading(heading.textContent || '') || 'section';
    let nextId = baseId;
    let suffix = 2;

    while (usedIds.has(nextId)) {
      nextId = `${baseId}-${suffix}`;
      suffix += 1;
    }

    heading.id = nextId;
    usedIds.add(nextId);
  });
}

let siteControlsAbortController;

function buildSectionsNav(signal) {
  const root = document.querySelector('[data-sections-root]');
  const aside = document.querySelector('[data-sections-nav]');
  const list = document.querySelector('[data-sections-list]');

  if (!(root instanceof HTMLElement) || !(aside instanceof HTMLElement) || !(list instanceof HTMLElement)) {
    return;
  }

  const headings = Array.from(root.querySelectorAll('h2, h3, h4')).filter(
    (heading) => heading instanceof HTMLHeadingElement && heading.textContent?.trim(),
  );

  if (headings.length === 0) {
    aside.hidden = true;
    return;
  }

  ensureHeadingIds(headings);
  list.replaceChildren();

  const linkEntries = headings.map((heading) => {
    const link = document.createElement('a');
    link.href = `#${heading.id}`;
    link.textContent = heading.textContent?.trim() ?? heading.id;
    link.className = 'sections-nav__link';
    link.dataset.level = heading.tagName.slice(1);
    list.append(link);

    return { heading, link };
  });

  aside.hidden = false;

  const updateActiveSection = () => {
    const offset = 140;
    let current = linkEntries[0];

    linkEntries.forEach((entry) => {
      if (entry.heading.getBoundingClientRect().top - offset <= 0) {
        current = entry;
      }
    });

    linkEntries.forEach((entry) => {
      entry.link.classList.toggle('sections-nav__link--active', entry === current);
      entry.link.setAttribute('aria-current', entry === current ? 'true' : 'false');
    });
  };

  updateActiveSection();
  window.addEventListener('scroll', updateActiveSection, { passive: true, signal });
  window.addEventListener('resize', updateActiveSection, { passive: true, signal });
}

function initSiteControls() {
  siteControlsAbortController?.abort();
  siteControlsAbortController = new AbortController();
  const { signal } = siteControlsAbortController;
  const html = document.documentElement;

  document.querySelectorAll('[data-theme-toggle]').forEach((button) => {
    button.addEventListener('click', () => {
      const nextTheme = html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      applyTheme(nextTheme);
    }, { signal });
  });

  buildSectionsNav(signal);
}

document.addEventListener('astro:page-load', initSiteControls);
