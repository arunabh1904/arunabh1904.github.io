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

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

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

function initBlogImageViewer(signal) {
  const postRoot = document.querySelector('[data-post-section="blog"]');
  if (!(postRoot instanceof HTMLElement)) {
    return;
  }

  const sourceImages = Array.from(postRoot.querySelectorAll('img')).filter(
    (image) => image instanceof HTMLImageElement && (image.currentSrc !== '' || image.src !== ''),
  );
  if (sourceImages.length === 0) {
    return;
  }

  const viewer = document.createElement('div');
  viewer.className = 'image-viewer';
  viewer.dataset.imageViewer = '';
  viewer.hidden = true;
  viewer.setAttribute('role', 'dialog');
  viewer.setAttribute('aria-modal', 'true');
  viewer.setAttribute('aria-label', 'Image viewer');

  const panel = document.createElement('div');
  panel.className = 'image-viewer__panel';

  const toolbar = document.createElement('div');
  toolbar.className = 'image-viewer__toolbar';

  const caption = document.createElement('p');
  caption.className = 'image-viewer__caption';

  const controls = document.createElement('div');
  controls.className = 'image-viewer__controls';

  const zoomHint = document.createElement('span');
  zoomHint.className = 'image-viewer__zoom-hint';
  zoomHint.textContent = 'Scroll to zoom';

  const zoomValue = document.createElement('span');
  zoomValue.className = 'image-viewer__zoom-value';
  zoomValue.setAttribute('aria-live', 'polite');
  zoomValue.textContent = '100%';

  const closeButton = document.createElement('button');
  closeButton.type = 'button';
  closeButton.className = 'image-viewer__close';
  closeButton.textContent = 'Close';
  closeButton.setAttribute('aria-label', 'Close image viewer');

  const stage = document.createElement('div');
  stage.className = 'image-viewer__stage';
  stage.tabIndex = 0;
  stage.setAttribute('aria-label', 'Image viewport. Scroll to zoom and use the scrollbars to pan.');

  const canvas = document.createElement('div');
  canvas.className = 'image-viewer__canvas';

  const viewerImage = document.createElement('img');
  viewerImage.className = 'image-viewer__image';
  viewerImage.alt = '';

  controls.append(zoomHint, zoomValue, closeButton);
  toolbar.append(caption, controls);
  canvas.append(viewerImage);
  stage.append(canvas);
  panel.append(toolbar, stage);
  viewer.append(panel);
  document.body.append(viewer);

  const state = {
    source: null,
    baseWidth: 0,
    baseHeight: 0,
    zoom: 1,
  };
  let previouslyFocusedElement = null;

  const applyZoom = () => {
    const displayWidth = Math.max(1, state.baseWidth * state.zoom);
    const displayHeight = Math.max(1, state.baseHeight * state.zoom);

    viewerImage.style.width = `${displayWidth}px`;
    viewerImage.style.height = `${displayHeight}px`;
    canvas.style.width = `${Math.max(stage.clientWidth, displayWidth)}px`;
    canvas.style.height = `${Math.max(stage.clientHeight, displayHeight)}px`;
    zoomValue.textContent = `${Math.round(state.zoom * 100)}%`;
  };

  const fitImageToViewport = (preserveZoom = false) => {
    if (!(state.source instanceof HTMLImageElement)) {
      return;
    }

    const sourceRect = state.source.getBoundingClientRect();
    const imageWidth = state.source.naturalWidth || Math.round(sourceRect.width) || 1;
    const imageHeight = state.source.naturalHeight || Math.round(sourceRect.height) || 1;
    const availableWidth = Math.max(1, stage.clientWidth - 32);
    const availableHeight = Math.max(1, stage.clientHeight - 32);
    const fitScale = Math.min(1, availableWidth / imageWidth, availableHeight / imageHeight);

    state.baseWidth = imageWidth * fitScale;
    state.baseHeight = imageHeight * fitScale;
    state.zoom = preserveZoom ? clamp(state.zoom, 0.5, 6) : 1;
    applyZoom();
  };

  const closeViewer = () => {
    if (viewer.hidden) {
      return;
    }

    viewer.hidden = true;
    document.body.classList.remove('image-viewer-open');
    viewerImage.removeAttribute('src');
    viewerImage.style.removeProperty('width');
    viewerImage.style.removeProperty('height');
    canvas.style.removeProperty('width');
    canvas.style.removeProperty('height');
    state.source = null;

    if (previouslyFocusedElement instanceof HTMLElement && previouslyFocusedElement.isConnected) {
      previouslyFocusedElement.focus({ preventScroll: true });
    }
    previouslyFocusedElement = null;
  };

  const openViewer = (sourceImage) => {
    state.source = sourceImage;
    previouslyFocusedElement = document.activeElement;
    viewerImage.src = sourceImage.currentSrc || sourceImage.src;
    viewerImage.alt = sourceImage.alt;
    caption.textContent = sourceImage.alt || 'Image preview';
    viewer.hidden = false;
    document.body.classList.add('image-viewer-open');

    requestAnimationFrame(() => {
      fitImageToViewport();
      stage.scrollTo({ left: 0, top: 0 });
      closeButton.focus({ preventScroll: true });
    });
  };

  sourceImages.forEach((image) => {
    const linkedImage = image.closest('a[href]');
    const activationTarget = linkedImage instanceof HTMLAnchorElement ? linkedImage : image;
    const description = image.alt.trim() || 'image';

    image.classList.add('post-image--zoomable');
    if (activationTarget instanceof HTMLAnchorElement) {
      activationTarget.classList.add('post-image-link--zoomable');
      activationTarget.setAttribute('aria-label', `Open ${description} in image viewer`);
      activationTarget.setAttribute('aria-haspopup', 'dialog');
    } else {
      image.tabIndex = 0;
      image.setAttribute('role', 'button');
      image.setAttribute('aria-label', `Open ${description} in image viewer`);
      image.setAttribute('aria-haspopup', 'dialog');
      image.addEventListener('keydown', (event) => {
        if (event.key !== 'Enter' && event.key !== ' ') {
          return;
        }

        event.preventDefault();
        openViewer(image);
      }, { signal });
    }

    activationTarget.addEventListener('click', (event) => {
      event.preventDefault();
      openViewer(image);
    }, { signal });
  });

  viewer.addEventListener('click', (event) => {
    if (event.target === viewer) {
      closeViewer();
    }
  }, { signal });
  closeButton.addEventListener('click', closeViewer, { signal });

  stage.addEventListener('wheel', (event) => {
    if (viewer.hidden || state.baseWidth === 0 || state.baseHeight === 0) {
      return;
    }

    event.preventDefault();

    const previousWidth = state.baseWidth * state.zoom;
    const previousHeight = state.baseHeight * state.zoom;
    const stageRect = stage.getBoundingClientRect();
    const cursorX = event.clientX - stageRect.left;
    const cursorY = event.clientY - stageRect.top;
    const xRatio = clamp((stage.scrollLeft + cursorX) / previousWidth, 0, 1);
    const yRatio = clamp((stage.scrollTop + cursorY) / previousHeight, 0, 1);

    state.zoom = clamp(state.zoom * (event.deltaY < 0 ? 1.12 : 0.88), 0.5, 6);
    applyZoom();

    stage.scrollLeft = Math.max(0, xRatio * state.baseWidth * state.zoom - cursorX);
    stage.scrollTop = Math.max(0, yRatio * state.baseHeight * state.zoom - cursorY);
  }, { passive: false, signal });

  viewer.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      event.preventDefault();
      closeViewer();
      return;
    }

    if (event.key === '+' || event.key === '=') {
      event.preventDefault();
      state.zoom = clamp(state.zoom * 1.12, 0.5, 6);
      applyZoom();
      return;
    }

    if (event.key === '-') {
      event.preventDefault();
      state.zoom = clamp(state.zoom * 0.88, 0.5, 6);
      applyZoom();
      return;
    }

    if (event.key === '0') {
      event.preventDefault();
      state.zoom = 1;
      applyZoom();
      return;
    }

    if (event.key === 'Tab') {
      const focusable = [closeButton, stage];
      const currentIndex = focusable.indexOf(document.activeElement);
      const nextIndex = event.shiftKey
        ? (currentIndex <= 0 ? focusable.length - 1 : currentIndex - 1)
        : (currentIndex + 1) % focusable.length;

      event.preventDefault();
      focusable[nextIndex].focus();
    }
  }, { signal });

  window.addEventListener('resize', () => {
    if (!viewer.hidden) {
      fitImageToViewport(true);
    }
  }, { passive: true, signal });

  signal.addEventListener('abort', () => {
    document.body.classList.remove('image-viewer-open');
    viewer.remove();
  }, { once: true });
}

async function initBlogFrameExplainers(signal) {
  const explainers = Array.from(document.querySelectorAll('[data-blog-frame-explainer]')).filter(
    (node) => node instanceof HTMLElement && node.dataset.frameExplainerReady !== 'true',
  );
  if (explainers.length === 0) {
    return;
  }

  let manifest;
  try {
    const response = await fetch('/assets/images/blog-explainer-frames/manifest.json', { signal });
    if (!response.ok) {
      throw new Error(`Explainer manifest returned ${response.status}`);
    }
    manifest = await response.json();
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      return;
    }
    console.error('Could not initialize Blog frame explainers.', error);
    return;
  }

  explainers.forEach((explainer) => {
    const storyName = explainer.dataset.blogFrameExplainer;
    const story = manifest[storyName];
    const image = explainer.querySelector('img');
    const link = explainer.querySelector('a');
    if (!storyName || !story || !Array.isArray(story.frames) || story.frames.length === 0 || !(image instanceof HTMLImageElement)) {
      return;
    }

    const baseAlt = image.alt.trim() || 'Technical explainer';
    const controls = document.createElement('div');
    controls.className = 'blog-frame-explainer__controls';

    const previous = document.createElement('button');
    previous.type = 'button';
    previous.className = 'blog-frame-explainer__button';
    previous.textContent = '←';
    previous.setAttribute('aria-label', 'Previous explainer frame');

    const range = document.createElement('input');
    range.className = 'blog-frame-explainer__range';
    range.type = 'range';
    range.min = '0';
    range.max = String(story.frames.length - 1);
    range.step = '1';
    range.value = '0';
    range.setAttribute('aria-label', 'Explainer frame');

    const next = document.createElement('button');
    next.type = 'button';
    next.className = 'blog-frame-explainer__button';
    next.textContent = '→';
    next.setAttribute('aria-label', 'Next explainer frame');

    const status = document.createElement('output');
    status.className = 'blog-frame-explainer__status';
    status.setAttribute('aria-live', 'polite');

    const hint = document.createElement('span');
    hint.className = 'blog-frame-explainer__hint';
    hint.textContent = 'Use the slider or ← → keys';

    let index = 0;
    const update = (nextIndex) => {
      index = clamp(nextIndex, 0, story.frames.length - 1);
      const frame = story.frames[index];
      image.src = frame.src;
      image.alt = `${baseAlt} Frame ${index + 1}: ${frame.description || frame.title}`;
      if (link instanceof HTMLAnchorElement) {
        link.href = frame.src;
      }
      range.value = String(index);
      range.setAttribute('aria-valuetext', `${index + 1} of ${story.frames.length}: ${frame.title}`);
      status.textContent = `${index + 1} / ${story.frames.length}`;
      previous.disabled = index === 0;
      next.disabled = index === story.frames.length - 1;

      [story.frames[index - 1], story.frames[index + 1]].forEach((adjacentFrame) => {
        if (adjacentFrame?.src) {
          const preload = new Image();
          preload.src = adjacentFrame.src;
        }
      });
    };

    previous.addEventListener('click', () => update(index - 1), { signal });
    next.addEventListener('click', () => update(index + 1), { signal });
    range.addEventListener('input', () => update(Number(range.value)), { signal });
    explainer.addEventListener('keydown', (event) => {
      if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') {
        return;
      }
      event.preventDefault();
      update(index + (event.key === 'ArrowRight' ? 1 : -1));
    }, { signal });

    controls.append(previous, range, next, status, hint);
    explainer.append(controls);
    explainer.dataset.frameExplainerReady = 'true';
    explainer.tabIndex = 0;
    explainer.setAttribute('role', 'group');
    explainer.setAttribute('aria-label', `${baseAlt}. Manual frame-by-frame explainer.`);
    update(0);
  });
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
  initBlogFrameExplainers(signal);
  initBlogImageViewer(signal);
}

document.addEventListener('astro:page-load', initSiteControls);
