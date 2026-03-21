const THEME_STORAGE_KEY = 'theme';
const MUSIC_PROVIDER_KEY = 'musicPlayerProvider';
const MUSIC_PLAYER_NAME = 'music-player';
const MUSIC_PLAYER_FEATURES = 'width=420,height=560,resizable=yes,scrollbars=yes';
const MUSIC_PAGE_PATH = '/music.html';
const DEFAULT_MUSIC_PROVIDER = 'spotify';

function normalizeProvider(value) {
  return value === 'youtube' ? 'youtube' : DEFAULT_MUSIC_PROVIDER;
}

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem(THEME_STORAGE_KEY, theme);
}

function getMusicPlayerUrl(provider) {
  const url = new URL(MUSIC_PAGE_PATH, window.location.origin);
  url.searchParams.set('provider', normalizeProvider(provider));
  return url.toString();
}

function setMusicMenuState(control, isOpen) {
  const toggle = control.querySelector('[data-music-menu-toggle]');
  const menu = control.querySelector('[data-music-menu]');

  if (!(toggle instanceof HTMLButtonElement) || !(menu instanceof HTMLElement)) {
    return;
  }

  toggle.setAttribute('aria-expanded', String(isOpen));
  toggle.classList.toggle('page-controls__button--active', isOpen);
  menu.hidden = !isOpen;
}

function closeMusicMenus(exceptControl = null) {
  document.querySelectorAll('[data-music-control]').forEach((control) => {
    if (exceptControl && control === exceptControl) {
      return;
    }

    if (control instanceof HTMLElement) {
      setMusicMenuState(control, false);
    }
  });
}

function syncMusicProviderButtons(provider) {
  document.querySelectorAll('[data-music-provider]').forEach((button) => {
    if (!(button instanceof HTMLButtonElement)) {
      return;
    }

    const isActive = button.dataset.musicProvider === provider;
    button.classList.toggle('music-menu__item--active', isActive);
    button.setAttribute('aria-pressed', String(isActive));
  });
}

function openMusicPlayer(provider) {
  const nextProvider = normalizeProvider(provider);
  const requestedUrl = getMusicPlayerUrl(nextProvider);

  localStorage.setItem(MUSIC_PROVIDER_KEY, nextProvider);
  syncMusicProviderButtons(nextProvider);

  const playerWindow = window.open(requestedUrl, MUSIC_PLAYER_NAME, MUSIC_PLAYER_FEATURES);
  if (!playerWindow) {
    window.location.href = requestedUrl;
  }
}

function initSiteControls() {
  const html = document.documentElement;
  const savedProvider = normalizeProvider(localStorage.getItem(MUSIC_PROVIDER_KEY));

  syncMusicProviderButtons(savedProvider);

  document.querySelectorAll('[data-theme-toggle]').forEach((button) => {
    button.addEventListener('click', () => {
      const nextTheme = html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      applyTheme(nextTheme);
    });
  });

  document.querySelectorAll('[data-home-button]').forEach((button) => {
    button.addEventListener('click', () => {
      window.location.href = '/';
    });
  });

  document.querySelectorAll('[data-music-control]').forEach((control) => {
    if (!(control instanceof HTMLElement)) {
      return;
    }

    const toggle = control.querySelector('[data-music-menu-toggle]');
    const providerButtons = control.querySelectorAll('[data-music-provider]');

    if (toggle instanceof HTMLButtonElement) {
      toggle.addEventListener('click', (event) => {
        event.stopPropagation();

        const isOpen = toggle.getAttribute('aria-expanded') === 'true';
        closeMusicMenus(control);
        setMusicMenuState(control, !isOpen);
      });
    }

    providerButtons.forEach((button) => {
      if (!(button instanceof HTMLButtonElement)) {
        return;
      }

      button.addEventListener('click', () => {
        openMusicPlayer(button.dataset.musicProvider);
        closeMusicMenus();
      });
    });
  });

  document.addEventListener('click', (event) => {
    const target = event.target;
    if (!(target instanceof Element) || target.closest('[data-music-control]')) {
      return;
    }

    closeMusicMenus();
  });

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      closeMusicMenus();
    }
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initSiteControls, { once: true });
} else {
  initSiteControls();
}
