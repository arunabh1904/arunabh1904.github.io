const SPOTIFY_SRC =
  'https://open.spotify.com/embed/playlist/4RHYceSp9R1bHyL0dDqTuQ?utm_source=generator&theme=0';
const YOUTUBE_VIDEO_IDS = ['kGuGH_UvvxA', 'z8Dz-IFFFY4'];
const YOUTUBE_SRC = `https://www.youtube.com/embed/${YOUTUBE_VIDEO_IDS[0]}?playlist=${YOUTUBE_VIDEO_IDS
  .slice(1)
  .join(',')}&rel=0`;
const MUSIC_PROVIDER_KEY = 'musicPlayerProvider';
const MUSIC_SRC_KEY = 'musicPlayerSrc';
const DEFAULT_MUSIC_PROVIDER = 'spotify';

function normalizeProvider(value) {
  return value === 'youtube' ? 'youtube' : DEFAULT_MUSIC_PROVIDER;
}

function getRequestedProvider() {
  const params = new URLSearchParams(window.location.search);
  const queryProvider = params.get('provider');

  if (queryProvider === 'spotify' || queryProvider === 'youtube') {
    return queryProvider;
  }

  return normalizeProvider(localStorage.getItem(MUSIC_PROVIDER_KEY));
}

function getProviderSrc(provider) {
  return provider === 'youtube' ? YOUTUBE_SRC : SPOTIFY_SRC;
}

function syncUrl(provider) {
  const url = new URL(window.location.href);
  url.searchParams.set('provider', provider);
  window.history.replaceState({}, '', url);
}

function initMusicPlayer() {
  const spotifyOption = document.getElementById('choose-spotify');
  const youtubeOption = document.getElementById('choose-youtube');
  const closePlayerBtn = document.getElementById('close-music-player');
  const musicIframe = document.getElementById('music-iframe');

  if (
    !(spotifyOption instanceof HTMLButtonElement) ||
    !(youtubeOption instanceof HTMLButtonElement) ||
    !(closePlayerBtn instanceof HTMLButtonElement) ||
    !(musicIframe instanceof HTMLIFrameElement)
  ) {
    return;
  }

  function syncChoiceState(provider) {
    spotifyOption.classList.toggle('music-choice--active', provider === 'spotify');
    spotifyOption.setAttribute('aria-pressed', String(provider === 'spotify'));
    youtubeOption.classList.toggle('music-choice--active', provider === 'youtube');
    youtubeOption.setAttribute('aria-pressed', String(provider === 'youtube'));
  }

  function renderProvider(provider) {
    const nextProvider = normalizeProvider(provider);
    const nextSrc = getProviderSrc(nextProvider);

    document.body.dataset.musicProvider = nextProvider;
    musicIframe.src = nextSrc;
    localStorage.setItem(MUSIC_PROVIDER_KEY, nextProvider);
    localStorage.setItem(MUSIC_SRC_KEY, nextSrc);
    syncUrl(nextProvider);
    syncChoiceState(nextProvider);
  }

  spotifyOption.addEventListener('click', () => {
    renderProvider('spotify');
  });

  youtubeOption.addEventListener('click', () => {
    renderProvider('youtube');
  });

  closePlayerBtn.addEventListener('click', () => {
    if (window.opener) {
      window.close();
      return;
    }

    window.location.href = '/';
  });

  renderProvider(getRequestedProvider());
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initMusicPlayer, { once: true });
} else {
  initMusicPlayer();
}
