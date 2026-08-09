const MUSIC_CHANNEL_KEY = 'musicPlayerChannel';
const MUSIC_PLAYBACK_KEY = 'musicPlayerPlayback';
const MUSIC_VOLUME_KEY = 'musicPlayerVolume';
const DEFAULT_MUSIC_CHANNEL = 'iranian-jazz';
const DEFAULT_MUSIC_VOLUME = 70;
const closeTimers = new WeakMap();
const tuneTimers = new WeakMap();
const youtubePlayers = new WeakMap();
let youtubeApiPromise;

function clampMusicVolume(value) {
  const parsed = Number.parseInt(String(value), 10);
  if (!Number.isFinite(parsed)) {
    return DEFAULT_MUSIC_VOLUME;
  }

  return Math.min(100, Math.max(0, parsed));
}

function readMusicVolume() {
  try {
    return clampMusicVolume(localStorage.getItem(MUSIC_VOLUME_KEY));
  } catch {
    return DEFAULT_MUSIC_VOLUME;
  }
}

function writeMusicVolume(volume) {
  try {
    localStorage.setItem(MUSIC_VOLUME_KEY, String(volume));
  } catch {
    // The control remains usable when browser storage is unavailable.
  }
}

function readPlaybackState() {
  try {
    return JSON.parse(sessionStorage.getItem(MUSIC_PLAYBACK_KEY) || '{}');
  } catch {
    return {};
  }
}

function writePlaybackState(channelId, nextState) {
  const playback = readPlaybackState();
  playback[channelId] = {
    ...playback[channelId],
    ...nextState,
    updatedAt: Date.now(),
  };
  sessionStorage.setItem(MUSIC_PLAYBACK_KEY, JSON.stringify(playback));
}

function loadYouTubeApi() {
  if (window.YT?.Player) {
    return Promise.resolve(window.YT);
  }

  if (youtubeApiPromise) {
    return youtubeApiPromise;
  }

  youtubeApiPromise = new Promise((resolve) => {
    const previousReady = window.onYouTubeIframeAPIReady;
    window.onYouTubeIframeAPIReady = () => {
      if (typeof previousReady === 'function') {
        previousReady();
      }
      resolve(window.YT);
    };

    if (!document.querySelector('script[src="https://www.youtube.com/iframe_api"]')) {
      const script = document.createElement('script');
      script.src = 'https://www.youtube.com/iframe_api';
      script.async = true;
      document.head.append(script);
    }
  });

  return youtubeApiPromise;
}

function getActiveChannel(player) {
  return getChannelButtons(player).find((button) => button.classList.contains('music-channel--active'));
}

function getVolumeSlider(player) {
  const slider = player.querySelector('[data-music-volume]');
  return slider instanceof HTMLInputElement ? slider : null;
}

function syncVolumeUi(player, volume) {
  const slider = getVolumeSlider(player);
  const control = player.querySelector('[data-music-volume-control]');
  const readout = player.querySelector('[data-music-volume-readout]');

  if (slider) {
    slider.value = String(volume);
    slider.setAttribute('aria-valuetext', `${volume}%`);
  }
  if (control instanceof HTMLElement) {
    control.style.setProperty('--music-volume', `${volume}%`);
  }
  if (readout) {
    readout.textContent = String(volume);
  }
}

function syncVolumeAvailability(player, channel = getActiveChannel(player)) {
  const slider = getVolumeSlider(player);
  const control = player.querySelector('[data-music-volume-control]');
  const isSupported = channel?.dataset.channelKind === 'youtube';

  if (slider) {
    slider.disabled = !isSupported;
    slider.setAttribute(
      'aria-label',
      isSupported ? 'Volume' : 'Volume is controlled in the Spotify player',
    );
  }
  if (control instanceof HTMLElement) {
    control.classList.toggle('retro-tv__volume-control--unavailable', !isSupported);
    control.title = isSupported ? 'Adjust volume' : 'Spotify controls volume in its player';
  }
}

function applyYouTubeVolume(player, volume = readMusicVolume()) {
  const iframe = player.querySelector('[data-music-iframe]');
  const channel = getActiveChannel(player);
  const record = iframe instanceof HTMLIFrameElement ? youtubePlayers.get(iframe) : null;

  if (!record || channel?.dataset.channelKind !== 'youtube') {
    return;
  }

  try {
    record.api.setVolume(volume);
  } catch {
    // The player can briefly be unavailable while an iframe changes channels.
  }
}

function setMusicVolume(player, requestedVolume) {
  const volume = clampMusicVolume(requestedVolume);
  syncVolumeUi(player, volume);
  writeMusicVolume(volume);
  applyYouTubeVolume(player, volume);
}

function persistYouTubePlayback(player) {
  const iframe = player.querySelector('[data-music-iframe]');
  const record = iframe instanceof HTMLIFrameElement ? youtubePlayers.get(iframe) : null;
  const channel = getActiveChannel(player);

  if (!record || channel?.dataset.channelKind !== 'youtube') {
    return;
  }

  try {
    const time = record.api.getCurrentTime();
    const state = record.api.getPlayerState();
    if (Number.isFinite(time)) {
      writePlaybackState(channel.dataset.musicChannel, {
        time,
        playing: state === window.YT.PlayerState.PLAYING || state === window.YT.PlayerState.BUFFERING,
      });
    }
  } catch {
    // The iframe may be between channels or reconnecting during navigation.
  }
}

function startYouTubeProgress(player, iframe, api) {
  const previousRecord = youtubePlayers.get(iframe);
  if (previousRecord?.timer) {
    window.clearInterval(previousRecord.timer);
  }

  const timer = window.setInterval(() => persistYouTubePlayback(player), 1000);
  youtubePlayers.set(iframe, { api, timer });
}

function stopYouTubeProgress(iframe) {
  const record = youtubePlayers.get(iframe);
  if (record?.timer) {
    window.clearInterval(record.timer);
    youtubePlayers.set(iframe, { api: record.api, timer: null });
  }
}

function restoreYouTubePlayback(player, api) {
  const channel = getActiveChannel(player);
  const channelId = channel?.dataset.musicChannel;
  const saved = channelId ? readPlaybackState()[channelId] : null;

  if (!saved || !Number.isFinite(saved.time) || saved.time < 1) {
    return;
  }

  try {
    api.seekTo(saved.time, true);
    if (saved.playing) {
      api.playVideo();
    } else {
      api.pauseVideo();
    }
  } catch {
    // The player will remain usable even if a provider blocks programmatic resume.
  }
}

function connectYouTubePlayer(player) {
  const iframe = player.querySelector('[data-music-iframe]');
  const channel = getActiveChannel(player);

  if (
    !(iframe instanceof HTMLIFrameElement) ||
    channel?.dataset.channelKind !== 'youtube' ||
    !iframe.src.includes('youtube.com/embed/')
  ) {
    return;
  }

  loadYouTubeApi().then((YT) => {
    const existing = youtubePlayers.get(iframe);
    if (existing) {
      applyYouTubeVolume(player);
      restoreYouTubePlayback(player, existing.api);
      startYouTubeProgress(player, iframe, existing.api);
      return;
    }

    const api = new YT.Player(iframe, {
      events: {
        onReady: (event) => {
          startYouTubeProgress(player, iframe, event.target);
          applyYouTubeVolume(player);
          restoreYouTubePlayback(player, event.target);
        },
        onStateChange: (event) => {
          startYouTubeProgress(player, iframe, event.target);
          persistYouTubePlayback(player);
        },
      },
    });
    youtubePlayers.set(iframe, { api, timer: null });
  });
}

function getChannelButtons(player) {
  return Array.from(player.querySelectorAll('[data-music-channel]')).filter(
    (button) => button instanceof HTMLButtonElement,
  );
}

function getChannel(player, requestedId) {
  const channels = getChannelButtons(player);
  return (
    channels.find((button) => button.dataset.musicChannel === requestedId) ||
    channels.find((button) => button.dataset.musicChannel === DEFAULT_MUSIC_CHANNEL) ||
    channels[0]
  );
}

function getRequestedChannel(player) {
  if (player.dataset.musicStandalone === 'true') {
    const requested = new URLSearchParams(window.location.search).get('channel');
    if (requested) {
      return requested;
    }
  }

  return localStorage.getItem(MUSIC_CHANNEL_KEY) || DEFAULT_MUSIC_CHANNEL;
}

function syncStandaloneUrl(player, channelId) {
  if (player.dataset.musicStandalone !== 'true') {
    return;
  }

  const url = new URL(window.location.href);
  url.searchParams.set('channel', channelId);
  window.history.replaceState({}, '', url);
}

function syncChannelDetails(player, channel) {
  const number = player.querySelector('[data-music-channel-number]');
  const name = player.querySelector('[data-music-channel-name]');
  const detail = player.querySelector('[data-music-channel-detail]');

  if (number) number.textContent = `CH ${channel.dataset.channelNumber}`;
  if (name) name.textContent = channel.dataset.channelName || '';
  if (detail) detail.textContent = channel.dataset.channelDetail || '';

  getChannelButtons(player).forEach((button) => {
    const isActive = button === channel;
    button.classList.toggle('music-channel--active', isActive);
    button.setAttribute('aria-pressed', String(isActive));
  });
}

function tuneToChannel(player, requestedId, animate = true) {
  const channel = getChannel(player, requestedId);
  const iframe = player.querySelector('[data-music-iframe]');
  const tv = player.querySelector('[data-music-tv]');

  if (!(channel instanceof HTMLButtonElement) || !(iframe instanceof HTMLIFrameElement)) {
    return;
  }

  const channelId = channel.dataset.musicChannel || DEFAULT_MUSIC_CHANNEL;
  const channelSrc = channel.dataset.channelSrc;
  if (!channelSrc) {
    return;
  }

  const previousTimer = tuneTimers.get(player);
  if (previousTimer) {
    window.clearTimeout(previousTimer);
  }

  syncChannelDetails(player, channel);
  syncVolumeAvailability(player, channel);
  if (tv instanceof HTMLElement) {
    tv.dataset.musicKind = channel.dataset.channelKind || 'youtube';
  }
  localStorage.setItem(MUSIC_CHANNEL_KEY, channelId);
  syncStandaloneUrl(player, channelId);

  if (animate && tv) {
    tv.classList.add('retro-tv--tuning');
  }

  const updateFrame = () => {
    const nextUrl = new URL(channelSrc);
    if (channel.dataset.channelKind === 'youtube') {
      nextUrl.searchParams.set('origin', window.location.origin);
    }

    iframe.title = `${channel.dataset.channelName || 'Music'} player`;
    if (channel.dataset.channelKind !== 'youtube') {
      stopYouTubeProgress(iframe);
    }
    iframe.src = nextUrl.toString();

    const settle = () => {
      tv?.classList.remove('retro-tv--tuning');
      connectYouTubePlayer(player);
    };
    iframe.addEventListener('load', settle, { once: true });
    window.setTimeout(settle, 900);
  };

  if (animate) {
    tuneTimers.set(player, window.setTimeout(updateFrame, 140));
  } else {
    updateFrame();
  }
}

function clearCloseTimer(player) {
  const priorTimer = closeTimers.get(player);
  if (priorTimer) {
    window.clearTimeout(priorTimer);
  }
}

function syncTriggerState(trigger, status) {
  if (!(trigger instanceof HTMLButtonElement)) {
    return;
  }

  const isOpen = status === 'open';
  const isPlaying = status === 'open' || status === 'minimized';
  trigger.setAttribute('aria-expanded', String(isOpen));
  trigger.setAttribute(
    'aria-label',
    isOpen ? 'Minimize music player' : isPlaying ? 'Restore music player' : 'Open music player',
  );
  trigger.classList.toggle('page-controls__button--active', isOpen);
  trigger.classList.toggle('page-controls__button--playing', isPlaying);
}

function showPlayer(player, trigger) {
  clearCloseTimer(player);
  const wasMinimized = player.dataset.musicStatus === 'minimized';

  player.dataset.musicStatus = 'open';
  player.hidden = false;
  player.setAttribute('aria-hidden', 'false');
  syncTriggerState(trigger, 'open');

  window.requestAnimationFrame(() => {
    player.classList.add('music-player--visible');
  });

  if (!wasMinimized) {
    tuneToChannel(player, getRequestedChannel(player), false);
  }
}

function minimizePlayer(player, trigger) {
  clearCloseTimer(player);
  player.dataset.musicStatus = 'minimized';
  player.classList.remove('music-player--visible');
  player.setAttribute('aria-hidden', 'true');
  syncTriggerState(trigger, 'minimized');

  closeTimers.set(
    player,
    window.setTimeout(() => {
      if (player.dataset.musicStatus === 'minimized') {
        player.hidden = true;
      }
    }, 240),
  );
}

function closePlayer(player, trigger) {
  clearCloseTimer(player);
  player.dataset.musicStatus = 'closed';
  player.classList.remove('music-player--visible');
  player.setAttribute('aria-hidden', 'true');
  syncTriggerState(trigger, 'closed');

  closeTimers.set(
    player,
    window.setTimeout(() => {
      player.hidden = true;
      const iframe = player.querySelector('[data-music-iframe]');
      if (iframe instanceof HTMLIFrameElement) {
        persistYouTubePlayback(player);
        stopYouTubeProgress(iframe);
        iframe.src = 'about:blank';
      }
    }, 240),
  );
}

function initMusicPlayer(player) {
  if (!(player instanceof HTMLElement) || player.dataset.musicReady === 'true') {
    return;
  }

  player.dataset.musicReady = 'true';
  const controller = player.closest('[data-music-controller]');
  const trigger = controller?.querySelector('[data-music-open]');
  const minimizeButton = player.querySelector('[data-music-minimize]');
  const closeButton = player.querySelector('[data-music-close]');
  const isStandalone = player.dataset.musicStandalone === 'true';
  const iframe = player.querySelector('[data-music-iframe]');
  const volumeSlider = getVolumeSlider(player);

  syncVolumeUi(player, readMusicVolume());
  syncVolumeAvailability(player);

  if (trigger instanceof HTMLButtonElement) {
    trigger.addEventListener('click', (event) => {
      event.stopPropagation();
      if (player.dataset.musicStatus === 'open') {
        minimizePlayer(player, trigger);
      } else {
        showPlayer(player, trigger);
      }
    });
  }

  if (minimizeButton instanceof HTMLButtonElement) {
    minimizeButton.addEventListener('click', () => {
      minimizePlayer(player, trigger instanceof HTMLButtonElement ? trigger : null);
      trigger?.focus();
    });
  }

  if (closeButton instanceof HTMLButtonElement) {
    closeButton.addEventListener('click', () => {
      if (isStandalone) {
        if (window.opener) {
          window.close();
        } else {
          window.location.href = '/';
        }
        return;
      }

      closePlayer(player, trigger instanceof HTMLButtonElement ? trigger : null);
      trigger?.focus();
    });
  }

  getChannelButtons(player).forEach((button) => {
    button.addEventListener('click', () => {
      persistYouTubePlayback(player);
      tuneToChannel(player, button.dataset.musicChannel);
    });
  });

  if (volumeSlider) {
    volumeSlider.addEventListener('input', () => {
      setMusicVolume(player, volumeSlider.value);
    });
  }

  if (iframe instanceof HTMLIFrameElement) {
    iframe.addEventListener('load', () => connectYouTubePlayer(player));
  }

  if (isStandalone) {
    tuneToChannel(player, getRequestedChannel(player), false);
  }
}

function initMusicPlayers() {
  document.querySelectorAll('[data-music-player]').forEach(initMusicPlayer);
}

function closeMusicPlayersWithEscape(event) {
  if (event.key !== 'Escape') {
    return;
  }

  document.querySelectorAll('[data-music-player]:not([hidden])').forEach((player) => {
    if (!(player instanceof HTMLElement) || player.dataset.musicStandalone === 'true') {
      return;
    }

    const trigger = player.closest('[data-music-controller]')?.querySelector('[data-music-open]');
    minimizePlayer(player, trigger instanceof HTMLButtonElement ? trigger : null);
    trigger?.focus();
  });
}

if (!window.__musicPlayerNavigationReady) {
  window.__musicPlayerNavigationReady = true;
  document.addEventListener('keydown', closeMusicPlayersWithEscape);
  document.addEventListener('astro:before-swap', () => {
    document.querySelectorAll('[data-music-player]').forEach((player) => {
      if (player instanceof HTMLElement) {
        persistYouTubePlayback(player);
      }
    });
  });
  document.addEventListener('astro:page-load', initMusicPlayers);
}

if (document.readyState !== 'loading') {
  initMusicPlayers();
} else {
  document.addEventListener('DOMContentLoaded', initMusicPlayers, { once: true });
}
