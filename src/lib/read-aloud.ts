const formatTime = (seconds: number): string => {
  if (!Number.isFinite(seconds)) return '--:--';
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.floor(seconds % 60).toString().padStart(2, '0');
  return `${minutes}:${remainingSeconds}`;
};

export const initReadAloud = (scope: ParentNode = document): void => {
  scope.querySelectorAll<HTMLElement>('[data-read-aloud]').forEach((root) => {
    if (root.dataset.readAloudReady === 'true') return;

    const audio = root.querySelector<HTMLAudioElement>('[data-read-aloud-audio]');
    const toggle = root.querySelector<HTMLButtonElement>('[data-read-aloud-toggle]');
    const label = root.querySelector<HTMLElement>('[data-read-aloud-label]');
    const icon = root.querySelector<HTMLElement>('[data-read-aloud-icon]');
    const progress = root.querySelector<HTMLInputElement>('[data-read-aloud-progress]');
    const current = root.querySelector<HTMLElement>('[data-read-aloud-current]');
    const duration = root.querySelector<HTMLElement>('[data-read-aloud-duration]');
    const speed = root.querySelector<HTMLSelectElement>('[data-read-aloud-speed]');
    if (!audio || !toggle || !label || !icon || !progress || !current || !duration || !speed) return;

    root.dataset.readAloudReady = 'true';
    root.classList.add('read-aloud--enhanced');

    const renderState = () => {
      const playing = !audio.paused && !audio.ended;
      icon.textContent = playing ? 'Ⅱ' : '▶';
      label.textContent = playing ? 'Pause' : audio.ended ? 'Listen again' : 'Listen';
      toggle.setAttribute('aria-label', playing ? 'Pause audio' : 'Listen to this post');
      toggle.setAttribute('aria-pressed', String(playing));
      root.classList.toggle('read-aloud--playing', playing);
    };

    const renderMetadata = () => {
      duration.textContent = formatTime(audio.duration);
      progress.max = String(audio.duration);
      progress.disabled = false;
    };

    toggle.addEventListener('click', () => {
      if (audio.paused || audio.ended) {
        if (audio.ended) audio.currentTime = 0;
        // Astro's client router swaps the element into the page without always
        // selecting its source. load() makes the first click work after an
        // in-site navigation instead of requiring a full page reload.
        if (!audio.currentSrc) audio.load();
        void audio.play().catch(() => {
          label.textContent = 'Press again to play';
        });
      } else {
        audio.pause();
      }
    });
    audio.addEventListener('play', renderState);
    audio.addEventListener('pause', renderState);
    audio.addEventListener('ended', renderState);
    audio.addEventListener('loadedmetadata', renderMetadata);
    audio.addEventListener('timeupdate', () => {
      current.textContent = formatTime(audio.currentTime);
      progress.value = String(audio.currentTime);
    });
    progress.addEventListener('input', () => {
      audio.currentTime = Number(progress.value);
    });
    speed.addEventListener('change', () => {
      audio.playbackRate = Number(speed.value);
    });

    progress.disabled = true;
    renderState();
    if (audio.readyState >= audio.HAVE_METADATA) renderMetadata();

    // A swapped-in media element can have a src attribute while currentSrc is
    // still empty. Start metadata loading as part of initialization as well as
    // guarding the user-initiated play path above.
    if (!audio.currentSrc) audio.load();
  });
};
