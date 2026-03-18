function initMusicLauncher() {
  const musicButton = document.getElementById('music-button');
  const musicBar = document.getElementById('music-bar');
  if (!musicButton) return;

  const HIDDEN_KEY = 'musicButtonHidden';
  const SEEN_KEY = 'musicButtonSeen';

  function startTimer(isFirst) {
    clearTimeout(musicButton.hideTimeout);
    clearTimeout(musicButton.vibeTimeout);

    localStorage.setItem(HIDDEN_KEY, 'false');

    musicButton.style.display = 'block';
    if (musicBar) musicBar.style.display = 'none';

    musicButton.vibeTimeout = setTimeout(() => {
      musicButton.classList.add('vibe');
    }, 10000);

    const hideDelay = isFirst ? 60000 : 15000;

    musicButton.hideTimeout = setTimeout(() => {
      musicButton.classList.remove('vibe');
      musicButton.style.display = 'none';
      if (musicBar) musicBar.style.display = 'block';
      localStorage.setItem(HIDDEN_KEY, 'true');
    }, hideDelay);
  }

  musicButton.addEventListener('click', () => {
    window.open('/music.html', 'music-player', 'width=360,height=220');
  });

  if (musicBar) {
    musicBar.addEventListener('click', () => startTimer(false));
  }

  const isHidden = localStorage.getItem(HIDDEN_KEY) === 'true';
  const isFirst = !localStorage.getItem(SEEN_KEY);

  if (isHidden) {
    musicButton.style.display = 'none';
    if (musicBar) musicBar.style.display = 'block';
  } else {
    startTimer(isFirst);
  }

  if (isFirst) localStorage.setItem(SEEN_KEY, 'true');
}

document.addEventListener('DOMContentLoaded', initMusicLauncher);
