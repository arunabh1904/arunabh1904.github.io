function initMusicLauncher() {
  const musicButton = document.getElementById('music-button');
  const musicBar = document.getElementById('music-bar');
  if (!musicButton) return;

  function startTimer() {
    clearTimeout(musicButton.hideTimeout);
    clearTimeout(musicButton.vibeTimeout);

    musicButton.style.display = 'block';
    if (musicBar) musicBar.style.display = 'none';

    musicButton.vibeTimeout = setTimeout(() => {
      musicButton.classList.add('vibe');
    }, 10000);

    musicButton.hideTimeout = setTimeout(() => {
      musicButton.classList.remove('vibe');
      musicButton.style.display = 'none';
      if (musicBar) musicBar.style.display = 'block';
    }, 15000);
  }

  musicButton.addEventListener('click', () => {
    window.open('/music.html', 'music-player', 'width=360,height=220');
  });

  if (musicBar) {
    musicBar.addEventListener('click', startTimer);
  }

  startTimer();
}

document.addEventListener('DOMContentLoaded', initMusicLauncher);
