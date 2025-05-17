function initMusicLauncher() {
  const musicButton = document.getElementById('music-button');
  if (!musicButton) return;
  musicButton.addEventListener('click', () => {
    const playerWindow = window.open('/music.html', 'MusicPlayer', 'width=320,height=200');
    if (playerWindow) {
      playerWindow.focus();
    }
  });
}

document.addEventListener('DOMContentLoaded', initMusicLauncher);
