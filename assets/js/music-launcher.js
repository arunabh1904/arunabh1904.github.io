function initMusicLauncher() {
  const musicButton = document.getElementById('music-button');
  if (!musicButton) return;

  musicButton.addEventListener('click', () => {
    window.open('/music.html', 'music-player', 'width=360,height=220');
  });
}

document.addEventListener('DOMContentLoaded', initMusicLauncher);
