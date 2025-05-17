function initMusicLauncher() {
  const musicButton = document.getElementById('music-button');
  if (!musicButton) return;

  function showOverlay() {
    let overlay = document.getElementById('music-overlay');
    if (!overlay) {
      overlay = document.createElement('div');
      overlay.id = 'music-overlay';
      overlay.innerHTML =
        '<iframe src="/music.html" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"></iframe>';
      document.body.appendChild(overlay);
    } else {
      overlay.style.display = 'block';
    }
  }

  musicButton.addEventListener('click', showOverlay);

  window.addEventListener('message', (event) => {
    if (event.data === 'close-music-overlay') {
      const overlay = document.getElementById('music-overlay');
      if (overlay) {
        overlay.style.display = 'none';
      }
    }
  });
}

document.addEventListener('DOMContentLoaded', initMusicLauncher);
