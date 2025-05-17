const SPOTIFY_SRC = 'https://open.spotify.com/embed/playlist/4RHYceSp9R1bHyL0dDqTuQ?utm_source=generator&theme=0';
const YOUTUBE_SRC = 'https://www.youtube.com/embed/videoseries?list=PLFgquLnL59alCl_2TQvOiD5Vgm1hCaGSI';
const MUSIC_SRC_KEY = 'musicPlayerSrc';
const MUSIC_MINIMIZED_KEY = 'musicPlayerMinimized';

function initMusicPlayer() {
  const spotifyOption = document.getElementById('choose-spotify');
  const youtubeOption = document.getElementById('choose-youtube');
  const togglePlayerBtn = document.getElementById('toggle-music-player');
  const closePlayerBtn = document.getElementById('close-music-player');
  const musicIframe = document.getElementById('music-iframe');

  const savedSrc = localStorage.getItem(MUSIC_SRC_KEY);
  musicIframe.src = savedSrc || SPOTIFY_SRC;

  if (localStorage.getItem(MUSIC_MINIMIZED_KEY) === 'true') {
    musicIframe.style.display = 'none';
    spotifyOption.style.display = 'none';
    youtubeOption.style.display = 'none';
    togglePlayerBtn.textContent = 'Expand';
  }

  spotifyOption.addEventListener('click', () => {
    musicIframe.src = SPOTIFY_SRC;
    localStorage.setItem(MUSIC_SRC_KEY, SPOTIFY_SRC);
  });

  youtubeOption.addEventListener('click', () => {
    musicIframe.src = YOUTUBE_SRC;
    localStorage.setItem(MUSIC_SRC_KEY, YOUTUBE_SRC);
  });

  togglePlayerBtn.addEventListener('click', () => {
    if (musicIframe.style.display === 'none') {
      musicIframe.style.display = 'block';
      spotifyOption.style.display = 'inline-block';
      youtubeOption.style.display = 'inline-block';
      togglePlayerBtn.textContent = 'Minimize';
      localStorage.setItem(MUSIC_MINIMIZED_KEY, 'false');
    } else {
      musicIframe.style.display = 'none';
      spotifyOption.style.display = 'none';
      youtubeOption.style.display = 'none';
      togglePlayerBtn.textContent = 'Expand';
      localStorage.setItem(MUSIC_MINIMIZED_KEY, 'true');
    }
  });

  closePlayerBtn.addEventListener('click', () => {
    if (window.parent && window.parent !== window) {
      window.parent.postMessage('close-music-overlay', '*');
    } else {
      window.close();
    }
  });
}

document.addEventListener('DOMContentLoaded', initMusicPlayer);
