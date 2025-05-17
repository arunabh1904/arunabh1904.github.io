const MUSIC_SRC_KEY = 'musicPlayerSrc';
const MUSIC_MINIMIZED_KEY = 'musicPlayerMinimized';

function initMusicPlayer() {
  const spotifyOption = document.getElementById('choose-spotify');
  const youtubeOption = document.getElementById('choose-youtube');
  const togglePlayerBtn = document.getElementById('toggle-music-player');
  const closePlayerBtn = document.getElementById('close-music-player');
  const musicIframe = document.getElementById('music-iframe');

  const savedSrc = localStorage.getItem(MUSIC_SRC_KEY);
  if (savedSrc) {
    musicIframe.src = savedSrc;
  }

  if (localStorage.getItem(MUSIC_MINIMIZED_KEY) === 'true') {
    musicIframe.style.display = 'none';
    spotifyOption.style.display = 'none';
    youtubeOption.style.display = 'none';
    togglePlayerBtn.textContent = 'Expand';
  }

  spotifyOption.addEventListener('click', () => {
    musicIframe.src = 'https://open.spotify.com/embed/playlist/4RHYceSp9R1bHyL0dDqTuQ?utm_source=generator&theme=0';
    localStorage.setItem(MUSIC_SRC_KEY, musicIframe.src);
  });

  youtubeOption.addEventListener('click', () => {
    musicIframe.src = 'https://www.youtube.com/embed/kGuGH_UvvxA';
    localStorage.setItem(MUSIC_SRC_KEY, musicIframe.src);
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
    window.close();
  });
}

document.addEventListener('DOMContentLoaded', initMusicPlayer);
