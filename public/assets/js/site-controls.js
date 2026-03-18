const STORAGE_KEY = 'theme';

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem(STORAGE_KEY, theme);
}

function initSiteControls() {
  const html = document.documentElement;

  document.querySelectorAll('[data-theme-toggle]').forEach((button) => {
    button.addEventListener('click', () => {
      const nextTheme = html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      applyTheme(nextTheme);
    });
  });

  document.querySelectorAll('[data-home-button]').forEach((button) => {
    button.addEventListener('click', () => {
      window.location.href = '/';
    });
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initSiteControls, { once: true });
} else {
  initSiteControls();
}
