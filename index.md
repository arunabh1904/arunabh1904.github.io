---
layout: default
---

<button class="toggle-button" id="theme-toggle">Toggle Dark Mode</button>

<section id="welcome">
  <h1>Welcome to My Blog</h1>
  <p>Discover stories, thoughts, and photos from my journey.</p>
</section>

<section class="gallery">
  <img src="/images/picture1.jpg" alt="Picture 1">
  <img src="/images/picture2.jpg" alt="Picture 2">
  <img src="/images/picture3.jpg" alt="Picture 3">
  <!-- Add more images as needed -->
</section>

<script>
  const toggleButton = document.getElementById('theme-toggle');
  toggleButton.addEventListener('click', () => {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    document.documentElement.setAttribute('data-theme', currentTheme === 'dark' ? 'light' : 'dark');
  });
</script>
