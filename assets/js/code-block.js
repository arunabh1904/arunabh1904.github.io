function initCodeCopy() {
  document.querySelectorAll('pre.highlight').forEach(pre => {
    const btn = document.createElement('button');
    btn.className = 'copy-btn';
    btn.type = 'button';
    btn.textContent = 'Copy';
    pre.appendChild(btn);

    btn.addEventListener('click', () => {
      const text = pre.innerText;
      navigator.clipboard.writeText(text).then(() => {
        btn.textContent = 'Copied!';
        setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
      });
    });
  });
}

document.addEventListener('DOMContentLoaded', initCodeCopy);
