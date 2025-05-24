import { loadPyodide } from 'https://cdn.jsdelivr.net/pyodide/v0.23.2/full/pyodide.mjs';

async function initPythonRunner() {
  const pyodide = await loadPyodide();
  document.querySelectorAll('pre code.language-python').forEach(code => {
    const pre = code.parentElement;
    const btn = document.createElement('button');
    btn.className = 'run-btn';
    btn.type = 'button';
    btn.textContent = 'Run';
    pre.appendChild(btn);

    const output = document.createElement('pre');
    output.className = 'run-output';
    pre.insertAdjacentElement('afterend', output);

    btn.addEventListener('click', async () => {
      output.textContent = '';
      try {
        const result = await pyodide.runPythonAsync(code.textContent);
        output.textContent = result ?? '';
      } catch (err) {
        output.textContent = err;
      }
    });
  });
}

document.addEventListener('DOMContentLoaded', initPythonRunner);
