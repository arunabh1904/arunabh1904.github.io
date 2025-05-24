---
layout: content
title: Python Playground
---

<p>This page provides a simple browser-based Python interpreter powered by Pyodide.</p>

<div class="py-terminal">
  <textarea id="py-input" placeholder="print('hello, world')"></textarea>
  <button id="py-run" type="button">Run</button>
  <pre id="py-output"></pre>
</div>

<script type="module">
  import { loadPyodide } from 'https://cdn.jsdelivr.net/pyodide/v0.23.2/full/pyodide.mjs';
  async function main() {
    const pyodide = await loadPyodide();
    document.getElementById('py-run').addEventListener('click', async () => {
      const code = document.getElementById('py-input').value;
      let result = '';
      try {
        result = await pyodide.runPythonAsync(code);
      } catch (err) {
        result = err;
      }
      document.getElementById('py-output').textContent = result ?? '';
    });
  }
  main();
</script>

<p>Use <code>uv</code> locally to install additional packages and create a custom environment before bundling it with Pyodide.</p>

