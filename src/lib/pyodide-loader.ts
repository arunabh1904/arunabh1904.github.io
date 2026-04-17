const PYODIDE_INDEX_URL = 'https://cdn.jsdelivr.net/pyodide/v0.28.3/full/';
const PYODIDE_SCRIPT_SRC = `${PYODIDE_INDEX_URL}pyodide.js`;

export interface PyodideRuntime {
  loadPackage?: (packages: string | string[]) => Promise<void>;
  runPythonAsync(code: string): Promise<unknown>;
}

declare global {
  interface Window {
    loadPyodide?: (options: { indexURL: string }) => Promise<PyodideRuntime>;
  }
}

let scriptPromise: Promise<void> | null = null;
let runtimePromise: Promise<PyodideRuntime> | null = null;

function ensurePyodideScript() {
  if (typeof window === 'undefined') {
    return Promise.reject(new Error('Pyodide can only load in the browser.'));
  }

  if (window.loadPyodide) {
    return Promise.resolve();
  }

  if (!scriptPromise) {
    scriptPromise = new Promise<void>((resolve, reject) => {
      const existing = document.querySelector<HTMLScriptElement>(
        `script[data-pyodide-loader="true"]`,
      );

      if (existing) {
        existing.addEventListener('load', () => resolve(), { once: true });
        existing.addEventListener(
          'error',
          () => reject(new Error('Failed to load the Pyodide script.')),
          { once: true },
        );
        return;
      }

      const script = document.createElement('script');
      script.src = PYODIDE_SCRIPT_SRC;
      script.async = true;
      script.dataset.pyodideLoader = 'true';
      script.onload = () => resolve();
      script.onerror = () => reject(new Error('Failed to load the Pyodide script.'));
      document.head.appendChild(script);
    });
  }

  return scriptPromise;
}

export async function loadPyodideRuntime() {
  if (!runtimePromise) {
    runtimePromise = ensurePyodideScript().then(async () => {
      if (!window.loadPyodide) {
        throw new Error('Pyodide loader was unavailable after script load.');
      }
      return window.loadPyodide({ indexURL: PYODIDE_INDEX_URL });
    });
  }

  return runtimePromise;
}

export function resetPyodideLoaderForTests() {
  scriptPromise = null;
  runtimePromise = null;
}
