// @vitest-environment jsdom

import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import PythonPlayground from '../src/components/PythonPlayground';
import type { PythonPlaygroundProps } from '../src/lib/python-playground';

const { loadPyodideRuntime } = vi.hoisted(() => ({
  loadPyodideRuntime: vi.fn(),
}));

vi.mock('../src/lib/pyodide-loader', () => ({
  loadPyodideRuntime,
}));

const baseProps: PythonPlaygroundProps = {
  title: 'Widget Test',
  initialCode: 'print("hello")',
  samples: [
    {
      label: 'Default Example',
      code: 'print("hello")',
      description: 'Base sample',
    },
  ],
  walkthroughSteps: [
    {
      label: 'Inspect input',
      lineHint: 1,
      variables: {
        value: '1',
      },
      output: 'Starting point',
    },
    {
      label: 'Inspect output',
      lineHint: 2,
      variables: {
        value: '2',
      },
      output: 'Next point',
    },
  ],
  notes: 'Notes',
};

describe('PythonPlayground', () => {
  let container: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT =
      true;
    globalThis.IntersectionObserver = class {
      private callback: IntersectionObserverCallback;
      readonly root = null;
      readonly rootMargin = '0px';
      readonly thresholds = [0];

      constructor(callback: IntersectionObserverCallback) {
        this.callback = callback;
      }

      disconnect() {}

      observe(target: Element) {
        this.callback([{ isIntersecting: true, target } as IntersectionObserverEntry], this);
      }

      unobserve() {}

      takeRecords() {
        return [];
      }
    } as unknown as typeof IntersectionObserver;
    container = document.createElement('div');
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => {
      root.unmount();
    });
    container.remove();
    vi.clearAllMocks();
  });

  async function render(props: PythonPlaygroundProps = baseProps) {
    await act(async () => {
      root.render(<PythonPlayground {...props} />);
    });
    await flushAsyncWork();
  }

  async function flushAsyncWork() {
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
  }

  it('shows a ready message after the Python runtime loads', async () => {
    loadPyodideRuntime.mockResolvedValueOnce({
      runPythonAsync: vi.fn(),
    });

    await render();

    expect(container.textContent).toContain('Python is ready. Edit the code and run it.');
  });

  it('shows an error message when the Python runtime fails to load', async () => {
    loadPyodideRuntime.mockRejectedValueOnce(new Error('Runtime failed to boot'));

    await render();

    expect(container.textContent).toContain('Runtime failed to boot');
  });

  it('runs code and prints stdout and stderr', async () => {
    loadPyodideRuntime.mockResolvedValueOnce({
      runPythonAsync: vi.fn().mockResolvedValue({
        toJs: () => ['hello from python\n', 'warning output\n'],
      }),
    });

    await render();

    const buttons = container.querySelectorAll('button');
    const runButton = Array.from(buttons).find((button) => button.textContent === 'Run');
    expect(runButton).toBeTruthy();

    await act(async () => {
      runButton?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    });
    await flushAsyncWork();

    expect(container.textContent).toContain('hello from python');
    expect(container.textContent).toContain('warning output');
  });

  it('resets the edited code and clears prior output', async () => {
    loadPyodideRuntime.mockResolvedValueOnce({
      runPythonAsync: vi.fn().mockResolvedValue({
        toJs: () => ['hello from python\n', ''],
      }),
    });

    await render();

    const editor = container.querySelector('textarea');
    expect(editor).toBeTruthy();

    await act(async () => {
      editor!.value = 'print("changed")';
      editor!.dispatchEvent(new Event('input', { bubbles: true }));
    });

    const buttons = Array.from(container.querySelectorAll('button'));
    const runButton = buttons.find((button) => button.textContent === 'Run');
    const resetButton = buttons.find((button) => button.textContent === 'Reset');

    await act(async () => {
      runButton?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    });
    await flushAsyncWork();

    expect(container.textContent).toContain('hello from python');

    await act(async () => {
      resetButton?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    });

    expect(editor?.value).toBe('print("hello")');
    expect(container.textContent).not.toContain('hello from python');
  });

  it('moves through the guided walkthrough values', async () => {
    loadPyodideRuntime.mockResolvedValueOnce({
      runPythonAsync: vi.fn(),
    });

    await render();

    expect(container.textContent).toContain('Inspect input');
    expect(container.textContent).toContain('Starting point');
    expect(container.textContent).toContain('value1');

    const buttons = Array.from(container.querySelectorAll('button'));
    const nextButton = buttons.find((button) => button.textContent === 'Next Step');
    expect(nextButton).toBeTruthy();

    await act(async () => {
      nextButton?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    });

    expect(container.textContent).toContain('Inspect output');
    expect(container.textContent).toContain('Next point');
    expect(container.textContent).toContain('value2');
  });
});
