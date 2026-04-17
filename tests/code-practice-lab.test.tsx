// @vitest-environment jsdom

import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import CodePracticeLab from '../src/components/CodePracticeLab';
import type { CodePracticeProblem } from '../src/lib/code-practice';

const { loadPyodideRuntime } = vi.hoisted(() => ({
  loadPyodideRuntime: vi.fn(),
}));

vi.mock('../src/lib/pyodide-loader', () => ({
  loadPyodideRuntime,
}));

const testProblem: CodePracticeProblem = {
  id: 'stable-softmax-cross-entropy',
  order: 1,
  title: 'Stable softmax cross-entropy',
  difficulty: 'Medium',
  summary: 'Implement a stable softmax loss.',
  prompt: ['Prompt copy'],
  signature: 'def softmax_cross_entropy(logits, labels):\n    ...',
  requirements: ['Do the thing'],
  examples: [
    {
      label: 'Example',
      lines: ['logits = [[2.0, 1.0, 0.1]]', 'labels = [0]'],
      result: 'loss ~= 0.41703',
    },
  ],
  hint: ['Subtract the row max first.'],
  solutionNotes: ['Use a row-wise max shift before the exponentials.'],
  solutionCode: 'print("solution")',
  starterCode: 'print("starter")',
  packages: ['numpy'],
  tags: ['NumPy'],
};

describe('CodePracticeLab', () => {
  let container: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT =
      true;
    globalThis.ResizeObserver = class {
      observe() {}
      unobserve() {}
      disconnect() {}
    };
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

  async function flushAsyncWork() {
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
  }

  async function render(problem: CodePracticeProblem = testProblem) {
    await act(async () => {
      root.render(<CodePracticeLab problem={problem} />);
    });
    await flushAsyncWork();
  }

  function getEditor() {
    const editor = container.querySelector('.cm-editor');
    expect(editor).not.toBeNull();
    return editor as HTMLElement;
  }

  it('reveals the hint and solution only after the user clicks', async () => {
    loadPyodideRuntime.mockResolvedValueOnce({
      runPythonAsync: vi.fn(),
    });

    await render();

    expect(container.textContent).toContain('Problem 01');
    expect(container.textContent).toContain('Stable softmax cross-entropy');
    expect(container.textContent).not.toContain('Subtract the row max first.');
    expect(container.textContent).not.toContain('print("solution")');

    const buttons = Array.from(container.querySelectorAll('button'));
    const hintButton = buttons.find((button) => button.textContent === 'Hint');
    const solutionButton = buttons.find((button) => button.textContent === 'Solution');

    await act(async () => {
      hintButton?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
      solutionButton?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    });

    expect(container.textContent).toContain('Subtract the row max first.');
    expect(container.textContent).toContain('print("solution")');
  });

  it('loads required packages and prints run output', async () => {
    const loadPackage = vi.fn().mockResolvedValue(undefined);
    loadPyodideRuntime.mockResolvedValueOnce({
      loadPackage,
      runPythonAsync: vi.fn().mockResolvedValue({
        toJs: () => ['0.41703\n', ''],
      }),
    });

    await render();

    const buttons = Array.from(container.querySelectorAll('button'));
    const runButton = buttons.find((button) => button.textContent === 'Run code');

    await act(async () => {
      runButton?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    });
    await flushAsyncWork();

    expect(loadPackage).toHaveBeenCalledWith(['numpy']);
    expect(container.textContent).toContain('0.41703');
  });

  it('renders a CodeMirror editor with the starter code', async () => {
    loadPyodideRuntime.mockResolvedValueOnce({
      runPythonAsync: vi.fn(),
    });

    await render();

    const editor = getEditor();
    expect(editor.textContent).toContain('print("starter")');
    expect(container.textContent).toContain('Cmd/Ctrl + /');
  });
});
