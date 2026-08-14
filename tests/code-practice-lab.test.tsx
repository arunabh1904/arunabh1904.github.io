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
  solutionCode: `def softmax_cross_entropy(logits, labels):
    # Return the reference value after following the stable path.
    return "solution"`,
  starterCode: `def softmax_cross_entropy(logits, labels):
    # TODO: implement the stable path.
    raise NotImplementedError("Implement softmax_cross_entropy")

print("starter")`,
  packages: ['torch', 'numpy'],
  tags: ['PyTorch', 'NumPy'],
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

  it('keeps a direct solution action in the workspace and applies it to the editor', async () => {
    loadPyodideRuntime.mockResolvedValueOnce({
      runPythonAsync: vi.fn(),
    });

    await render();

    expect(container.textContent).toContain('Problem 01');
    expect(container.textContent).toContain('Stable softmax cross-entropy');
    expect(container.textContent).not.toContain('Use a row-wise max shift before the exponentials.');
    expect(container.textContent).not.toContain('return "solution"');
    expect(getEditor().textContent).not.toContain('TODO');

    const buttons = Array.from(container.querySelectorAll('button'));
    const solutionButton = buttons.find((button) => button.textContent === 'Solution');

    expect(buttons.find((button) => button.textContent === 'Add hints')).toBeUndefined();
    expect(buttons.find((button) => button.textContent === 'Need help?')).toBeUndefined();
    expect(solutionButton?.closest('.code-practice-lab__workspace')).not.toBeNull();

    await act(async () => {
      solutionButton?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    });

    expect(getEditor().textContent).toContain('print("starter")');
    expect(getEditor().textContent).toContain('# Original placeholder: raise NotImplementedError');
    expect(getEditor().textContent).toContain('# Reference solution');
    expect(getEditor().textContent).toContain('return "solution"');
    expect(container.querySelector('.code-practice-lab--reference')).not.toBeNull();
    expect(getEditor().textContent).not.toMatch(
      /^\s+raise NotImplementedError\("Implement softmax_cross_entropy"\)$/m,
    );
    expect(container.querySelectorAll('.cm-solution-line-toggle').length).toBeGreaterThan(0);

    await act(async () => {
      container.querySelector<HTMLButtonElement>('.cm-solution-line-toggle')?.click();
    });

    expect(container.querySelector('.code-practice-lab__walkthrough-panel')).not.toBeNull();
    expect(container.textContent).toContain('Return the reference value after following the stable path.');
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

    const runButton = container.querySelector<HTMLButtonElement>('button[aria-label="Run code"]');

    expect(runButton?.closest('.code-practice-lab__workspace-header')).not.toBeNull();

    await act(async () => {
      runButton?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    });
    await flushAsyncWork();

    expect(loadPackage).toHaveBeenCalledWith(['numpy']);
    expect(container.textContent).toContain('0.41703');
    expect(container.querySelector('.code-practice-lab__output')).not.toBeNull();
  });

  it('runs the current editor contents with Ctrl+Enter', async () => {
    const runPythonAsync = vi.fn().mockResolvedValue({
      toJs: () => ['keyboard run\n', ''],
    });
    loadPyodideRuntime.mockResolvedValueOnce({
      runPythonAsync,
    });

    await render();

    const editorContent = container.querySelector<HTMLElement>('.cm-content');
    expect(editorContent).not.toBeNull();

    const shortcutEvent = new KeyboardEvent('keydown', {
      key: 'Enter',
      code: 'Enter',
      ctrlKey: true,
      bubbles: true,
      cancelable: true,
    });
    Object.defineProperties(shortcutEvent, {
      keyCode: { value: 13 },
      which: { value: 13 },
    });

    await act(async () => {
      editorContent?.dispatchEvent(shortcutEvent);
    });
    await flushAsyncWork();

    expect(shortcutEvent.defaultPrevented).toBe(true);
    expect(runPythonAsync).toHaveBeenCalledOnce();
    expect(container.textContent).toContain('keyboard run');
  });

  it('renders a CodeMirror editor with the starter code', async () => {
    loadPyodideRuntime.mockResolvedValueOnce({
      runPythonAsync: vi.fn(),
    });

    await render();

    const editor = getEditor();
    expect(editor.textContent).toContain('print("starter")');
    expect(container.textContent).toContain('Ctrl / Cmd + Enter');
  });

  it('keeps the default workspace focused until a reference solution is loaded', async () => {
    loadPyodideRuntime.mockResolvedValueOnce({
      runPythonAsync: vi.fn(),
    });

    await render();

    expect(container.querySelector('.code-practice-lab__view-toggle')).toBeNull();
    expect(container.querySelector('.code-practice-lab__editor-layout--single')).not.toBeNull();
    expect(container.querySelector('.code-practice-lab__annotations')).toBeNull();
    expect(container.querySelector('.cm-solution-line-toggle')).toBeNull();
    expect(container.querySelector('.code-practice-lab__walkthrough-panel')).toBeNull();
    expect(container.textContent).not.toContain('What this solution line is doing');
    expect(container.querySelector('.cm-editor')).not.toBeNull();
  });
});
