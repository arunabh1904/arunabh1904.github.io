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
  prompt: ['Prompt copy', 'Explanation-only detail.'],
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
  solutionNotes: [
    'Use a row-wise max shift before the exponentials:\n`shifted = logits - row_max`',
  ],
  reasoning: [
    {
      axis: 'Tensor reasoning',
      detail: 'Keep the row maximum shaped (N, 1) so it broadcasts across classes.',
    },
    {
      axis: 'Memory / computation tradeoff',
      detail: 'A fused kernel avoids materializing every intermediate tensor.',
    },
  ],
  visual: {
    src: '/assets/images/code-glance-stable-softmax-cross-entropy.svg',
    alt: 'Tensor operation visual',
    caption: 'Shape visual',
  },
  solutionDiagram: '(N, 1) × (1, M) → (N, M)',
  numpyAlternative: {
    code: `import numpy as np

def softmax_cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    return np.asarray(logits)[0, labels[0]]`,
    exampleCode: `logits = np.array([[2.0, 1.0, 0.1]])
labels = np.array([0])
print(softmax_cross_entropy(logits, labels))`,
    memory: ['Keep the row axis when subtracting the maximum.'],
  },
  solutionCode: `def softmax_cross_entropy(logits, labels):
    # Return the reference value after following the stable path.
    return "solution"`,
  starterCode: `def softmax_cross_entropy(logits, labels):
    # TODO: implement the stable path.
    raise NotImplementedError("Implement softmax_cross_entropy")

print("starter")`,
  packages: ['torch', 'numpy'],
  tags: ['PyTorch', 'NumPy'],
  editorStart: 'blank',
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

  it('opens one coherent explanation without repeating the prompt or code', async () => {
    loadPyodideRuntime.mockResolvedValueOnce({
      runPythonAsync: vi.fn(),
    });

    await render();

    const explanationButton = Array.from(container.querySelectorAll('button')).find(
      (button) => button.textContent === 'Explanation',
    );
    await act(async () => {
      explanationButton?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    });

    expect(getEditor().textContent).not.toContain('def softmax_cross_entropy');
    const dialog = container.querySelector('[role="dialog"]');
    expect(dialog).not.toBeNull();
    expect(dialog?.querySelector('h2')?.textContent).toBe('Explanation');
    expect(dialog?.textContent).not.toContain('torch.Tensor');
    expect(dialog?.textContent).not.toContain('np.ndarray');
    expect(dialog?.textContent).toContain('Use a row-wise max shift before the exponentials');
    expect(dialog?.querySelectorAll('.code-practice-lab__standalone-expression')).toHaveLength(1);
    expect(dialog?.querySelector('.code-practice-lab__standalone-expression')?.textContent).toBe(
      'shifted = logits - row_max',
    );
    expect(dialog?.textContent).not.toContain('Keep the row axis when subtracting the maximum.');
    expect(dialog?.textContent).not.toContain('Subtract the row max first.');
    expect(dialog?.textContent).not.toContain('Prompt copy');
    expect(dialog?.textContent).not.toContain('Explanation-only detail.');
    expect(dialog?.textContent).not.toContain('def softmax_cross_entropy');
    expect(dialog?.querySelector('.code-practice-lab__explanation-walkthrough')?.firstElementChild)
      .toBe(dialog?.querySelector('.code-practice-lab__explanation-visual'));
    expect(dialog?.querySelector('pre')).toBeNull();
    expect(dialog?.querySelector('.code-practice-lab__explanation-steps')).toBeNull();
    expect(dialog?.querySelector('.code-practice-lab__explanation-footer')).toBeNull();
  });

  it('switches between clean Torch and NumPy solutions with matching explanations', async () => {
    const runPythonAsync = vi.fn().mockResolvedValue({
      toJs: () => ['solution output\n', ''],
    });
    loadPyodideRuntime.mockResolvedValueOnce({
      runPythonAsync,
    });

    await render();

    expect(container.textContent).toContain('Problem 01');
    expect(container.textContent).toContain('Stable softmax cross-entropy');
    expect(container.textContent).not.toContain('What good looks like');
    expect(container.textContent).not.toContain('Explanation-only detail.');
    expect(container.textContent).not.toContain('Reason through the system');
    expect(container.textContent).not.toContain('Tensor reasoning');
    expect(container.querySelector('.code-practice-lab__visual')).toBeNull();
    expect(container.textContent).not.toContain('Use a row-wise max shift before the exponentials');
    expect(container.textContent).not.toContain('How the NumPy version works');
    expect(container.textContent).not.toContain('Python ready');
    expect(container.querySelector('[role="dialog"]')).toBeNull();
    expect(container.textContent).not.toContain('return "solution"');
    expect(getEditor().textContent).not.toContain('TODO');
    expect(getEditor().textContent).not.toContain('def softmax_cross_entropy');
    expect(getEditor().textContent).not.toContain('print("starter")');

    const buttons = Array.from(container.querySelectorAll('button'));
    const torchButton = buttons.find((button) => button.textContent === 'Torch');
    const numpyButton = buttons.find((button) => button.textContent === 'NumPy');

    expect(buttons.find((button) => button.textContent === 'Add hints')).toBeUndefined();
    expect(buttons.find((button) => button.textContent === 'Need help?')).toBeUndefined();
    expect(torchButton?.closest('.code-practice-lab__solution-picker')).not.toBeNull();
    expect(numpyButton?.closest('.code-practice-lab__solution-picker')).not.toBeNull();

    await act(async () => {
      torchButton?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    });

    expect(getEditor().textContent).toContain('def softmax_cross_entropy');
    expect(getEditor().textContent).toContain('return "solution"');
    expect(getEditor().textContent).toContain('print("starter")');
    expect(getEditor().textContent).not.toContain('# Reference solution');
    expect(getEditor().textContent).not.toContain('# Original placeholder:');
    expect(container.querySelector('.code-practice-lab--reference')).not.toBeNull();
    expect(getEditor().textContent).not.toContain('raise NotImplementedError');
    expect(container.querySelector('.cm-solution-line-toggle')).toBeNull();
    expect(container.querySelector('[role="dialog"]')).toBeNull();

    const explanationButton = Array.from(container.querySelectorAll('button')).find(
      (button) => button.textContent === 'Explanation',
    );
    await act(async () => {
      explanationButton?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    });

    expect(container.querySelector('[role="dialog"]')).not.toBeNull();
    expect(container.querySelector('.code-practice-lab__explanation-visual img')?.getAttribute('src')).toBe(
      '/assets/images/code-glance-stable-softmax-cross-entropy.svg',
    );
    expect(container.querySelector('[role="dialog"] h2')?.textContent).toBe('Explanation');
    expect(container.textContent).toContain('Use a row-wise max shift before the exponentials');
    expect(container.querySelector('[role="dialog"]')?.textContent).not.toContain(
      'Keep the row axis when subtracting the maximum.',
    );
    expect(container.querySelector('.code-practice-lab__solution-diagram')).toBeNull();
    expect(container.textContent).not.toContain('(N, 1) × (1, M) → (N, M)');
    expect(container.textContent).not.toContain('Tensor reasoning');
    expect(container.textContent).toContain('Keep the row maximum shaped (N, 1)');
    expect(container.querySelectorAll('.code-practice-lab__explanation-reasoning p')).toHaveLength(2);
    expect(container.querySelector('[role="dialog"]')?.textContent).not.toContain('Torch reference');
    expect(container.querySelector('[role="dialog"]')?.textContent).not.toContain('NumPy reference');
    expect(container.querySelector('[role="dialog"]')?.textContent).not.toContain(
      'print(softmax_cross_entropy(logits, labels))',
    );

    const closeButton = Array.from(container.querySelectorAll('button')).find(
      (button) => button.textContent === 'Close',
    );
    await act(async () => {
      closeButton?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    });
    expect(container.querySelector('[role="dialog"]')).toBeNull();
    expect(getEditor().textContent).toContain('return "solution"');

    await act(async () => {
      explanationButton?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    });
    expect(container.querySelector('[role="dialog"]')).not.toBeNull();

    const secondCloseButton = Array.from(container.querySelectorAll('button')).find(
      (button) => button.textContent === 'Close',
    );
    await act(async () => {
      secondCloseButton?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    });
    expect(container.querySelector('[role="dialog"]')).toBeNull();

    await act(async () => {
      numpyButton?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    });

    expect(getEditor().textContent).toContain('import numpy as np');
    expect(getEditor().textContent).toContain('logits: np.ndarray');
    expect(getEditor().textContent).toContain('print(softmax_cross_entropy(logits, labels))');
    expect(getEditor().textContent).not.toContain('# Reference solution');
    expect(container.querySelector('[role="dialog"]')).toBeNull();

    await act(async () => {
      explanationButton?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    });

    expect(container.querySelector('[role="dialog"]')).not.toBeNull();
    expect(container.querySelector('[role="dialog"] h2')?.textContent).toBe('Explanation');
    expect(container.textContent).toContain('Use a row-wise max shift before the exponentials');
    expect(container.querySelector('[role="dialog"]')?.textContent).not.toContain(
      'Keep the row axis when subtracting the maximum.',
    );

    const runButton = container.querySelector<HTMLButtonElement>('button[aria-label="Run code"]');
    await act(async () => {
      runButton?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    });
    await flushAsyncWork();

    expect(runPythonAsync).toHaveBeenCalledOnce();
    expect(runPythonAsync.mock.calls[0][0]).toContain('import numpy as np');
    expect(container.textContent).toContain('solution output');
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

  it('starts function-only exercises from an empty Python file', async () => {
    loadPyodideRuntime.mockResolvedValueOnce({
      runPythonAsync: vi.fn(),
    });

    await render();

    const editor = getEditor();
    expect(editor.textContent).not.toContain('def softmax_cross_entropy');
    expect(editor.textContent).not.toContain('print("starter")');
    expect(container.textContent).toContain('Ctrl / Cmd + Enter');
  });

  it('resets a function-only exercise to an empty file', async () => {
    loadPyodideRuntime.mockResolvedValueOnce({
      runPythonAsync: vi.fn(),
    });

    await render();
    const buttons = Array.from(container.querySelectorAll('button'));
    const solutionButton = buttons.find((button) => button.textContent === 'Torch');
    const resetButton = buttons.find((button) => button.textContent === 'Reset');

    await act(async () => {
      solutionButton?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    });
    expect(getEditor().textContent).toContain('def softmax_cross_entropy');

    await act(async () => {
      resetButton?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    });
    expect(getEditor().textContent).not.toContain('def softmax_cross_entropy');
    expect(container.querySelector('.code-practice-lab--reference')).toBeNull();
  });

  it('runs full nn.Module exercises in the same browser workspace', async () => {
    loadPyodideRuntime.mockResolvedValueOnce({
      runPythonAsync: vi.fn(),
    });

    await render({
      ...testProblem,
      id: 'resnet-from-building-blocks',
      title: 'Build a configurable ResNet',
      track: 'architecture',
      numpyAlternative: undefined,
      editorStart: 'scaffold',
      starterCode: `from torch import nn

class ResNet(nn.Module):
    def forward(self, x):
        raise NotImplementedError("Implement forward")`,
      solutionCode: `from torch import nn

class ResNet(nn.Module):
    def forward(self, x):
        return x`,
    });

    expect(loadPyodideRuntime).toHaveBeenCalledOnce();
    expect(container.textContent).not.toContain('Local PyTorch');
    expect(container.querySelector('button[aria-label="Run code"]')).not.toBeNull();
    expect(getEditor().textContent).toContain('class ResNet(nn.Module):');
  });

  it('keeps the default workspace focused until a reference solution is loaded', async () => {
    loadPyodideRuntime.mockResolvedValueOnce({
      runPythonAsync: vi.fn(),
    });

    await render();

    expect(container.querySelector('.code-practice-lab__view-toggle')).toBeNull();
    expect(container.querySelector('.code-practice-lab__editor-layout')).not.toBeNull();
    expect(container.querySelector('[role="dialog"]')).toBeNull();
    expect(container.querySelector('.cm-solution-line-toggle')).toBeNull();
    expect(container.textContent).not.toContain('How it works');
    expect(container.querySelector('.cm-editor')).not.toBeNull();
  });
});
