import { describe, expect, it, vi } from 'vitest';
import { runPythonSnippet } from '../src/lib/python-runner';

describe('runPythonSnippet', () => {
  it('loads NumPy for the browser PyTorch compatibility layer without requesting a torch wheel', async () => {
    const loadPackage = vi.fn().mockResolvedValue(undefined);
    const runPythonAsync = vi.fn().mockResolvedValue({
      toJs: () => ['ready\n', ''],
    });

    const result = await runPythonSnippet(
      { loadPackage, runPythonAsync },
      'import torch\nprint(torch.Tensor)',
      ['torch'],
    );

    expect(loadPackage).toHaveBeenCalledWith(['numpy']);
    expect(runPythonAsync).toHaveBeenCalledTimes(1);
    expect(runPythonAsync.mock.calls[0][0]).toContain("torch = _types.ModuleType('torch')");
    expect(result).toEqual({ stdout: 'ready\n', stderr: '' });
  });

  it('keeps direct NumPy snippets available without injecting the torch shim', async () => {
    const loadPackage = vi.fn().mockResolvedValue(undefined);
    const runPythonAsync = vi.fn().mockResolvedValue({
      toJs: () => ['[1. 1.]\n', ''],
    });

    const result = await runPythonSnippet(
      { loadPackage, runPythonAsync },
      'import numpy as np\nprint(np.ones(2))',
      ['numpy'],
    );

    expect(loadPackage).toHaveBeenCalledWith(['numpy']);
    expect(runPythonAsync.mock.calls[0][0]).not.toContain("torch = _types.ModuleType('torch')");
    expect(result).toEqual({ stdout: '[1. 1.]\n', stderr: '' });
  });
});
