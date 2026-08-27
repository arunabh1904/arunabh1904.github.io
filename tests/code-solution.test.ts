import { execFileSync } from 'node:child_process';
import { describe, expect, it } from 'vitest';
import { codePracticeProblems } from '../src/lib/code-practice';
import { augmentCodeWithSolution } from '../src/lib/code-solution';

function compilePython(code: string) {
  execFileSync(
    'python3',
    ['-c', 'import sys; compile(sys.stdin.read(), "solution.py", "exec")'],
    { encoding: 'utf8', input: code },
  );
}

function getPlaceholderLines(code: string) {
  return code
    .split('\n')
    .filter((line) => /^\s*raise NotImplementedError\("Implement [A-Za-z_]\w*"\)$/.test(line));
}

describe('augmentCodeWithSolution', () => {
  it('keeps every starter scaffold and inserts annotated reference bodies in place', () => {
    for (const problem of codePracticeProblems) {
      const placeholders = getPlaceholderLines(problem.starterCode);
      const lastPlaceholder = problem.starterCode.lastIndexOf(placeholders.at(-1) ?? '');
      const trailingScaffold = problem.starterCode.slice(
        problem.starterCode.indexOf('\n', lastPlaceholder) + 1,
      );
      const annotatedCode = augmentCodeWithSolution(problem);

      expect(placeholders, problem.id).not.toHaveLength(0);
      for (const starterLine of problem.starterCode.split('\n')) {
        if (starterLine.trim() && !placeholders.includes(starterLine)) {
          expect(annotatedCode, problem.id).toContain(starterLine);
        }
      }
      expect(annotatedCode, problem.id).toContain(trailingScaffold);
      expect(annotatedCode, problem.id).toContain('# Reference solution');

      for (const placeholder of placeholders) {
        expect(annotatedCode, problem.id).toContain(
          `# Original placeholder: ${placeholder.trim()}`,
        );
      }

      expect(annotatedCode, problem.id).not.toMatch(
        /^\s+raise NotImplementedError\("Implement [A-Za-z_]\w*"\)$/m,
      );
      expect(() => compilePython(annotatedCode), problem.id).not.toThrow();
      expect(augmentCodeWithSolution(problem, annotatedCode), problem.id).toBe(annotatedCode);
    }
  }, 15_000);

  it('keeps the reference path focused on the operation being taught', () => {
    const softmaxProblem = codePracticeProblems.find(
      (problem) => problem.id === 'stable-softmax-cross-entropy',
    );
    const temperatureProblem = codePracticeProblems.find(
      (problem) => problem.id === 'temperature-scaling-of-logits',
    );

    expect(softmaxProblem).toBeDefined();
    expect(temperatureProblem).toBeDefined();

    const softmaxCode = augmentCodeWithSolution(softmaxProblem!);
    const temperatureCode = augmentCodeWithSolution(temperatureProblem!);

    expect(softmaxCode).not.toContain('def _validate_classification_inputs');
    expect(softmaxCode).toContain('torch.amax(logits, dim=1, keepdim=True)');
    expect(temperatureCode).not.toContain('raise ValueError');
  });

  it('inserts class helper methods required by attention forward passes', () => {
    const expectedHelpers = new Map([
      ['scaled-dot-product-self-attention', ['_split_heads']],
      ['cross-attention', ['split_heads']],
      ['grouped-query-and-multi-query-attention', ['_split', '_repeat_kv']],
    ]);

    for (const [id, helpers] of expectedHelpers) {
      const problem = codePracticeProblems.find((candidate) => candidate.id === id)!;
      const annotatedCode = augmentCodeWithSolution(problem);

      for (const helper of helpers) {
        expect(annotatedCode, id).toContain(`def ${helper}(`);
        expect(annotatedCode, id).not.toMatch(
          new RegExp(`^\\s+raise NotImplementedError\\("Implement ${helper}"\\)$`, 'm'),
        );
      }
    }
  });

  it('keeps Cross-Attention helper parameters aligned after solution insertion', () => {
    const problem = codePracticeProblems.find((candidate) => candidate.id === 'cross-attention')!;
    const annotatedCode = augmentCodeWithSolution(problem);

    expect(annotatedCode).toContain(
      'def split_heads(self, x: torch.Tensor) -> torch.Tensor:',
    );
    expect(annotatedCode).toContain('batch_size, length, _ = x.shape');
    expect(annotatedCode).not.toContain('def forward(self, seq_a, seq_b, mask');
  });

  it('keeps reference snippets short and leaves validation in the prompt', () => {
    const problem = codePracticeProblems.find(
      (candidate) => candidate.id === 'stable-softmax-cross-entropy',
    );
    const nmsProblem = codePracticeProblems.find(
      (candidate) => candidate.id === 'non-maximum-suppression',
    );
    const annotatedCode = augmentCodeWithSolution(problem!);
    const nmsCode = augmentCodeWithSolution(nmsProblem!);
    expect(problem!.solutionCode.split('\n').length).toBeLessThanOrEqual(45);
    expect(nmsProblem!.solutionCode.split('\n').length).toBeLessThanOrEqual(45);
    expect(annotatedCode).not.toContain('raise ValueError');
    expect(annotatedCode).toContain('return torch.mean(torch.log(normalizers)');
    expect(nmsCode).toContain('while order:');
  });

  it('augments the current editor contents without discarding an inserted hint', () => {
    const problem = codePracticeProblems.find(
      (candidate) => candidate.id === 'stable-softmax-cross-entropy',
    );
    const hintedStarter = `# Hint: subtract the row maximum before exponentiating.\n\n${problem!.starterCode}`;

    const annotatedCode = augmentCodeWithSolution(problem!, hintedStarter);

    expect(annotatedCode).toContain(
      '# Hint: subtract the row maximum before exponentiating.',
    );
    expect(annotatedCode).toContain('# Reference solution');
    expect(annotatedCode).toContain('return torch.mean(torch.log(normalizers)');
  });

  it('preserves the full KV-cache layout invariant in the loaded solution', () => {
    const problem = codePracticeProblems.find(
      (candidate) => candidate.id === 'incremental-kv-cache',
    );
    const annotatedCode = augmentCodeWithSolution(problem!);

    expect(annotatedCode).toContain(
      'if cached_layout != update_layout: raise ValueError("cache layout changed")',
    );
    expect(annotatedCode).not.toContain('assert any(self.key.shape[axis] == key.shape[axis]');
  });

  it('negates compound binary-label guards without rewriting either inequality', () => {
    const problem = codePracticeProblems.find(
      (candidate) => candidate.id === 'binary-cross-entropy-from-probabilities',
    );
    const annotatedCode = augmentCodeWithSolution(problem!);

    expect(annotatedCode).toContain(
      'assert not (bool(torch.any((target != 0) & (target != 1))))',
    );
    expect(annotatedCode).not.toContain('(target == 0) & (target != 1)');
  });
});
