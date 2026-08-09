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
      expect(annotatedCode, problem.id).toContain('# Reference solution loaded:');
      expect(annotatedCode, problem.id).toContain(
        '# The TODO plan above stays in place; the annotated lines below implement it.',
      );

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
  });

  it('adds solution-only imports and helpers before the starter function that uses them', () => {
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

    expect(softmaxCode.indexOf('def _validate_classification_inputs')).toBeLessThan(
      softmaxCode.indexOf('def softmax_cross_entropy'),
    );
    expect(temperatureCode).toContain('import math');
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
    expect(annotatedCode).toContain('# Reference solution loaded:');
    expect(annotatedCode).toContain('return torch.mean(losses)');
  });
});
