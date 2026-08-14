import { describe, expect, it } from 'vitest';
import { codePracticeProblems } from '../src/lib/code-practice';
import { TORCH_COMPAT_SOURCE } from '../src/lib/torch-compat';

// These helpers solve the operation being taught instead of exposing its tensor steps.
const BANNED_CONVENIENCE_CALLS = [
  'torch.softmax(',
  'torch.log_softmax(',
  'torch.logsumexp(',
  'torch.topk(',
  'torch.norm(',
  'torch.tril(',
  'torch.triu(',
  'torch.nn.functional.',
] as const;

const REQUIRED_PRIMITIVES = [
  {
    id: 'stable-softmax-cross-entropy',
    fragments: ['torch.amax(', 'torch.exp(', 'torch.sum(', 'torch.log('],
  },
  {
    id: 'causal-attention-mask',
    fragments: ['positions[:, None] >= positions[None, :]'],
  },
  {
    id: 'pairwise-cosine-similarity',
    fragments: ['torch.sqrt(', 'torch.sum('],
  },
  {
    id: 'top-k-accuracy',
    fragments: ['torch.argsort(', 'ranked[:, :top_k]'],
  },
  {
    id: 'nearest-centroid-classifier',
    fragments: ['torch.sum(class_points, dim=0)', '/ class_points.shape[0]'],
  },
  {
    id: 'temperature-scaling-of-logits',
    fragments: ['torch.amax(', 'torch.exp(', 'torch.sum('],
  },
  {
    id: 'scaled-dot-product-self-attention',
    fragments: ['torch.amax(', 'torch.exp(', 'torch.sum('],
  },
  {
    id: 'cross-attention',
    fragments: ['torch.amax(', 'torch.exp(', 'torch.sum('],
  },
  {
    id: 'manual-backprop-for-a-2-layer-mlp',
    fragments: ['torch.amax(', 'torch.exp(', 'torch.sum(', 'torch.log('],
  },
  {
    id: 'classic-mlp-forward-backward',
    fragments: ['torch.amax(', 'torch.exp(', 'torch.sum(', 'torch.log('],
  },
] as const;

describe('code-practice primitive-first solutions', () => {
  it('gives every exercise a reference walkthrough and unique ordered route', () => {
    const ids = new Set<string>();

    for (const [index, problem] of codePracticeProblems.entries()) {
      expect(ids.has(problem.id), problem.id).toBe(false);
      ids.add(problem.id);
      expect(problem.order, problem.id).toBe(index + 1);
      expect(problem.solutionNotes.length, problem.id).toBeGreaterThan(0);
      expect(problem.solutionCode.trim(), problem.id).not.toBe('');
    }
  });

  it('keeps learner-facing reference implementations concise', () => {
    for (const problem of codePracticeProblems) {
      expect(problem.solutionCode.split('\n').length, problem.id).toBeLessThanOrEqual(45);
      expect(
        problem.solutionCode.split('\n').filter((line) => line.trimStart().startsWith('#')),
        problem.id,
      ).toHaveLength(0);
    }
  });

  it('does not bypass a lesson with a matching PyTorch convenience helper', () => {
    const torchProblems = codePracticeProblems.filter((problem) => problem.packages?.includes('torch'));

    for (const problem of torchProblems) {
      for (const call of BANNED_CONVENIENCE_CALLS) {
        expect(
          problem.solutionCode,
          `${problem.id} solution must build ${call} from lower-level tensor operations`,
        ).not.toContain(call);
        expect(
          problem.starterCode,
          `${problem.id} starter must not direct learners to ${call}`,
        ).not.toContain(call);
      }
    }
  });

  it('keeps the core mechanics explicit in the relevant worked solutions', () => {
    for (const { id, fragments } of REQUIRED_PRIMITIVES) {
      const problem = codePracticeProblems.find((candidate) => candidate.id === id);
      expect(problem, `missing code-practice problem: ${id}`).toBeDefined();

      for (const fragment of fragments) {
        expect(problem?.solutionCode, `${id} must include ${fragment}`).toContain(fragment);
      }
    }
  });

  it('handles the two audited edge cases in the reference code', () => {
    const ngram = codePracticeProblems.find((problem) => problem.id === 'simple-n-gram-language-model');
    const matching = codePracticeProblems.find((problem) => problem.id === 'greedy-detection-matching');

    expect(ngram?.solutionCode).toContain('key = tuple(context[-size:]) if size else ()');
    expect(matching?.solutionCode).toContain('candidate_ious = torch.where(available');
    expect(matching?.solutionCode).not.toContain('best_gt not in used');
  });

  it('supports tensor transpose methods used by 2D and attention references', () => {
    expect(TORCH_COMPAT_SOURCE).toContain('def permute(self, *dims):');
    expect(TORCH_COMPAT_SOURCE).toContain('def transpose(self, dim0, dim1):');
  });
});
