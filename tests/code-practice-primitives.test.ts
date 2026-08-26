import { execFileSync } from 'node:child_process';
import { describe, expect, it } from 'vitest';
import { codePracticeProblems } from '../src/lib/code-practice';
import { TORCH_COMPAT_SOURCE } from '../src/lib/torch-compat';

function compilePython(code: string) {
  execFileSync(
    'python3',
    ['-c', 'import sys; compile(sys.stdin.read(), "solution.py", "exec")'],
    { encoding: 'utf8', input: code },
  );
}

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
    id: 'class-weighted-cross-entropy',
    fragments: [
      'class_weight[labels]',
      'torch.sum(losses * example_weight)',
      '/ torch.sum(example_weight)',
    ],
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
    id: 'incremental-kv-cache',
    fragments: ['@dataclass', 'start_pos != self.length', 'torch.cat((self.key, key), dim=2)'],
  },
  {
    id: 'grouped-query-and-multi-query-attention',
    fragments: [
      'query_heads % kv_heads != 0',
      'torch.broadcast_to(',
      'query_heads // kv_heads',
      'torch.amax(',
    ],
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
      expect(problem.prompt.length, problem.id).toBeGreaterThan(0);
      expect(problem.requirements.length, problem.id).toBeGreaterThan(0);
      expect(problem.examples.length, problem.id).toBeGreaterThan(0);
      expect(problem.hint.length, problem.id).toBeGreaterThan(0);
    }
  });

  it('keeps learner-facing reference implementations concise', () => {
    const fundamentals = codePracticeProblems.filter((problem) => problem.track === 'fundamentals');

    for (const problem of fundamentals) {
      expect(problem.solutionCode.split('\n').length, problem.id).toBeLessThanOrEqual(45);
      expect(
        problem.solutionCode.split('\n').filter((line) => line.trimStart().startsWith('#')),
        problem.id,
      ).toHaveLength(0);
    }
  });

  it('offers NumPy only for concise array-math interviews', () => {
    const expectedIds = [
      'l1-regression-loss',
      'binary-cross-entropy-from-probabilities',
      'masked-mean',
      'binary-classification-metrics',
      'top-k-accuracy',
      'single-box-iou',
      'wrapped-angular-difference',
      'smooth-l1-huber-loss',
      'stable-softmax-cross-entropy',
      'class-weighted-cross-entropy',
      'temperature-scaling-of-logits',
      'pairwise-squared-distance',
      'pairwise-cosine-similarity',
      'nearest-centroid-classifier',
      'iou-matrix',
      'non-maximum-suppression',
      'dice-loss',
      'segmentation-iou-loss',
      'focal-loss',
      'top-k-gather',
      'homogeneous-coordinate-transform',
      '2d-patchify-for-images',
      'unpatchify-back-to-image',
      'sinusoidal-positional-encoding',
      'causal-attention-mask',
      'rope-rotary-positional-embedding',
      'scaled-dot-product-self-attention',
      'average-precision-from-matches',
      'manual-backprop-for-a-2-layer-mlp',
    ];
    const numpyProblems = codePracticeProblems.filter((problem) => problem.numpyAlternative);

    expect(numpyProblems.map((problem) => problem.id)).toEqual(expectedIds);

    for (const problem of numpyProblems) {
      const alternative = problem.numpyAlternative!;
      expect(problem.tags, problem.id).toContain('NumPy');
      expect(alternative.code, problem.id).toContain('import numpy as np');
      expect(alternative.code, problem.id).toContain('np.ndarray');
      expect(alternative.code, problem.id).toMatch(/\) -> [^:]+:/);
      expect(alternative.code, problem.id).not.toContain('torch');
      expect(alternative.code.split('\n').length, problem.id).toBeLessThanOrEqual(30);
      expect(alternative.exampleCode, problem.id).toContain('print(');
      expect(alternative.exampleCode, problem.id).not.toContain('torch');
      expect(alternative.exampleCode.split('\n').length, problem.id).toBeLessThanOrEqual(12);
      expect(alternative.memory.length, problem.id).toBeGreaterThan(0);
      expect(alternative.memory.length, problem.id).toBeLessThanOrEqual(2);
      expect(
        () => compilePython(`${alternative.code}\n\n${alternative.exampleCode}`),
        problem.id,
      ).not.toThrow();
    }

    for (const id of [
      'incremental-kv-cache',
      'grouped-query-and-multi-query-attention',
      'simple-n-gram-language-model',
      'resnet-from-building-blocks',
      'unet-encoder-decoder',
      'centernet-style-detector',
    ]) {
      expect(codePracticeProblems.find((problem) => problem.id === id)?.numpyAlternative, id).toBeUndefined();
    }
  });

  it('keeps representative NumPy formulas and shape transforms explicit', () => {
    const getAlternative = (id: string) =>
      codePracticeProblems.find((problem) => problem.id === id)!.numpyAlternative!.code;

    expect(getAlternative('stable-softmax-cross-entropy')).toContain(
      'np.max(logits, axis=1, keepdims=True)',
    );
    expect(getAlternative('pairwise-squared-distance')).toContain(
      'x_squared + y_squared - 2 * x @ y.T',
    );
    expect(getAlternative('top-k-gather')).toContain('np.take_along_axis');
    expect(getAlternative('2d-patchify-for-images')).toContain(
      'transpose(0, 2, 4, 1, 3, 5)',
    );
    expect(getAlternative('unpatchify-back-to-image')).toContain(
      'transpose(0, 3, 1, 4, 2, 5)',
    );
    expect(getAlternative('manual-backprop-for-a-2-layer-mlp')).toContain(
      'dhidden = (dlogits @ W2.T) * (hidden_pre > 0)',
    );
  });

  it('starts plain-function interviews blank while preserving class scaffolds', () => {
    for (const problem of codePracticeProblems) {
      const isPlainFunction = problem.signature.trimStart().startsWith('def ');
      expect(problem.editorStart, problem.id).toBe(isPlainFunction ? 'blank' : 'scaffold');
    }

    expect(
      codePracticeProblems.find((problem) => problem.id === 'incremental-kv-cache')?.editorStart,
    ).toBe('scaffold');
    expect(
      codePracticeProblems.find((problem) => problem.id === 'simple-n-gram-language-model')
        ?.editorStart,
    ).toBe('scaffold');
  });

  it('does not bypass a lesson with a matching PyTorch convenience helper', () => {
    const torchProblems = codePracticeProblems.filter(
      (problem) => problem.track === 'fundamentals' && problem.packages?.includes('torch'),
    );

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
    const bce = codePracticeProblems.find(
      (problem) => problem.id === 'binary-cross-entropy-from-probabilities',
    );

    expect(ngram?.solutionCode).toContain('key = tuple(context[-size:]) if size else ()');
    expect(matching?.solutionCode).toContain('candidate_ious = torch.where(available');
    expect(matching?.solutionCode).not.toContain('best_gt not in used');
    expect(bce?.solutionCode).toContain('(target != 0) & (target != 1)');
    expect(bce?.solutionCode).toContain('(probability < 0) | (probability > 1)');
    expect(bce?.solutionNotes.join(' ')).toContain('max(z, 0) - z*y + log(1 + exp(-|z|))');
    expect(bce?.solutionNotes.join(' ')).not.toContain('sigmoid, then clamp, then log is stable');
    expect(bce?.visual?.src).toBe('/assets/images/code-bce-probabilities-vs-logits.gif');
  });

  it('supports tensor transpose methods used by 2D and attention references', () => {
    expect(TORCH_COMPAT_SOURCE).toContain('def permute(self, *dims):');
    expect(TORCH_COMPAT_SOURCE).toContain('def transpose(self, dim0, dim1):');
  });

  it('covers the inference-attention interview sequence and its reasoning axes', () => {
    const ids = [
      'stable-softmax-cross-entropy',
      'causal-attention-mask',
      'rope-rotary-positional-embedding',
      'scaled-dot-product-self-attention',
      'incremental-kv-cache',
      'grouped-query-and-multi-query-attention',
    ];
    const attentionProblems = ids.map((id) =>
      codePracticeProblems.find((problem) => problem.id === id),
    );

    expect(attentionProblems.every(Boolean)).toBe(true);
    expect(attentionProblems[3]?.title).toContain('MHA');
    expect(attentionProblems[4]?.solutionCode).toContain('class KVCache:');
    expect(attentionProblems[4]?.solutionCode).toContain('cached_layout != update_layout');
    expect(attentionProblems[5]?.title).toContain('GQA');
    expect(attentionProblems[5]?.title).toContain('MQA');

    const axes = new Set(
      attentionProblems.flatMap((problem) => problem?.reasoning?.map((point) => point.axis) ?? []),
    );
    expect(axes).toEqual(
      new Set([
        'Inference efficiency',
        'Tensor reasoning',
        'Memory / computation tradeoff',
        'Cache update correctness',
      ]),
    );

    for (const problem of attentionProblems.slice(1)) {
      expect(problem?.interview?.followUps.length, problem?.id).toBeGreaterThan(0);
    }
  });

  it('keeps architecture interviews typed, modular, and local-PyTorch only', () => {
    const architectures = codePracticeProblems.filter((problem) => problem.track === 'architecture');

    expect(architectures.map((problem) => problem.id)).toEqual([
      'resnet-from-building-blocks',
      'unet-encoder-decoder',
      'centernet-style-detector',
    ]);

    for (const problem of architectures) {
      expect(problem.environment, problem.id).toBe('local-pytorch');
      expect(problem.interview?.durationMinutes, problem.id).toBeGreaterThanOrEqual(45);
      expect(problem.solutionCode, problem.id).toContain('@dataclass');
      expect(problem.solutionCode, problem.id).toContain('nn.Module');
      expect(problem.solutionCode, problem.id).toContain('def smoke_test()');
      expect(problem.solutionCode, problem.id).toContain('with torch.inference_mode():');
    }

    expect(architectures[0].solutionCode).toContain('nn.AdaptiveAvgPool2d(1)');
    expect(architectures[1].solutionCode).toContain('F.interpolate(');
    expect(architectures[1].solutionCode).toContain('nn.ModuleList(');
    expect(architectures[2].solutionCode).toContain('class CenterNetOutput:');
    expect(architectures[2].solutionCode).toContain('nn.init.constant_');
    expect(architectures[2].solutionCode).toContain('-2.19');
  });
});
