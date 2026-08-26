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

function runPython(code: string) {
  return execFileSync('python3', ['-c', code], {
    encoding: 'utf8',
    timeout: 15_000,
  });
}

const CLASS_BASED_PROBLEMS = new Set([
  'nearest-centroid-classifier',
  'scaled-dot-product-self-attention',
  'incremental-kv-cache',
  'grouped-query-and-multi-query-attention',
  'cross-attention',
  'simple-n-gram-language-model',
  'resnet-from-building-blocks',
  'unet-encoder-decoder',
  'centernet-style-detector',
]);

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
      'torch.broadcast_to(',
      'self.config.num_query_heads // heads',
      'torch.amax(',
      'return "mqa" if self.num_kv_heads == 1 else "gqa"',
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
  it('keeps every explanation scannable and puts key expressions on their own lines', () => {
    expect(codePracticeProblems).toHaveLength(40);

    for (const problem of codePracticeProblems) {
      const hasStandaloneExpression = problem.solutionNotes.some((note) =>
        note.split('\n').some((line) => /^`[^`]+`$/.test(line)),
      );
      expect(hasStandaloneExpression, problem.id).toBe(true);

      for (const note of problem.solutionNotes) {
        expect(note.length, `${problem.id} has a dense explanation paragraph`).toBeLessThanOrEqual(
          360,
        );

        for (const match of note.matchAll(/`([^`]+)`/g)) {
          const expression = match[1];
          const isKeyExpression =
            expression.length >= 24 && (/[=Σ@]/.test(expression) || expression.includes('shape'));

          if (isKeyExpression) {
            expect(note, `${problem.id} embeds a key expression in prose`).toContain(
              `\n\`${expression}\``,
            );
          }
        }
      }
    }
  });

  it('lets explanation depth follow the problem instead of a fixed template', () => {
    const explanationBundles = new Set<string>();

    for (const problem of codePracticeProblems) {
      const bundle = problem.solutionNotes.join('\n');
      expect(explanationBundles.has(bundle), `${problem.id} repeats another explanation`).toBe(false);
      explanationBundles.add(bundle);

      const minimumSteps =
        problem.track === 'architecture' ? 5 : problem.difficulty === 'Hard' ? 4 : 2;
      expect(problem.solutionNotes.length, problem.id).toBeGreaterThanOrEqual(minimumSteps);
    }

    const resnet = codePracticeProblems.find(
      (problem) => problem.id === 'resnet-from-building-blocks',
    )!;
    expect(resnet.solutionNotes.join(' ')).toContain('identity skip');
    expect(resnet.solutionNotes.join(' ')).toContain('1x1');
    expect(resnet.solutionNotes.join(' ')).toContain('_make_stage');
    expect(resnet.solutionNotes.join(' ')).toContain('Adaptive average pooling');

    const ngram = codePracticeProblems.find(
      (problem) => problem.id === 'simple-n-gram-language-model',
    )!;
    expect(ngram.solutionNotes.join(' ')).toContain('runs offline');
    expect(ngram.prompt.join(' ')).not.toContain('Tiny Shakespeare');
  });

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
      const lineLimit = CLASS_BASED_PROBLEMS.has(problem.id) ? 80 : 45;
      expect(problem.solutionCode.split('\n').length, problem.id).toBeLessThanOrEqual(lineLimit);
      expect(
        problem.solutionCode.split('\n').filter((line) => line.trimStart().startsWith('#')),
        problem.id,
      ).toHaveLength(0);
    }
  });

  it('uses classes only when the interview problem owns reusable state', () => {
    expect(
      codePracticeProblems
        .filter((problem) => !problem.signature.trimStart().startsWith('def '))
        .map((problem) => problem.id),
    ).toEqual([...CLASS_BASED_PROBLEMS]);

    for (const problem of codePracticeProblems) {
      if (CLASS_BASED_PROBLEMS.has(problem.id)) {
        expect(problem.solutionCode, problem.id).toContain('class ');
        expect(problem.solutionCode, problem.id).toContain('def smoke_test()');
        expect(problem.solutionCode.trimEnd(), problem.id).toMatch(/smoke_test\(\)$/);
      } else {
        expect(problem.signature.trimStart(), problem.id).toMatch(/^def /);
      }
    }

    for (const id of [
      'scaled-dot-product-self-attention',
      'grouped-query-and-multi-query-attention',
      'cross-attention',
      'resnet-from-building-blocks',
      'unet-encoder-decoder',
      'centernet-style-detector',
    ]) {
      expect(codePracticeProblems.find((problem) => problem.id === id)?.solutionCode, id).toContain(
        'nn.Module',
      );
    }
  });

  it('runs the n-gram reference solution end to end', () => {
    const ngram = codePracticeProblems.find(
      (problem) => problem.id === 'simple-n-gram-language-model',
    )!;
    expect(runPython(ngram.solutionCode)).toContain('n-gram smoke test passed');
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

  it('starts every interview from a blank editor', () => {
    for (const problem of codePracticeProblems) {
      expect(problem.editorStart, problem.id).toBe('blank');
    }

    expect(
      codePracticeProblems.find((problem) => problem.id === 'incremental-kv-cache')?.editorStart,
    ).toBe('blank');
    expect(
      codePracticeProblems.find((problem) => problem.id === 'scaled-dot-product-self-attention')
        ?.editorStart,
    ).toBe('blank');
    expect(
      codePracticeProblems.find((problem) => problem.id === 'simple-n-gram-language-model')
        ?.editorStart,
    ).toBe('blank');
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
    expect(bce?.solutionNotes).toHaveLength(6);
    expect(bce?.solutionNotes.join(' ')).toContain(
      'max(z_i, 0) - z_i y_i + log(1 + exp(-|z_i|))',
    );
    expect(bce?.solutionNotes.join('\n')).toContain(
      '\n`L = -(1 / K) Σ_i [y_i log(p_i) + (1 - y_i) log(1 - p_i)]`',
    );
    expect(bce?.solutionNotes.join(' ')).not.toContain('sigmoid, then clamp, then log is stable');
    expect(bce?.visual?.src).toBe(
      '/assets/images/code-glance-binary-cross-entropy-from-probabilities.svg',
    );
  });

  it('supports tensor transpose methods used by 2D and attention references', () => {
    expect(TORCH_COMPAT_SOURCE).toContain('def permute(self, *dims):');
    expect(TORCH_COMPAT_SOURCE).toContain('def transpose(self, dim0, dim1):');
  });

  it('supports the inference layers used by browser architecture builds', () => {
    for (const layer of [
      '_nn.Conv2d = _Conv2d',
      '_nn.ConvTranspose2d = _ConvTranspose2d',
      '_nn.BatchNorm2d = _BatchNorm2d',
      '_nn.MaxPool2d = _MaxPool2d',
      '_nn.AdaptiveAvgPool2d = _AdaptiveAvgPool2d',
      '_functional.interpolate = _interpolate',
      '_init.constant_',
    ]) {
      expect(TORCH_COMPAT_SOURCE).toContain(layer);
    }
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
    expect(attentionProblems[3]?.solutionCode).toContain('class MultiHeadSelfAttention(nn.Module):');
    expect(attentionProblems[4]?.solutionCode).toContain('class KVCache:');
    expect(attentionProblems[4]?.solutionCode).toContain('cached_layout != update_layout');
    expect(attentionProblems[5]?.title).toContain('GQA');
    expect(attentionProblems[5]?.title).toContain('MQA');
    expect(attentionProblems[5]?.solutionCode).toContain('for kv_heads, expected_mode in');

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

  it('keeps architecture interviews typed, modular, and browser-runnable', () => {
    const architectures = codePracticeProblems.filter((problem) => problem.track === 'architecture');

    expect(architectures.map((problem) => problem.id)).toEqual([
      'resnet-from-building-blocks',
      'unet-encoder-decoder',
      'centernet-style-detector',
    ]);

    for (const problem of architectures) {
      expect(problem.packages, problem.id).toContain('torch');
      expect(problem.interview?.durationMinutes, problem.id).toBeGreaterThanOrEqual(45);
      expect(problem.solutionCode, problem.id).toContain('@dataclass');
      expect(problem.solutionCode, problem.id).toContain('nn.Module');
      expect(problem.solutionCode, problem.id).toContain('def smoke_test()');
      expect(problem.solutionCode, problem.id).toContain('with torch.inference_mode():');
      expect(problem.solutionCode, problem.id).toContain('.eval()');
    }

    expect(architectures[0].solutionCode).toContain('nn.AdaptiveAvgPool2d(1)');
    expect(architectures[1].solutionCode).toContain('F.interpolate(');
    expect(architectures[1].solutionCode).toContain('nn.ModuleList(');
    expect(architectures[2].solutionCode).toContain('class CenterNetOutput:');
    expect(architectures[2].solutionCode).toContain('nn.init.constant_');
    expect(architectures[2].solutionCode).toContain('-2.19');
  });
});
