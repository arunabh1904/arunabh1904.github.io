export interface CodePracticeExample {
  label: string;
  lines: string[];
  result: string;
}

export interface CodePracticeProblem {
  id: string;
  order: number;
  title: string;
  difficulty: 'Easy' | 'Medium' | 'Hard';
  summary: string;
  prompt: string[];
  signature: string;
  requirements: string[];
  examples: CodePracticeExample[];
  hint: string[];
  solutionNotes: string[];
  solutionCode: string;
  starterCode: string;
  packages?: readonly string[];
  tags?: readonly string[];
}

export const CODE_PRACTICE_SECTION_SUMMARY =
  'Interview-style Python problems with runnable starter code, hints, and hidden solutions.';

export function getCodePracticeProblemPath(problem: Pick<CodePracticeProblem, 'id'> | string) {
  const problemId = typeof problem === 'string' ? problem : problem.id;
  return `/code/${problemId}.html`;
}

export function getCodePracticeProblemById(problemId: string) {
  return codePracticeProblems.find((problem) => problem.id === problemId);
}

export const codePracticeProblems: readonly CodePracticeProblem[] = [
  {
    id: 'stable-softmax-cross-entropy',
    order: 1,
    title: 'Stable softmax cross-entropy',
    difficulty: 'Medium',
    summary:
      'Implement a numerically stable batch softmax cross-entropy loss in NumPy with proper input validation.',
    prompt: [
      'Write `softmax_cross_entropy(logits, labels)` so it returns the mean cross-entropy loss across a batch.',
      'Treat this like an interview question: keep the implementation concise, validate the inputs, and avoid numerical overflow when computing the softmax terms.',
    ],
    signature: `def softmax_cross_entropy(logits, labels):
    ...`,
    requirements: [
      '`logits` is a 2D NumPy array of shape `(N, C)`.',
      '`labels` is a 1D NumPy array of shape `(N,)` with integer class ids in `[0, C - 1]`.',
      'Return the mean cross-entropy loss over the batch.',
      'The implementation must be numerically stable.',
      'Raise `ValueError` on invalid shapes or invalid labels.',
    ],
    examples: [
      {
        label: 'Example',
        lines: ['logits = [[2.0, 1.0, 0.1]]', 'labels = [0]'],
        result: 'loss ~= 0.41703',
      },
    ],
    hint: [
      'Subtract the per-row maximum from `logits` before exponentiating.',
      'Use `np.arange(N)` to gather the logit for the correct class in each row.',
      'Compute the loss as `logsumexp - correct_class_logit`, then average across the batch.',
      'Validate `ndim`, matching batch size, integer labels, non-empty shapes, and label range.',
    ],
    solutionNotes: [
      'The stable trick is to shift each row by its maximum value before applying `exp`, which preserves the softmax probabilities while avoiding overflow.',
      'Once shifted, the mean cross-entropy is just the average of `log(sum(exp(shifted))) - shifted[row, label]` across the batch.',
    ],
    solutionCode: `import numpy as np

def softmax_cross_entropy(logits, labels):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)

    if logits.ndim != 2:
        raise ValueError("logits must be a 2D array of shape (N, C)")
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array of shape (N,)")

    batch_size, num_classes = logits.shape
    if batch_size == 0:
        raise ValueError("logits must contain at least one sample")
    if num_classes == 0:
        raise ValueError("logits must contain at least one class")
    if labels.shape[0] != batch_size:
        raise ValueError("labels must have the same batch size as logits")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must contain integer class ids")
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError("labels contain out-of-range class ids")

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(shifted), axis=1))
    correct_class_logits = shifted[np.arange(batch_size), labels]
    losses = logsumexp - correct_class_logits

    return float(np.mean(losses))`,
    starterCode: `import numpy as np

def softmax_cross_entropy(logits, labels):
    logits = np.asarray(logits)
    labels = np.asarray(labels)

    # TODO:
    # 1. Validate the input shapes and labels.
    # 2. Compute a numerically stable mean cross-entropy loss.
    raise NotImplementedError("Implement softmax_cross_entropy")

sample_logits = np.array([[2.0, 1.0, 0.1]])
sample_labels = np.array([0])

print(f"{softmax_cross_entropy(sample_logits, sample_labels):.5f}")`,
    packages: ['numpy'],
    tags: ['NumPy', 'Numerical Stability', 'Interview Practice'],
  },
] as const;
