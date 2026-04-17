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

    # Numerical stability trick: subtract per-row max before exponentiating
    shifted = logits - np.max(logits, axis=1, keepdims=True)

    # log(sum(exp(logits))) computed in a stable way
    logsumexp = np.log(np.sum(np.exp(shifted), axis=1))

    # Gather the logit for the correct class for each example
    correct_class_logits = shifted[np.arange(batch_size), labels]

    # Cross-entropy loss per example
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
  {
    id: 'non-maximum-suppression',
    order: 2,
    title: 'Non-maximum suppression',
    difficulty: 'Medium',
    summary:
      'Implement NumPy-based non-maximum suppression with deterministic score tie-breaking and invalid-box checks.',
    prompt: [
      'Write `nms(boxes, scores, iou_threshold)` so it returns the indices of the boxes kept after non-maximum suppression.',
      'Process boxes in descending score order, break ties by smaller original index, suppress only boxes whose IoU with a kept box is strictly greater than `iou_threshold`, and raise `ValueError` when a box has `x2 < x1` or `y2 < y1`.',
    ],
    signature: `def nms(boxes, scores, iou_threshold):
    ...`,
    requirements: [
      '`boxes` is an `(N, 4)` NumPy array of `[x1, y1, x2, y2]`.',
      '`scores` is a NumPy array of shape `(N,)`.',
      'Return a list of selected box indices after non-maximum suppression.',
      'Process boxes in descending order of score.',
      'If scores tie, prefer the smaller original index first.',
      'Suppress boxes whose IoU with a kept box is strictly greater than `iou_threshold`.',
      'Raise `ValueError` if any box has `x2 < x1` or `y2 < y1`.',
    ],
    examples: [
      {
        label: 'Example 1',
        lines: [
          'boxes = [[0, 0, 2, 2], [0.5, 0.5, 2.5, 2.5], [5, 5, 7, 7]]',
          'scores = [0.9, 0.8, 0.7]',
          'iou_threshold = 0.3',
        ],
        result: '[0, 2]',
      },
      {
        label: 'Example 2',
        lines: [
          'boxes = [[0, 0, 2, 2], [0, 0, 2, 2]]',
          'scores = [0.5, 0.5]',
          'iou_threshold = 0.1',
        ],
        result: '[0]',
      },
    ],
    hint: [
      'Sort the candidate indices by `(-score, index)` so the traversal order is deterministic.',
      'A small helper that computes IoU between one box and many remaining boxes keeps the main loop clean.',
      'After keeping the current best box, remove only the boxes with `IoU > iou_threshold`; boxes with equal IoU to the threshold should stay.',
      'Validate the box coordinates before you start suppressing anything.',
    ],
    solutionNotes: [
      'The clean approach is greedy: sort indices by descending score with index-based tie-breaking, repeatedly keep the first remaining box, and compare it against the rest.',
      'Using a vectorized IoU helper lets the loop filter the remaining candidates in one shot while still keeping the implementation short and readable.',
    ],
    solutionCode: `import numpy as np

def compute_iou(box, boxes):
    """
    Compute IoU between one box and many boxes.

    Args:
        box: np.ndarray of shape (4,)
        boxes: np.ndarray of shape (M, 4)

    Returns:
        np.ndarray of shape (M,)
    """
    # Compute the overlap rectangle by clamping the corners.
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    # Negative widths or heights mean there is no overlap.
    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    # Area of the reference box and each candidate box.
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # IoU is intersection divided by union; guard against zero union.
    union = box_area + boxes_area - inter_area
    iou = np.where(union > 0.0, inter_area / union, 0.0)
    return iou


def nms(boxes, scores, iou_threshold):
    """
    Perform non-maximum suppression.

    Args:
        boxes: np.ndarray of shape (N, 4)
        scores: np.ndarray of shape (N,)
        iou_threshold: float

    Returns:
        list[int]: selected indices in the order they are kept
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)

    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape (N, 4)")
    if scores.ndim != 1 or scores.shape[0] != boxes.shape[0]:
        raise ValueError("scores must have shape (N,)")
    if not (0.0 <= iou_threshold <= 1.0):
        raise ValueError("iou_threshold must be in [0, 1]")

    # Reject malformed boxes before running the suppression loop.
    if np.any(boxes[:, 2] < boxes[:, 0]) or np.any(boxes[:, 3] < boxes[:, 1]):
        raise ValueError("invalid boxes detected")

    n = boxes.shape[0]
    if n == 0:
        return []

    # Sort by descending score, then by smaller original index on ties.
    order = sorted(range(n), key=lambda i: (-scores[i], i))
    keep = []

    while order:
        current = order[0]
        keep.append(current)

        if len(order) == 1:
            break

        remaining = np.array(order[1:], dtype=int)
        ious = compute_iou(boxes[current], boxes[remaining])

        # Keep only boxes whose overlap is not strictly above the threshold.
        survivors = remaining[ious <= iou_threshold]
        order = survivors.tolist()

    return keep`,
    starterCode: `import numpy as np

def nms(boxes, scores, iou_threshold):
    boxes = np.asarray(boxes)
    scores = np.asarray(scores)

    # TODO:
    # 1. Validate the shapes and reject invalid boxes.
    # 2. Sort candidate indices by score, breaking ties with the smaller index.
    # 3. Repeatedly keep the best remaining box and suppress boxes
    #    whose IoU is strictly greater than the threshold.
    raise NotImplementedError("Implement nms")

sample_boxes = np.array([
    [0, 0, 2, 2],
    [0.5, 0.5, 2.5, 2.5],
    [5, 5, 7, 7],
])
sample_scores = np.array([0.9, 0.8, 0.7])

print(nms(sample_boxes, sample_scores, iou_threshold=0.3))`,
    packages: ['numpy'],
    tags: ['NumPy', 'Computer Vision', 'Greedy'],
  },
  {
    id: 'causal-attention-mask',
    order: 3,
    title: 'Causal attention mask',
    difficulty: 'Easy',
    summary:
      'Build a batch of lower-triangular attention masks from per-example sequence lengths, with optional padding to a shared width.',
    prompt: [
      'Write `make_causal_attention_mask(seq_lens, max_len=None)` to build a batch of causal attention masks.',
      'Each example gets its own valid length. Positions outside that valid length must stay `0`, while valid positions should form a lower-triangular mask where token `i` can attend to itself and earlier tokens only.',
    ],
    signature: `def make_causal_attention_mask(seq_lens, max_len=None):
    ...`,
    requirements: [
      '`seq_lens` is a 1D list or NumPy array of length `B`.',
      'Each entry is the valid sequence length for one example.',
      'Return a mask of shape `(B, T, T)` where `T = max(max(seq_lens), max_len if given)`.',
      '`mask[b, i, j] == 1` iff `i < seq_lens[b]`, `j < seq_lens[b]`, and `j <= i`.',
      'Otherwise the entry must be `0`.',
      'Raise `ValueError` on invalid input.',
    ],
    examples: [
      {
        label: 'Example',
        lines: ['seq_lens = [3, 1]'],
        result:
          '[[[1, 0, 0], [1, 1, 0], [1, 1, 1]], [[1, 0, 0], [0, 0, 0], [0, 0, 0]]]',
      },
    ],
    hint: [
      'Use `np.tri(T, dtype=np.int64)` or `np.tril` to build the causal lower triangle once.',
      'Create a `(B, T)` validity mask from `seq_lens`, then broadcast it across rows and columns.',
      'Multiply the causal triangle by the validity masks so padded rows and columns stay zero.',
      'Validate the rank of `seq_lens`, integer lengths, non-negative values, and `max_len` when it is provided.',
    ],
    solutionNotes: [
      'The problem is really two masks multiplied together: the causal rule (`j <= i`) and the per-example validity rule (`i, j < seq_len[b]`).',
      'Broadcasting a single lower-triangular template against a batch-wise validity mask gives the full `(B, T, T)` answer without explicit Python loops.',
    ],
    solutionCode: `import numpy as np

def make_causal_attention_mask(seq_lens, max_len=None):
    seq_lens = np.asarray(seq_lens)

    if seq_lens.ndim != 1:
        raise ValueError("seq_lens must be a 1D array")
    if seq_lens.size == 0:
        raise ValueError("seq_lens must not be empty")
    if not np.issubdtype(seq_lens.dtype, np.integer):
        raise ValueError("seq_lens must contain integers")
    if np.any(seq_lens < 0):
        raise ValueError("seq_lens must be non-negative")

    T = int(seq_lens.max())
    if max_len is not None:
        if isinstance(max_len, bool) or not isinstance(max_len, (int, np.integer)):
            raise ValueError("max_len must be an integer or None")
        if max_len < 0:
            raise ValueError("max_len must be non-negative")
        T = max(T, int(max_len))

    valid = np.arange(T) < seq_lens[:, None]
    causal = np.tri(T, dtype=np.int64)
    mask = causal[None, :, :] * valid[:, :, None] * valid[:, None, :]
    return mask.astype(np.int64)`,
    starterCode: `import numpy as np

def make_causal_attention_mask(seq_lens, max_len=None):
    seq_lens = np.asarray(seq_lens)

    # TODO:
    # 1. Validate the input shape and sequence lengths.
    # 2. Build a causal lower-triangular mask for each batch element.
    raise NotImplementedError("Implement make_causal_attention_mask")

sample_seq_lens = np.array([3, 1])
print(make_causal_attention_mask(sample_seq_lens, max_len=4))`,
    packages: ['numpy'],
    tags: ['NumPy', 'Attention Masks', 'Sequence Modeling'],
  },
  {
    id: 'binary-classification-metrics',
    order: 4,
    title: 'Binary classification metrics',
    difficulty: 'Easy',
    summary:
      'Compute confusion-matrix counts and common binary classification metrics with zero-division safeguards.',
    prompt: [
      'Write `binary_classification_metrics(y_true, y_pred)` so it returns the confusion-matrix counts and derived metrics for a binary classifier.',
      'Treat `y_true` and `y_pred` as equal-length 1D collections of binary labels. Validate the inputs, and make sure any metric with a zero denominator returns `0.0` instead of failing.',
    ],
    signature: `def binary_classification_metrics(y_true, y_pred):
    ...`,
    requirements: [
      '`y_true` and `y_pred` are equal-length 1D arrays or lists containing only `0` and `1`.',
      'Return a dictionary with keys `tp`, `tn`, `fp`, `fn`, `precision`, `recall`, `f1`, and `accuracy`.',
      '`precision`, `recall`, `f1`, and `accuracy` should be floats.',
      'If a denominator is zero, return `0.0` for that metric.',
      'Raise `ValueError` on invalid inputs.',
    ],
    examples: [
      {
        label: 'Example 1',
        lines: ['y_true = [1, 0, 1, 0]', 'y_pred = [1, 0, 0, 1]'],
        result:
          `{'tp': 1, 'tn': 1, 'fp': 1, 'fn': 1, 'precision': 0.5, 'recall': 0.5, 'f1': 0.5, 'accuracy': 0.5}`,
      },
      {
        label: 'Example 2',
        lines: ['y_true = [0, 0]', 'y_pred = [0, 0]'],
        result:
          `{'tp': 0, 'tn': 2, 'fp': 0, 'fn': 0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'accuracy': 1.0}`,
      },
    ],
    hint: [
      'Count `tp`, `tn`, `fp`, and `fn` in one pass over paired labels.',
      'Precision and recall each have their own denominator; guard each one separately.',
      'Compute `f1` from precision and recall, but return `0.0` if both are zero.',
      'Validate that both inputs are 1D, the same length, non-empty, and restricted to `0` or `1`.',
    ],
    solutionNotes: [
      'This is mostly a confusion-matrix exercise: once the four counts are correct, the derived metrics are straightforward ratios.',
      'The subtle part is the edge handling. Returning `0.0` for undefined metrics keeps the function predictable when there are no predicted positives or no actual positives.',
    ],
    solutionCode: `def _coerce_binary_labels(values, name):
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a 1D sequence of binary labels")

    try:
        items = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a 1D sequence of binary labels") from exc

    if not items:
        raise ValueError(f"{name} must not be empty")

    for item in items:
        if hasattr(item, "__iter__") and not isinstance(item, (str, bytes)):
            raise ValueError(f"{name} must be one-dimensional")
        if item not in (0, 1):
            raise ValueError(f"{name} must contain only 0 and 1")

    return items


def binary_classification_metrics(y_true, y_pred):
    y_true = _coerce_binary_labels(y_true, "y_true")
    y_pred = _coerce_binary_labels(y_pred, "y_pred")

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    tp = tn = fp = fn = 0
    for truth, pred in zip(y_true, y_pred):
        if truth == 1 and pred == 1:
            tp += 1
        elif truth == 0 and pred == 0:
            tn += 1
        elif truth == 0 and pred == 1:
            fp += 1
        else:
            fn += 1

    total = len(y_true)
    precision_den = tp + fp
    recall_den = tp + fn

    precision = tp / precision_den if precision_den else 0.0
    recall = tp / recall_den if recall_den else 0.0
    f1_den = precision + recall
    f1 = (2.0 * precision * recall / f1_den) if f1_den else 0.0
    accuracy = (tp + tn) / total

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }`,
    starterCode: `def binary_classification_metrics(y_true, y_pred):
    # TODO:
    # 1. Validate the inputs.
    # 2. Count tp, tn, fp, and fn.
    # 3. Compute precision, recall, f1, and accuracy with zero-division guards.
    raise NotImplementedError("Implement binary_classification_metrics")

sample_true = [1, 0, 1, 0]
sample_pred = [1, 0, 0, 1]

print(binary_classification_metrics(sample_true, sample_pred))`,
    tags: ['Classification', 'Metrics', 'Confusion Matrix'],
  },
  {
    id: 'pairwise-cosine-similarity',
    order: 5,
    title: 'Pairwise cosine similarity',
    difficulty: 'Easy',
    summary:
      'Compute an (N, M) cosine-similarity matrix between two batches of vectors with zero-norm safeguards.',
    prompt: [
      'Write `pairwise_cosine_similarity(x, y)` so it returns the pairwise cosine similarity between every row of `x` and every row of `y`.',
      'Treat this like an interview question: validate the shapes, use a vectorized implementation, and make sure rows with zero norm produce `0.0` similarities instead of `nan` or `inf`.',
    ],
    signature: `def pairwise_cosine_similarity(x, y):
    ...`,
    requirements: [
      '`x` is a 2D array or list with shape `(N, D)`.',
      '`y` is a 2D array or list with shape `(M, D)`.',
      'Return an `(N, M)` matrix of cosine similarities.',
      'If any row in either input has zero norm, all similarities involving that row must be `0.0`.',
      'Raise `ValueError` on invalid shapes.',
    ],
    examples: [
      {
        label: 'Example 1',
        lines: ['x = [[1, 0], [0, 0]]', 'y = [[1, 0], [1, 1]]'],
        result: '[[1.0, 0.70711], [0.0, 0.0]]',
      },
      {
        label: 'Example 2',
        lines: ['x = [[0, 1]]', 'y = [[0, 0], [0, 1]]'],
        result: '[[0.0, 1.0]]',
      },
    ],
    hint: [
      'Compute the numerator with `x @ y.T`.',
      'Compute row norms once, then broadcast them into an `(N, M)` denominator.',
      'Use `np.divide(..., where=denominator != 0)` so zero-norm rows become `0.0` instead of raising warnings.',
      'Validate that both inputs are 2D and share the same feature dimension before doing any math.',
    ],
    solutionNotes: [
      'Cosine similarity is just a dot product divided by the product of L2 norms. Once the row norms are in hand, the whole pairwise matrix can be computed with broadcasting.',
      'The key edge case is a zero vector: its norm is zero, so any similarity involving that row is undefined. Filling those positions with `0.0` keeps the result stable and matches the prompt.',
    ],
    solutionCode: `import numpy as np

def pairwise_cosine_similarity(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2D arrays")
    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have the same feature dimension")
    if x.shape[1] == 0:
        raise ValueError("feature dimension must be positive")

    x_norms = np.linalg.norm(x, axis=1)
    y_norms = np.linalg.norm(y, axis=1)
    similarities = x @ y.T
    denominator = x_norms[:, None] * y_norms[None, :]

    return np.divide(
        similarities,
        denominator,
        out=np.zeros_like(similarities),
        where=denominator != 0,
    )`,
    starterCode: `import numpy as np

def pairwise_cosine_similarity(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    # TODO:
    # 1. Validate that x and y are 2D and share the same feature dimension.
    # 2. Compute pairwise cosine similarities with zero-norm rows mapped to 0.0.
    raise NotImplementedError("Implement pairwise_cosine_similarity")

sample_x = np.array([[1.0, 0.0], [0.0, 0.0]])
sample_y = np.array([[1.0, 0.0], [1.0, 1.0]])

print(pairwise_cosine_similarity(sample_x, sample_y))`,
    packages: ['numpy'],
    tags: ['NumPy', 'Linear Algebra', 'Vectorization'],
  },
  {
    id: 'top-k-accuracy',
    order: 6,
    title: 'Top-k accuracy',
    difficulty: 'Easy',
    summary:
      'Compute the fraction of examples whose true label appears among the top k logits in each row.',
    prompt: [
      'Write `top_k_accuracy(logits, labels, k)` so it returns the fraction of examples where the true label is among the top `k` logits.',
      'Treat this like an interview question: validate the inputs, use NumPy for the ranking logic, and accept NumPy’s default ordering behavior when scores tie.',
    ],
    signature: `def top_k_accuracy(logits, labels, k):
    ...`,
    requirements: [
      '`logits` is a 2D array or list of shape `(N, C)`.',
      '`labels` is a 1D array or list of shape `(N,)` with integer class ids.',
      '`k` is a positive integer.',
      'Return the fraction of examples whose true label is in the top `k` logits for that row.',
      'Raise `ValueError` on invalid inputs.',
    ],
    examples: [
      {
        label: 'Example 1',
        lines: [
          'logits = [[0.1, 0.9, 0.2], [3.0, 1.0, 2.0]]',
          'labels = [1, 2]',
          'k = 1',
        ],
        result: '0.5',
      },
      {
        label: 'Example 2',
        lines: [
          'logits = [[0.1, 0.9, 0.2], [3.0, 1.0, 2.0]]',
          'labels = [1, 2]',
          'k = 2',
        ],
        result: '1.0',
      },
    ],
    hint: [
      'Sort each row in descending order and take the first `k` class indices.',
      'A vectorized comparison against `labels[:, None]` makes it easy to test membership in the top-k set.',
      'Use `np.mean` on the boolean correctness mask to turn per-example hits into a fraction.',
      'Validate that `logits` is 2D, `labels` is 1D, the batch sizes match, the labels are in range, and `k` is positive.',
    ],
    solutionNotes: [
      'The implementation is straightforward once the inputs are validated: rank each row from largest to smallest, take the first `k` class ids, and check whether the true label appears in that slice.',
      'Because the check is fully vectorized, the result is a simple mean over a boolean mask, which keeps the code short and easy to read.',
    ],
    solutionCode: `import numpy as np

def top_k_accuracy(logits, labels, k):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)

    if logits.ndim != 2:
        raise ValueError("logits must be a 2D array of shape (N, C)")
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array of shape (N,)")
    if logits.shape[0] == 0:
        raise ValueError("logits must contain at least one sample")
    if logits.shape[1] == 0:
        raise ValueError("logits must contain at least one class")
    if labels.shape[0] != logits.shape[0]:
        raise ValueError("labels must have the same batch size as logits")
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError("k must be a positive integer")
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must contain integer class ids")
    if np.any(labels < 0) or np.any(labels >= logits.shape[1]):
        raise ValueError("labels contain out-of-range class ids")

    top_k = min(int(k), logits.shape[1])
    ranked = np.argsort(-logits, axis=1)[:, :top_k]
    hits = np.any(ranked == labels[:, None], axis=1)
    return float(np.mean(hits))`,
    starterCode: `import numpy as np

def top_k_accuracy(logits, labels, k):
    logits = np.asarray(logits)
    labels = np.asarray(labels)

    # TODO:
    # 1. Validate shapes, label values, and that k is positive.
    # 2. Rank each row of logits and check whether the true label appears in the top k.
    raise NotImplementedError("Implement top_k_accuracy")

sample_logits = np.array([[0.1, 0.9, 0.2], [3.0, 1.0, 2.0]])
sample_labels = np.array([1, 2])

print(top_k_accuracy(sample_logits, sample_labels, k=1))`,
    packages: ['numpy'],
    tags: ['NumPy', 'Classification', 'Metrics'],
  },
  {
    id: 'iou-matrix',
    order: 7,
    title: 'IoU matrix',
    difficulty: 'Medium',
    summary:
      'Compute a pairwise intersection-over-union matrix between two sets of bounding boxes.',
    prompt: [
      'Write `box_iou_matrix(boxes1, boxes2)` so it returns the pairwise IoU between every box in `boxes1` and every box in `boxes2`.',
      'Treat this like an interview question: validate the shapes, use a vectorized implementation, and raise `ValueError` for malformed boxes.',
    ],
    signature: `def box_iou_matrix(boxes1, boxes2):
    ...`,
    requirements: [
      '`boxes1` is an `(N, 4)` array or list.',
      '`boxes2` is an `(M, 4)` array or list.',
      'Each box is in `[x1, y1, x2, y2]` format.',
      'Return an `(N, M)` matrix of IoU values.',
      'Raise `ValueError` for invalid boxes or invalid shapes.',
    ],
    examples: [
      {
        label: 'Example 1',
        lines: [
          'boxes1 = [[0, 0, 2, 2], [0, 0, 1, 1]]',
          'boxes2 = [[1, 1, 3, 3], [0, 0, 2, 2]]',
        ],
        result: '[[0.14286, 1.0], [0.0, 0.25]]',
      },
      {
        label: 'Example 2',
        lines: ['boxes1 = [[0, 0, 0, 1]]', 'boxes2 = [[0, 0, 1, 1]]'],
        result: '[[0.0]]',
      },
    ],
    hint: [
      'Broadcast `boxes1` against `boxes2` to compute the overlap corners in one shot.',
      'Intersection width and height should be clamped at `0.0` so non-overlapping boxes contribute zero area.',
      'Compute areas once, then divide intersection by union with a zero-safe `np.divide`.',
      'Validate that each box has `x2 >= x1` and `y2 >= y1` before computing anything else.',
    ],
    solutionNotes: [
      'The main trick is to form all pairwise overlap rectangles with broadcasting, then compute intersection areas, box areas, and union areas from those tensors.',
      'Once the pairwise union is known, `np.divide` with a zero-filled output array keeps the implementation numerically stable and handles degenerate boxes cleanly.',
    ],
    solutionCode: `import numpy as np

def _validate_boxes(boxes, name):
    boxes = np.asarray(boxes, dtype=np.float64)

    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"{name} must have shape (N, 4)")
    if np.any(boxes[:, 2] < boxes[:, 0]) or np.any(boxes[:, 3] < boxes[:, 1]):
        raise ValueError(f"{name} contains invalid boxes")

    return boxes


def box_iou_matrix(boxes1, boxes2):
    boxes1 = _validate_boxes(boxes1, "boxes1")
    boxes2 = _validate_boxes(boxes2, "boxes2")

    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter_area

    return np.divide(
        inter_area,
        union,
        out=np.zeros_like(inter_area),
        where=union > 0,
    )`,
    starterCode: `import numpy as np

def box_iou_matrix(boxes1, boxes2):
    boxes1 = np.asarray(boxes1)
    boxes2 = np.asarray(boxes2)

    # TODO:
    # 1. Validate the shapes and reject malformed boxes.
    # 2. Compute the pairwise intersection, union, and IoU matrices.
    raise NotImplementedError("Implement box_iou_matrix")

sample_boxes1 = np.array([[0, 0, 2, 2], [0, 0, 1, 1]])
sample_boxes2 = np.array([[1, 1, 3, 3], [0, 0, 2, 2]])

print(box_iou_matrix(sample_boxes1, sample_boxes2))`,
    packages: ['numpy'],
    tags: ['NumPy', 'Computer Vision', 'Bounding Boxes'],
  },
  {
    id: 'nearest-centroid-classifier',
    order: 8,
    title: 'Nearest centroid classifier',
    difficulty: 'Easy',
    summary:
      'Compute one centroid per class and predict each test point by the nearest Euclidean centroid.',
    prompt: [
      'Write `nearest_centroid_predict(train_X, train_y, test_X)` so it returns a 1D array of predicted class labels for `test_X`.',
      'Compute one centroid per class from `train_X`, then classify each test point by the nearest centroid using Euclidean distance. If distances tie, choose the smaller class label.',
    ],
    signature: `def nearest_centroid_predict(train_X, train_y, test_X):
    ...`,
    requirements: [
      '`train_X` is an `(N, D)` array or list.',
      '`train_y` is a 1D array or list of length `N` containing class labels.',
      '`test_X` is an `(M, D)` array or list.',
      'Return predictions as a 1D array.',
      'If distances tie, choose the smaller class label.',
      'Raise `ValueError` for invalid shapes or invalid labels.',
    ],
    examples: [
      {
        label: 'Example 1',
        lines: [
          'train_X = [[0.0], [2.0], [10.0], [12.0]]',
          'train_y = [0, 0, 1, 1]',
          'test_X = [[0.0], [6.0], [12.0]]',
        ],
        result: '[0, 0, 1]',
      },
      {
        label: 'Example 2',
        lines: [
          'train_X = [[1, 0], [0, 1], [3, 3], [4, 4], [10, 10]]',
          'train_y = [0, 0, 1, 1, 2]',
          'test_X = [[1, 1], [9, 9]]',
        ],
        result: '[0, 2]',
      },
    ],
    hint: [
      'Group `train_X` by label, then take the mean of each group to form the centroids.',
      'Sort the unique labels so that ties fall to the smaller class label when you take an argmin.',
      'Broadcast `test_X` against the centroid matrix to compute all distances at once.',
      'Use squared Euclidean distance to avoid an unnecessary square root.',
    ],
    solutionNotes: [
      'The nearest-centroid rule compresses each class into its mean feature vector, then assigns each test point to the closest mean.',
      'Squared Euclidean distance preserves the same ordering as Euclidean distance, and keeping the class labels sorted makes the tie-breaking rule deterministic.',
    ],
    solutionCode: `import numpy as np

def nearest_centroid_predict(train_X, train_y, test_X):
    train_X = np.asarray(train_X, dtype=np.float64)
    train_y = np.asarray(train_y)
    test_X = np.asarray(test_X, dtype=np.float64)

    if train_X.ndim != 2:
        raise ValueError("train_X must be a 2D array of shape (N, D)")
    if train_y.ndim != 1:
        raise ValueError("train_y must be a 1D array of shape (N,)")
    if test_X.ndim != 2:
        raise ValueError("test_X must be a 2D array of shape (M, D)")
    if train_X.shape[0] == 0:
        raise ValueError("train_X must contain at least one sample")
    if train_X.shape[0] != train_y.shape[0]:
        raise ValueError("train_X and train_y must have the same number of samples")
    if train_X.shape[1] != test_X.shape[1]:
        raise ValueError("train_X and test_X must have the same feature dimension")
    if not np.issubdtype(train_y.dtype, np.integer):
        raise ValueError("train_y must contain integer class labels")

    labels = np.unique(train_y)
    if labels.size == 0:
        raise ValueError("train_y must contain at least one class")

    centroids = np.vstack([train_X[train_y == label].mean(axis=0) for label in labels])
    deltas = test_X[:, None, :] - centroids[None, :, :]
    squared_distances = np.sum(deltas * deltas, axis=2)
    nearest_indices = np.argmin(squared_distances, axis=1)
    return labels[nearest_indices]`,
    starterCode: `import numpy as np

def nearest_centroid_predict(train_X, train_y, test_X):
    train_X = np.asarray(train_X)
    train_y = np.asarray(train_y)
    test_X = np.asarray(test_X)

    # TODO:
    # 1. Validate the shapes and labels.
    # 2. Compute one centroid per class from train_X.
    # 3. Predict each test point by the nearest centroid.
    raise NotImplementedError("Implement nearest_centroid_predict")

sample_train_X = np.array([[0.0], [2.0], [10.0], [12.0]])
sample_train_y = np.array([0, 0, 1, 1])
sample_test_X = np.array([[0.0], [6.0], [12.0]])

print(nearest_centroid_predict(sample_train_X, sample_train_y, sample_test_X))`,
    packages: ['numpy'],
    tags: ['NumPy', 'Classification', 'Centroids'],
  },
  {
    id: 'temperature-scaling-of-logits',
    order: 9,
    title: 'Temperature scaling of logits',
    difficulty: 'Medium',
    summary:
      'Convert logits into numerically stable softmax probabilities after dividing by a positive temperature.',
    prompt: [
      'Write `temperature_scaled_probs(logits, temperature)` so it returns softmax probabilities after scaling `logits` by `temperature`.',
      'Use a numerically stable implementation, validate the inputs, and make sure each row of the output sums to `1`.',
    ],
    signature: `def temperature_scaled_probs(logits, temperature):
    ...`,
    requirements: [
      '`logits` is an `(N, C)` array or list.',
      '`temperature` is a positive float.',
      'Return an `(N, C)` array of probabilities.',
      'Divide logits by `temperature` before applying softmax.',
      'Use a numerically stable implementation.',
      'Raise `ValueError` for invalid inputs.',
    ],
    examples: [
      {
        label: 'Example 1',
        lines: ['logits = [[1000.0, 1001.0, 1002.0]]', 'temperature = 1.0'],
        result: '[[0.09003, 0.24473, 0.66524]]',
      },
      {
        label: 'Example 2',
        lines: [
          'logits = [[2.0, 0.0], [1.0, 1.0]]',
          'temperature = 2.0',
        ],
        result: '[[0.73106, 0.26894], [0.5, 0.5]]',
      },
    ],
    hint: [
      'Divide the logits by the temperature before you do anything else.',
      'Subtract the maximum value in each row before exponentiating to keep the softmax stable.',
      'Normalize with the row-wise sum and rely on broadcasting for the final division.',
      'Reject non-2D logits and any temperature that is not a positive scalar.',
    ],
    solutionNotes: [
      'Temperature scaling is just softmax on the logits after rescaling them by a positive constant. The key implementation detail is to subtract the row maximum after scaling so the exponentials never blow up.',
      'Once the shifted logits are exponentiated, each row is normalized by its own sum, which gives a valid probability distribution that still sums to `1`.',
    ],
    solutionCode: `import numpy as np

def temperature_scaled_probs(logits, temperature):
    logits = np.asarray(logits, dtype=np.float64)

    if isinstance(temperature, (bool, np.bool_)):
        raise ValueError("temperature must be a positive float")
    try:
        temperature = float(temperature)
    except (TypeError, ValueError):
        raise ValueError("temperature must be a positive float")

    if logits.ndim != 2:
        raise ValueError("logits must be a 2D array of shape (N, C)")
    if logits.shape[0] == 0:
        raise ValueError("logits must contain at least one sample")
    if logits.shape[1] == 0:
        raise ValueError("logits must contain at least one class")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a positive float")

    scaled_logits = logits / temperature
    shifted = scaled_logits - np.max(scaled_logits, axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)`,
    starterCode: `import numpy as np

def temperature_scaled_probs(logits, temperature):
    logits = np.asarray(logits)

    # TODO:
    # 1. Validate that logits is 2D and temperature is a positive scalar.
    # 2. Divide by temperature, apply a numerically stable softmax, and return probabilities.
    raise NotImplementedError("Implement temperature_scaled_probs")

sample_logits = np.array([[1000.0, 1001.0, 1002.0]])
sample_temperature = 1.0

print(temperature_scaled_probs(sample_logits, sample_temperature))`,
    packages: ['numpy'],
    tags: ['NumPy', 'Classification', 'Calibration'],
  },
] as const;
