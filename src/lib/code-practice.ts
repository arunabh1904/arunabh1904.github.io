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
] as const;
