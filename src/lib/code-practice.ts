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
  'Interview-style PyTorch and NumPy exercises with runnable starter code, focused hints, and hidden solutions.';

const PYTORCH_AND_NUMPY_PACKAGES = ['torch', 'numpy'] as const;

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
      'Implement a numerically stable batch softmax cross-entropy loss in PyTorch with proper input validation.',
    prompt: [
      'Write `softmax_cross_entropy(logits, labels)` so it returns the mean cross-entropy loss across a batch.',
      'Treat this like an interview question: keep the implementation concise, validate the inputs, and avoid numerical overflow when computing the softmax terms.',
    ],
    signature: `def softmax_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    ...`,
    requirements: [
      '`logits` is a 2D PyTorch tensor of shape `(N, C)`.',
      '`labels` is a 1D PyTorch tensor of shape `(N,)` with integer class ids in `[0, C - 1]`.',
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
      'Use `torch.arange(N)` to gather the logit for the correct class in each row.',
      'Compute the loss as `log(sum(exp(shifted))) - shifted[row, label]`, then average across the batch.',
      'Validate `ndim`, matching batch size, integer labels, non-empty shapes, and label range.',
    ],
    solutionNotes: [
      'The stable trick is to shift each row by its maximum value before applying `exp`, which preserves the softmax probabilities while avoiding overflow.',
      'The solution spells out the row-wise `amax`, `exp`, `sum`, and `log` operations so the stable cross-entropy calculation stays visible instead of being hidden behind a softmax helper.',
    ],
    solutionCode: `from __future__ import annotations

import torch

def _validate_classification_inputs(logits, labels):
    # Convert compatible inputs once so the remaining checks use predictable tensor types.
    logits = torch.as_tensor(logits, dtype=torch.float64)
    labels = torch.as_tensor(labels)

    # Cross-entropy needs one row of class scores and one target index per row.
    if logits.ndim != 2:
        raise ValueError("logits must have shape (N, C)")
    if labels.ndim != 1:
        raise ValueError("labels must have shape (N,)")
    batch_size, num_classes = logits.shape
    # Empty batches or class dimensions would make the reduction or indexing undefined.
    if batch_size == 0 or num_classes == 0:
        raise ValueError("logits must have positive dimensions")
    if labels.shape[0] != batch_size:
        raise ValueError("labels must match the logits batch size")
    # Class labels become tensor indices below, so fractional values are invalid.
    if torch.is_floating_point(labels):
        raise ValueError("labels must contain integer class ids")
    # Reject NaN and infinity before log-softmax propagates them through the loss.
    if not bool(torch.all(torch.isfinite(logits))):
        raise ValueError("logits must contain only finite values")

    # Use the canonical index dtype before gathering one target from each row.
    labels = torch.as_tensor(labels, dtype=torch.long)
    # Every target must refer to an existing column in its row of logits.
    if bool(torch.any(labels < 0)) or bool(torch.any(labels >= num_classes)):
        raise ValueError("labels contain out-of-range class ids")
    return logits, labels

def softmax_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    # Validation also coerces compatible lists or tensors into the expected dtypes.
    logits, labels = _validate_classification_inputs(logits, labels)

    # Build stable cross-entropy from reductions and elementwise tensor primitives.
    # Take one maximum per row and keep its column axis so it broadcasts across every class score.
    row_maxes = torch.amax(logits, dim=1, keepdim=True)
    # Shifting all logits in a row by the same value preserves the softmax probabilities.
    shifted_logits = logits - row_maxes
    # The shift keeps every exponent at most one, avoiding overflow from large raw logits.
    exponentiated = torch.exp(shifted_logits)
    # Sum across classes to form the denominator of each row's softmax.
    normalizers = torch.sum(exponentiated, dim=1)
    # Pair every row number with its target class so advanced indexing selects N target logits.
    row_indices = torch.arange(logits.shape[0], dtype=torch.long)
    target_logits = shifted_logits[row_indices, labels]
    # -log(p_target) simplifies to log(denominator) minus the shifted target logit.
    losses = torch.log(normalizers) - target_logits
    # The requested scalar loss is the mean of those per-example losses.
    return torch.mean(losses)`,
    starterCode: `from __future__ import annotations

import torch

def softmax_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    # TODO:
    # 1. Validate the batch/class dimensions and integer label range.
    # 2. Subtract a row maximum, exponentiate, and sum the shifted logits.
    # 3. Form log(sum(exp(...))) minus the target logit, then average the loss.
    raise NotImplementedError("Implement softmax_cross_entropy")

sample_logits = torch.tensor([[2.0, 1.0, 0.1]], dtype=torch.float64)
sample_labels = torch.tensor([0], dtype=torch.long)

print(f"{softmax_cross_entropy(sample_logits, sample_labels).item():.5f}")`,
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'Numerical Stability', 'Interview Practice'],
  },
  {
    id: 'non-maximum-suppression',
    order: 2,
    title: 'Non-maximum suppression',
    difficulty: 'Medium',
    summary:
      'Implement PyTorch-based non-maximum suppression with deterministic score tie-breaking and invalid-box checks.',
    prompt: [
      'Write `nms(boxes, scores, iou_threshold)` so it returns the indices of the boxes kept after non-maximum suppression.',
      'Process boxes in descending score order, break ties by smaller original index, suppress only boxes whose IoU with a kept box is strictly greater than `iou_threshold`, and raise `ValueError` when a box has `x2 < x1` or `y2 < y1`.',
    ],
    signature: `def nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
) -> list[int]:
    ...`,
    requirements: [
      '`boxes` is an `(N, 4)` PyTorch tensor of `[x1, y1, x2, y2]`.',
      '`scores` is a PyTorch tensor of shape `(N,)`.',
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
    solutionCode: `from __future__ import annotations

import torch

def _pairwise_iou(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    # Intersect one selected box with every remaining candidate coordinate by coordinate.
    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])

    # Clamp non-overlapping widths and heights to zero before computing their area.
    inter_area = torch.clamp(x2 - x1, min=0.0) * torch.clamp(y2 - y1, min=0.0)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    # IoU is shared area divided by the area covered by either box.
    union = box_area + boxes_area - inter_area

    # Avoid division by zero for degenerate boxes while preserving exact zero IoU there.
    safe_union = torch.where(union > 0.0, union, torch.ones_like(union))
    return torch.where(union > 0.0, inter_area / safe_union, torch.zeros_like(union))

def nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
) -> list[int]:
    # Use one floating dtype for geometry and scores, even when callers pass Python lists.
    boxes = torch.as_tensor(boxes, dtype=torch.float64)
    scores = torch.as_tensor(scores, dtype=torch.float64)

    # Validate the contract before accessing coordinate columns or candidate indices.
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape (N, 4)")
    if scores.ndim != 1 or scores.shape[0] != boxes.shape[0]:
        raise ValueError("scores must have shape (N,)")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    if not bool(torch.all(torch.isfinite(boxes))) or not bool(torch.all(torch.isfinite(scores))):
        raise ValueError("boxes and scores must contain only finite values")
    if bool(torch.any(boxes[:, 2] < boxes[:, 0])) or bool(torch.any(boxes[:, 3] < boxes[:, 1])):
        raise ValueError("boxes must satisfy x2 >= x1 and y2 >= y1")

    # Sorting in Python makes the tie-break rule explicit and reviewable.
    order = sorted(range(boxes.shape[0]), key=lambda i: (-float(scores[i].item()), i))
    keep: list[int] = []

    # Greedily keep the highest-scoring candidate that has not been suppressed.
    while order:
        current = order.pop(0)
        keep.append(current)
        if not order:
            break

        # Compare the selected box with every remaining candidate in one vectorized operation.
        remaining = torch.as_tensor(order, dtype=torch.long)
        ious = _pairwise_iou(boxes[current], boxes[remaining])
        # Equal-threshold IoUs survive: only IoU values strictly above the threshold are suppressed.
        order = [int(index) for index in remaining[ious <= iou_threshold].tolist()]

    return keep`,
    starterCode: `from __future__ import annotations

import torch

def nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
) -> list[int]:
    # TODO:
    # 1. Validate shape, finite values, box geometry, and threshold.
    # 2. Sort by descending score with the original index as a deterministic tie-break.
    # 3. Keep the first candidate and suppress only boxes with IoU > iou_threshold.
    raise NotImplementedError("Implement nms")

sample_boxes = torch.tensor([
    [0.0, 0.0, 2.0, 2.0],
    [0.5, 0.5, 2.5, 2.5],
    [5.0, 5.0, 7.0, 7.0],
], dtype=torch.float64)
sample_scores = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float64)

print(nms(sample_boxes, sample_scores, iou_threshold=0.3))`,
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'Computer Vision', 'Greedy'],
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
    signature: `def make_causal_attention_mask(
    seq_lens: torch.Tensor,
    max_len: int | None = None,
) -> torch.Tensor:
    ...`,
    requirements: [
      '`seq_lens` is a 1D list or PyTorch tensor of length `B`.',
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
      'Compare query and key positions (`query_index >= key_index`) to build the causal lower triangle once.',
      'Create a `(B, T)` validity mask from `seq_lens`, then broadcast it across rows and columns.',
      'Multiply the causal triangle by the validity masks so padded rows and columns stay zero.',
      'Validate the rank of `seq_lens`, integer lengths, non-negative values, and `max_len` when it is provided.',
    ],
    solutionNotes: [
      'The problem is really two masks multiplied together: the causal rule (`j <= i`) and the per-example validity rule (`i, j < seq_len[b]`).',
      'Comparing query and key indices constructs that lower-triangular template from primitives, and broadcasting it against the batch-wise validity mask gives the full `(B, T, T)` answer without explicit Python loops.',
    ],
    solutionCode: `from __future__ import annotations

import torch

def make_causal_attention_mask(
    seq_lens: torch.Tensor,
    max_len: int | None = None,
) -> torch.Tensor:
    # Accept lists as well as tensors, then inspect the original integer-valued lengths.
    seq_lens = torch.as_tensor(seq_lens)

    # One non-empty sequence length is required for every batch element.
    if seq_lens.ndim != 1 or seq_lens.shape[0] == 0:
        raise ValueError("seq_lens must be a non-empty 1D tensor")
    if torch.is_floating_point(seq_lens):
        raise ValueError("seq_lens must contain integers")
    if bool(torch.any(seq_lens < 0)):
        raise ValueError("seq_lens must be non-negative")
    # bool is rejected explicitly because it is a Python subclass of int.
    if max_len is not None and (isinstance(max_len, bool) or not isinstance(max_len, int)):
        raise ValueError("max_len must be an integer or None")
    if max_len is not None and max_len < 0:
        raise ValueError("max_len must be non-negative")

    # Long indices work naturally in the broadcast comparison below.
    seq_lens = torch.as_tensor(seq_lens, dtype=torch.long)
    # The mask must fit both the longest real sequence and any requested padded width.
    length = int(torch.amax(seq_lens).item())
    if max_len is not None:
        length = max(length, max_len)

    # One shared position vector supplies both the query and key coordinates.
    positions = torch.arange(length, dtype=torch.long)
    # Broadcast positions against each batch length to mark real tokens and padding.
    valid = positions[None, :] < seq_lens[:, None]
    # Comparing query rows against key columns gives the lower-triangular causal template.
    causal = torch.as_tensor(
        positions[:, None] >= positions[None, :],
        dtype=torch.int64,
    )

    # Intersect causal visibility with both the valid query and valid key positions.
    return causal[None, :, :] * valid[:, :, None] * valid[:, None, :]`,
    starterCode: `from __future__ import annotations

import torch

def make_causal_attention_mask(
    seq_lens: torch.Tensor,
    max_len: int | None = None,
) -> torch.Tensor:
    # TODO:
    # 1. Validate non-empty integer sequence lengths and optional max_len.
    # 2. Compare query and key positions to build one causal template.
    # 3. Broadcast per-example validity across query and key axes.
    raise NotImplementedError("Implement make_causal_attention_mask")

sample_seq_lens = torch.tensor([3, 1], dtype=torch.long)
print(make_causal_attention_mask(sample_seq_lens, max_len=4))`,
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'Attention Masks', 'Sequence Modeling'],
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
    signature: `def binary_classification_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> dict[str, int | float]:
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
    solutionCode: `from __future__ import annotations

import torch

def _coerce_binary_labels(values, name: str) -> torch.Tensor:
    # torch.as_tensor accepts ordinary sequences while preserving a useful validation error.
    try:
        labels = torch.as_tensor(values)
    except Exception as exc:
        raise ValueError(f"{name} must be a 1D sequence of binary labels") from exc

    # Metrics are defined over a non-empty, one-dimensional label vector.
    if labels.ndim != 1 or labels.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty 1D tensor")
    # Floating-point values cannot be treated as category ids.
    if torch.is_floating_point(labels):
        raise ValueError(f"{name} must contain integer labels")
    # The four confusion-matrix cases below assume exactly the labels 0 and 1.
    if not bool(torch.all((labels == 0) | (labels == 1))):
        raise ValueError(f"{name} must contain only 0 and 1")
    # Normalize the representation used by comparisons and output counts.
    return torch.as_tensor(labels, dtype=torch.long)

def binary_classification_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> dict[str, int | float]:
    # Validate each side independently before checking that their lengths agree.
    y_true = _coerce_binary_labels(y_true, "y_true")
    y_pred = _coerce_binary_labels(y_pred, "y_pred")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    # Each Boolean conjunction isolates one cell of the binary confusion matrix.
    tp = int(torch.sum((y_true == 1) & (y_pred == 1)).item())
    tn = int(torch.sum((y_true == 0) & (y_pred == 0)).item())
    fp = int(torch.sum((y_true == 0) & (y_pred == 1)).item())
    fn = int(torch.sum((y_true == 1) & (y_pred == 0)).item())

    # Keep the denominators explicit so the zero-positive edge cases are easy to see.
    precision_denominator = tp + fp
    recall_denominator = tp + fn
    precision = tp / precision_denominator if precision_denominator else 0.0
    recall = tp / recall_denominator if recall_denominator else 0.0
    f1_denominator = precision + recall
    f1 = 2.0 * precision * recall / f1_denominator if f1_denominator else 0.0

    # Return counts and derived rates together so callers can inspect the calculation.
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / y_true.shape[0],
    }`,
    starterCode: `from __future__ import annotations

import torch

def binary_classification_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> dict[str, int | float]:
    # TODO:
    # 1. Validate non-empty, one-dimensional binary label tensors.
    # 2. Count tp, tn, fp, and fn with boolean tensor operations.
    # 3. Compute zero-safe precision, recall, f1, and accuracy.
    raise NotImplementedError("Implement binary_classification_metrics")

sample_true = torch.tensor([1, 0, 1, 0], dtype=torch.long)
sample_pred = torch.tensor([1, 0, 0, 1], dtype=torch.long)

print(binary_classification_metrics(sample_true, sample_pred))`,
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'Classification', 'Metrics', 'Confusion Matrix'],
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
    signature: `def pairwise_cosine_similarity(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
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
      'Square each row, sum across its feature dimension, and take a square root before broadcasting the row norms.',
      'Use a denominator mask with `torch.where` so zero-norm rows become `0.0` instead of `nan` or `inf`.',
      'Validate that both inputs are 2D and share the same feature dimension before doing any math.',
    ],
    solutionNotes: [
      'Cosine similarity is just a dot product divided by the product of L2 norms. The solution constructs each norm from squared entries, a sum, and a square root before broadcasting the whole pairwise matrix.',
      'The key edge case is a zero vector: its norm is zero, so any similarity involving that row is undefined. Filling those positions with `0.0` keeps the result stable and matches the prompt.',
    ],
    solutionCode: `from __future__ import annotations

import torch

def pairwise_cosine_similarity(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    # Convert compatible inputs to one floating dtype for dot products and norms.
    x = torch.as_tensor(x, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)

    # Every row is a feature vector, and the two sets must share its width.
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2D tensors")
    if x.shape[1] != y.shape[1] or x.shape[1] == 0:
        raise ValueError("x and y must share a positive feature dimension")
    if not bool(torch.all(torch.isfinite(x))) or not bool(torch.all(torch.isfinite(y))):
        raise ValueError("x and y must contain only finite values")

    # Matrix multiplication computes every x-row dot every y-row at once: shape (N, M).
    numerator = torch.matmul(x, torch.transpose(y, 0, 1))
    # L2 norm is the square root of the sum of squared features for each x row.
    x_norms = torch.sqrt(torch.sum(x * x, dim=1))
    # Compute the same one-dimensional norm vector for y rows.
    y_norms = torch.sqrt(torch.sum(y * y, dim=1))
    # Add singleton axes so every x norm multiplies every y norm in the pairwise denominator.
    denominator = x_norms[:, None] * y_norms[None, :]
    # Avoid dividing by zero eagerly; the final where assigns those similarities to zero.
    safe_denominator = torch.where(
        denominator > 0.0,
        denominator,
        torch.ones_like(denominator),
    )

    # Zero vectors have no direction; define every similarity involving one as zero.
    return torch.where(
        denominator > 0.0,
        numerator / safe_denominator,
        torch.zeros_like(numerator),
    )`,
    starterCode: `from __future__ import annotations

import torch

def pairwise_cosine_similarity(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    # TODO:
    # 1. Validate two-dimensional inputs with the same feature dimension.
    # 2. Compute pairwise dot products, then form row norms with square, sum, and square root.
    # 3. Map similarities involving a zero-norm row to 0.0.
    raise NotImplementedError("Implement pairwise_cosine_similarity")

sample_x = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
sample_y = torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float64)

print(pairwise_cosine_similarity(sample_x, sample_y))`,
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'Linear Algebra', 'Vectorization'],
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
      'Treat this like an interview question: validate the inputs, use PyTorch for the ranking logic, and accept PyTorch’s default ordering behavior when scores tie.',
    ],
    signature: `def top_k_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: int,
) -> torch.Tensor:
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
      'Use `torch.mean` on the boolean correctness mask to turn per-example hits into a fraction.',
      'Validate that `logits` is 2D, `labels` is 1D, the batch sizes match, the labels are in range, and `k` is positive.',
    ],
    solutionNotes: [
      'The implementation is straightforward once the inputs are validated: rank each row from largest to smallest with a descending sort, take the first `k` class ids, and check whether the true label appears in that slice.',
      'Because the check is fully vectorized, the result is a simple mean over a boolean mask, which keeps the code short and easy to read.',
    ],
    solutionCode: `from __future__ import annotations

import torch

def top_k_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: int,
) -> torch.Tensor:
    # Scores need a floating dtype; labels stay uncast until their integer check passes.
    logits = torch.as_tensor(logits, dtype=torch.float64)
    labels = torch.as_tensor(labels)

    # There must be one non-empty score row for each target label.
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] == 0:
        raise ValueError("logits must have positive shape (N, C)")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("labels must have shape (N,)")
    # bool is an int subclass, but it is not a meaningful value of k.
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    if torch.is_floating_point(labels):
        raise ValueError("labels must contain integer class ids")
    # topk and the equality test use class ids as long tensor indices.
    labels = torch.as_tensor(labels, dtype=torch.long)
    if bool(torch.any(labels < 0)) or bool(torch.any(labels >= logits.shape[1])):
        raise ValueError("labels contain out-of-range class ids")

    # Asking for more than C classes is equivalent to asking for all C classes.
    top_k = min(k, logits.shape[1])
    # Sort each score row from highest to lowest and retain the resulting class ids.
    ranked = torch.argsort(logits, dim=1, descending=True)
    # The first top_k columns are the candidate predictions for each sample.
    candidate_indices = ranked[:, :top_k]
    # Compare each target against all candidates in its row, then collapse that class axis.
    hits = torch.any(candidate_indices == labels[:, None], dim=1)
    # Cast True/False to 1.0/0.0 so the mean is the fraction of hits.
    return torch.mean(torch.as_tensor(hits, dtype=torch.float64))`,
    starterCode: `from __future__ import annotations

import torch

def top_k_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: int,
) -> torch.Tensor:
    # TODO:
    # 1. Validate shapes, integer labels, label range, and positive k.
    # 2. Rank each row with a descending argsort and slice its first k indices.
    # 3. Reduce the per-example membership mask to a mean accuracy.
    raise NotImplementedError("Implement top_k_accuracy")

sample_logits = torch.tensor([[0.1, 0.9, 0.2], [3.0, 1.0, 2.0]], dtype=torch.float64)
sample_labels = torch.tensor([1, 2], dtype=torch.long)

print(top_k_accuracy(sample_logits, sample_labels, k=1).item())`,
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'Classification', 'Metrics'],
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
    signature: `def box_iou_matrix(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> torch.Tensor:
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
      'Compute areas once, then divide intersection by union with an explicit zero-area policy.',
      'Validate that each box has `x2 >= x1` and `y2 >= y1` before computing anything else.',
    ],
    solutionNotes: [
      'The main trick is to form all pairwise overlap rectangles with broadcasting, then compute intersection areas, box areas, and union areas from those tensors.',
      'Once the pairwise union is known, a `torch.where` denominator mask keeps the implementation stable and handles degenerate boxes cleanly.',
    ],
    solutionCode: `from __future__ import annotations

import torch

def _validate_boxes(boxes: torch.Tensor, name: str) -> torch.Tensor:
    # Convert list-like box coordinates into a numeric tensor before geometry checks.
    boxes = torch.as_tensor(boxes, dtype=torch.float64)
    # Each row is one [x1, y1, x2, y2] box.
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"{name} must have shape (N, 4)")
    if not bool(torch.all(torch.isfinite(boxes))):
        raise ValueError(f"{name} must contain only finite values")
    # The bottom-right corner may coincide with, but cannot precede, the top-left corner.
    if bool(torch.any(boxes[:, 2] < boxes[:, 0])) or bool(torch.any(boxes[:, 3] < boxes[:, 1])):
        raise ValueError(f"{name} contains invalid boxes")
    return boxes

def box_iou_matrix(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> torch.Tensor:
    # Validate both collections independently; their counts may differ.
    boxes1 = _validate_boxes(boxes1, "boxes1")
    boxes2 = _validate_boxes(boxes2, "boxes2")

    # Insert singleton axes so broadcasting evaluates every boxes1-by-boxes2 pair.
    x1 = torch.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = torch.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = torch.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = torch.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    # Clamp non-overlap to zero, then form the (N, M) intersection-area matrix.
    inter_area = torch.clamp(x2 - x1, min=0.0) * torch.clamp(y2 - y1, min=0.0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    # Broadcast the two area vectors and subtract the overlap once to obtain union.
    union = area1[:, None] + area2[None, :] - inter_area
    # Keep the division defined for degenerate boxes; the final where supplies their zero IoU.
    safe_union = torch.where(union > 0.0, union, torch.ones_like(union))

    # Degenerate boxes have zero area, so their IoU is defined as zero here.
    return torch.where(
        union > 0.0,
        inter_area / safe_union,
        torch.zeros_like(inter_area),
    )`,
    starterCode: `from __future__ import annotations

import torch

def box_iou_matrix(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> torch.Tensor:
    # TODO:
    # 1. Validate each box tensor has shape (N, 4) and valid coordinates.
    # 2. Broadcast the two sets to form every overlap rectangle.
    # 3. Divide intersection by union with an explicit zero-area policy.
    raise NotImplementedError("Implement box_iou_matrix")

sample_boxes1 = torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 1.0, 1.0]], dtype=torch.float64)
sample_boxes2 = torch.tensor([[1.0, 1.0, 3.0, 3.0], [0.0, 0.0, 2.0, 2.0]], dtype=torch.float64)

print(box_iou_matrix(sample_boxes1, sample_boxes2))`,
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'Computer Vision', 'Bounding Boxes'],
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
    signature: `def nearest_centroid_predict(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    test_X: torch.Tensor,
) -> torch.Tensor:
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
      'Group `train_X` by label, then divide each class-wise feature sum by its count to form the centroids.',
      'Sort the unique labels so that ties fall to the smaller class label when you take an argmin.',
      'Broadcast `test_X` against the centroid matrix to compute all distances at once.',
      'Use squared Euclidean distance to avoid an unnecessary square root.',
    ],
    solutionNotes: [
      'The nearest-centroid rule compresses each class into its feature sum divided by its count, then assigns each test point to the closest centroid.',
      'Squared Euclidean distance preserves the same ordering as Euclidean distance, and keeping the class labels sorted makes the tie-breaking rule deterministic.',
    ],
    solutionCode: `from __future__ import annotations

import torch

def nearest_centroid_predict(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    test_X: torch.Tensor,
) -> torch.Tensor:
    # Features are numeric vectors, while labels are validated before becoming integer indices.
    train_X = torch.as_tensor(train_X, dtype=torch.float64)
    train_y = torch.as_tensor(train_y)
    test_X = torch.as_tensor(test_X, dtype=torch.float64)

    # Training and test rows must be matrices with the same feature width.
    if train_X.ndim != 2 or test_X.ndim != 2:
        raise ValueError("train_X and test_X must be 2D tensors")
    if train_X.shape[0] == 0 or train_X.shape[1] == 0:
        raise ValueError("train_X must have positive shape")
    if train_y.ndim != 1 or train_y.shape[0] != train_X.shape[0]:
        raise ValueError("train_y must have shape (N,)")
    if test_X.shape[1] != train_X.shape[1]:
        raise ValueError("train_X and test_X must share the feature dimension")
    if torch.is_floating_point(train_y):
        raise ValueError("train_y must contain integer class labels")
    train_y = torch.as_tensor(train_y, dtype=torch.long)

    # Sorting labels makes argmin's first-index tie behavior prefer the smaller label.
    labels = torch.unique(train_y, sorted=True)
    if labels.shape[0] == 0:
        raise ValueError("train_y must contain at least one class")

    # Sorted labels make argmin's first-index behavior implement the tie-break rule.
    centroid_rows: list[torch.Tensor] = []
    for label in labels:
        # Select just the training rows assigned to this class.
        class_points = train_X[train_y == label]
        # Sum then divide by the number of points to make the class mean explicit.
        centroid_rows.append(torch.sum(class_points, dim=0) / class_points.shape[0])
    # Stack the per-class vectors into a centroid matrix aligned with sorted labels.
    centroids = torch.stack(centroid_rows)
    # Broadcast test samples against centroids to form every sample-to-class displacement.
    deltas = test_X[:, None, :] - centroids[None, :, :]
    # Squared distance preserves the ordering while avoiding an unnecessary square root.
    squared_distances = torch.sum(deltas * deltas, dim=2)
    nearest_indices = torch.argmin(squared_distances, dim=1)
    # Map the winning centroid index back to its original class label.
    return labels[nearest_indices]`,
    starterCode: `from __future__ import annotations

import torch

def nearest_centroid_predict(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    test_X: torch.Tensor,
) -> torch.Tensor:
    # TODO:
    # 1. Validate feature dimensions and integer labels.
    # 2. Build each sorted class centroid as its feature sum divided by its count.
    # 3. Broadcast squared distances and choose the nearest centroid.
    raise NotImplementedError("Implement nearest_centroid_predict")

sample_train_X = torch.tensor([[0.0], [2.0], [10.0], [12.0]], dtype=torch.float64)
sample_train_y = torch.tensor([0, 0, 1, 1], dtype=torch.long)
sample_test_X = torch.tensor([[0.0], [6.0], [12.0]], dtype=torch.float64)

print(nearest_centroid_predict(sample_train_X, sample_train_y, sample_test_X))`,
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'Classification', 'Centroids'],
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
    signature: `def temperature_scaled_probs(
    logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
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
    solutionCode: `from __future__ import annotations

import math
import torch

def temperature_scaled_probs(
    logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    # Use a numeric dtype that can represent fractional scores and probabilities.
    logits = torch.as_tensor(logits, dtype=torch.float64)
    # Temperature scaling operates independently on each non-empty row of classes.
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] == 0:
        raise ValueError("logits must have positive shape (N, C)")
    if not bool(torch.all(torch.isfinite(logits))):
        raise ValueError("logits must contain only finite values")
    # bool is an int subclass, but accepting it would silently make an invalid temperature.
    if isinstance(temperature, bool):
        raise ValueError("temperature must be a positive finite scalar")

    # Accept scalar-like values while producing one consistent error for invalid inputs.
    try:
        temperature = float(temperature)
    except (TypeError, ValueError) as exc:
        raise ValueError("temperature must be a positive finite scalar") from exc
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be a positive finite scalar")

    # Temperature rescales score gaps before they are normalized into probabilities.
    scaled_logits = logits / temperature
    # Subtract each row maximum so the largest exponent is one rather than an overflowing value.
    shifted_logits = scaled_logits - torch.amax(scaled_logits, dim=1, keepdim=True)
    # Exponentiate the stable scores to obtain unnormalized positive weights.
    exponentiated = torch.exp(shifted_logits)
    # Keep the class axis so each row's denominator broadcasts during division.
    normalizers = torch.sum(exponentiated, dim=1, keepdim=True)
    # Dividing by the row total turns the unnormalized weights into probabilities.
    return exponentiated / normalizers`,
    starterCode: `from __future__ import annotations

import torch

def temperature_scaled_probs(
    logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    # TODO:
    # 1. Validate a positive finite temperature and finite 2D logits.
    # 2. Divide logits by temperature.
    # 3. Subtract the row max, exponentiate, sum each row, and normalize.
    raise NotImplementedError("Implement temperature_scaled_probs")

sample_logits = torch.tensor([[1000.0, 1001.0, 1002.0]], dtype=torch.float64)
print(temperature_scaled_probs(sample_logits, temperature=1.0))`,
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'Classification', 'Calibration'],
  },
  {
    id: 'sinusoidal-positional-encoding',
    order: 10,
    title: 'Sinusoidal positional encoding',
    difficulty: 'Medium',
    summary:
      'Build the classic sine-and-cosine positional encoding matrix with one frequency pair per even/odd column pair.',
    prompt: [
      'Write `sinusoidal_positional_encoding(length, dim)` so it returns a `(length, dim)` array of sinusoidal positional encodings.',
      'Use the standard Transformer formulas: even columns use `sin(pos / 10000^(2k/dim))` and odd columns use `cos(pos / 10000^(2k/dim))`. If `dim` is odd, the final column should use the even-column formula for its slot.',
    ],
    signature: `def sinusoidal_positional_encoding(length: int, dim: int) -> torch.Tensor:
    ...`,
    requirements: [
      '`length` is a positive integer.',
      '`dim` is a positive integer.',
      'Return an array of shape `(length, dim)`.',
      'For even columns `2k`, use `sin(pos / 10000^(2k/dim))`.',
      'For odd columns `2k+1`, use `cos(pos / 10000^(2k/dim))`.',
      'Raise `ValueError` on invalid input.',
    ],
    examples: [
      {
        label: 'Example 1',
        lines: ['length = 3', 'dim = 4'],
        result:
          '[[0.0, 1.0, 0.0, 1.0], [0.84147, 0.54030, 0.01000, 0.99995], [0.90930, -0.41615, 0.02000, 0.99980]]',
      },
      {
        label: 'Example 2',
        lines: ['length = 2', 'dim = 3'],
        result: '[[0.0, 1.0, 0.0], [0.84147, 0.54030, 0.00215]]',
      },
    ],
    hint: [
      'Build a column index vector `0..dim-1`, then reuse the same frequency for each even/odd pair.',
      'Broadcast a position vector of shape `(length, 1)` against the per-column frequency vector.',
      'Fill even columns with `torch.sin` and odd columns with `torch.cos` after computing the shared angles.',
      'If `dim` is odd, the last column still belongs to the even-column branch.',
    ],
    solutionNotes: [
      'Sinusoidal positional encoding is just a deterministic lookup table: each position gets a vector of sines and cosines at frequencies that decay geometrically across the embedding dimension.',
      'The implementation is compact if you compute one denominator per column pair and then broadcast positions across those frequencies. That also makes the odd-dimension case work naturally, because the final column is just the next even slot.',
    ],
    solutionCode: `from __future__ import annotations

import torch

def sinusoidal_positional_encoding(length: int, dim: int) -> torch.Tensor:
    # Both values determine tensor dimensions, so reject bool and non-positive integers early.
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ValueError("length must be a positive integer")
    if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
        raise ValueError("dim must be a positive integer")

    # Make positions a column vector so it broadcasts against one frequency per feature column.
    positions = torch.arange(length, dtype=torch.float64)[:, None]
    columns = torch.arange(dim, dtype=torch.long)
    # Adjacent even/odd columns share a frequency and later receive sine/cosine respectively.
    even_indices = 2 * (columns // 2)
    # The standard 10000-based schedule assigns slower variation to later feature pairs.
    angle_rates = torch.pow(
        torch.as_tensor(10000.0, dtype=torch.float64),
        torch.as_tensor(even_indices, dtype=torch.float64) / dim,
    )
    angles = positions / angle_rates

    # Fill interleaved columns so the final shape is (length, dim), including odd dim values.
    encoding = torch.empty((length, dim), dtype=torch.float64)
    encoding[:, 0::2] = torch.sin(angles[:, 0::2])
    encoding[:, 1::2] = torch.cos(angles[:, 1::2])
    return encoding`,
    starterCode: `from __future__ import annotations

import torch

def sinusoidal_positional_encoding(length: int, dim: int) -> torch.Tensor:
    # TODO:
    # 1. Validate positive integer length and embedding dimension.
    # 2. Build one angle table using positions and paired frequencies.
    # 3. Fill even columns with sine and odd columns with cosine.
    raise NotImplementedError("Implement sinusoidal_positional_encoding")

sample_length = 4
sample_dim = 5

print(sinusoidal_positional_encoding(sample_length, sample_dim))`,
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'Sequence Modeling', 'Embeddings'],
  },
  {
    id: 'unpatchify-back-to-image',
    order: 11,
    title: 'Unpatchify back to image',
    difficulty: 'Medium',
    summary:
      'Reconstruct batched images from flattened patch vectors using row-major patch order.',
    prompt: [
      'Write `unpatchify(patches, image_shape, patch_size)` so it reconstructs and returns a batch of images from flattened patch tokens.',
      'Assume the patches are in row-major order across the image grid. Validate the inputs, then reshape the patch tensor back into `(B, C, H, W)`.',
    ],
    signature: `def unpatchify(
    patches: torch.Tensor,
    image_shape: tuple[int, int, int],
    patch_size: int,
) -> torch.Tensor:
    ...`,
    requirements: [
      '`patches` has shape `(B, N, C * P * P)`.',
      '`image_shape` is `(C, H, W)`.',
      '`patch_size` is `P`.',
      'Reconstruct and return shape `(B, C, H, W)`.',
      'Assume patches are in row-major order.',
      'Raise `ValueError` on invalid input.',
    ],
    examples: [
      {
        label: 'Example 1',
        lines: [
          'patches = [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]',
          'image_shape = (1, 4, 4)',
          'patch_size = 2',
        ],
        result:
          '[[[[1, 2, 5, 6], [3, 4, 7, 8], [9, 10, 13, 14], [11, 12, 15, 16]]]]',
      },
      {
        label: 'Example 2',
        lines: [
          'patches = [[[1], [2], [3], [4]], [[5], [6], [7], [8]]]',
          'image_shape = (1, 2, 2)',
          'patch_size = 1',
        ],
        result: '[[[[1, 2], [3, 4]]], [[[5, 6], [7, 8]]]]',
      },
    ],
    hint: [
      'Check that the image height and width are divisible by `patch_size`.',
      'The number of patches should be `(H / P) * (W / P)` and each patch should have `C * P * P` values.',
      'Reshape the patches into a 6D tensor, then transpose axes to interleave the patch grid and patch pixels.',
      'The row-major assumption means the patch index should map to `(row, column)` in standard nested-loop order.',
    ],
    solutionNotes: [
      'This problem is the inverse of patch extraction: each flattened patch vector is first reshaped into `(C, P, P)`, then the patch grid is placed back into its `(H / P, W / P)` spatial layout.',
      'A reshape followed by a transpose is enough to undo the flattening as long as the patch order is row-major and the image dimensions divide evenly by the patch size.',
    ],
    solutionCode: `from __future__ import annotations

import torch

def unpatchify(
    patches: torch.Tensor,
    image_shape: tuple[int, int, int],
    patch_size: int,
) -> torch.Tensor:
    # Preserve patch values while making list-like inputs work with PyTorch shape operations.
    patches = torch.as_tensor(patches)
    shape = torch.as_tensor(image_shape)

    # A tokenized image has batch, patch, and flattened-patch axes.
    if patches.ndim != 3:
        raise ValueError("patches must have shape (B, N, C * P * P)")
    if shape.ndim != 1 or shape.shape[0] != 3 or torch.is_floating_point(shape):
        raise ValueError("image_shape must contain three integers")
    if isinstance(patch_size, bool) or not isinstance(patch_size, int) or patch_size <= 0:
        raise ValueError("patch_size must be a positive integer")

    # Unpack the requested output layout and ensure it describes a real patch grid.
    channels, height, width = (int(value) for value in shape.tolist())
    if channels <= 0 or height <= 0 or width <= 0:
        raise ValueError("image_shape must contain positive dimensions")
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("image dimensions must be divisible by patch_size")

    # Derive the exact token count and flattened width required by the requested image shape.
    grid_h, grid_w = height // patch_size, width // patch_size
    expected_patches = grid_h * grid_w
    expected_patch_dim = channels * patch_size * patch_size
    if patches.shape[1] != expected_patches or patches.shape[2] != expected_patch_dim:
        raise ValueError("patches do not match image_shape and patch_size")

    batch_size = patches.shape[0]
    # The transpose interleaves patch-grid coordinates with within-patch pixels.
    grid = torch.reshape(
        patches,
        (batch_size, grid_h, grid_w, channels, patch_size, patch_size),
    )
    # Move channels next to the batch axis and interleave grid coordinates with local pixels.
    grid = torch.permute(grid, (0, 3, 1, 4, 2, 5))
    # Collapse the paired grid/local axes back into image height and width.
    return torch.reshape(grid, (batch_size, channels, height, width))`,
    starterCode: `from __future__ import annotations

import torch

def unpatchify(
    patches: torch.Tensor,
    image_shape: tuple[int, int, int],
    patch_size: int,
) -> torch.Tensor:
    # TODO:
    # 1. Validate the patch tensor, image shape, and divisibility.
    # 2. Reshape into patch-grid and within-patch axes.
    # 3. Permute those axes back to (B, C, H, W).
    raise NotImplementedError("Implement unpatchify")

sample_patches = torch.tensor([
    [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
])
sample_image_shape = (1, 4, 4)

print(unpatchify(sample_patches, sample_image_shape, patch_size=2))`,
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'Computer Vision', 'Transformers'],
  },
  {
    id: '2d-patchify-for-images',
    order: 12,
    title: '2D patchify for images',
    difficulty: 'Medium',
    summary:
      'Split batched images into row-major flattened patch tokens for Vision Transformer style models.',
    prompt: [
      'Write `patchify(images, patch_size)` so it converts a batch of images into flattened patch tokens.',
      'Assume patches are ordered row-major over the image grid. Validate the inputs, then return an array of shape `(B, N, C * P * P)` where `N = (H // P) * (W // P)`.',
    ],
    signature: `def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    ...`,
    requirements: [
      '`images` has shape `(B, C, H, W)`.',
      '`patch_size` is a positive integer `P`.',
      'Assume `H` and `W` are divisible by `P`.',
      'Return an array of shape `(B, N, C * P * P)` where `N = (H // P) * (W // P)`.',
      'Patches should be ordered row-major over the image grid.',
      'Raise `ValueError` on invalid input.',
    ],
    examples: [
      {
        label: 'Example 1',
        lines: [
          'images = [[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]]',
          'patch_size = 2',
        ],
        result:
          '[[[1, 2, 5, 6], [3, 4, 7, 8], [9, 10, 13, 14], [11, 12, 15, 16]]]',
      },
      {
        label: 'Example 2',
        lines: [
          'images = [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]',
          'patch_size = 1',
        ],
        result: '[[[1, 5], [2, 6], [3, 7], [4, 8]]]',
      },
    ],
    hint: [
      'Split the height and width into patch-grid and within-patch axes with `reshape`.',
      'Transpose to move the grid axes before the channel and patch-pixel axes.',
      'Flatten the patch grid into `N = (H // P) * (W // P)` after the transpose.',
      'Validate that `patch_size` is positive and divides both spatial dimensions.',
    ],
    solutionNotes: [
      'The core trick is to expose the image grid as `(H // P, P, W // P, P)` so the patch structure becomes explicit.',
      'A reshape followed by a transpose keeps row-major patch order and makes the final flattening straightforward.',
    ],
    solutionCode: `from __future__ import annotations

import torch

def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    # Preserve the input dtype while allowing callers to provide a compatible array or list.
    images = torch.as_tensor(images)

    # Patchify starts from an image batch in the standard (B, C, H, W) layout.
    if images.ndim != 4:
        raise ValueError("images must have shape (B, C, H, W)")
    if isinstance(patch_size, bool) or not isinstance(patch_size, int) or patch_size <= 0:
        raise ValueError("patch_size must be a positive integer")

    batch_size, channels, height, width = images.shape
    if channels <= 0 or height <= 0 or width <= 0:
        raise ValueError("images must have positive dimensions")
    # A complete grid needs an integer number of patches along both spatial axes.
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("image dimensions must be divisible by patch_size")

    grid_h, grid_w = height // patch_size, width // patch_size
    # Expose grid and within-patch axes before flattening each row-major token.
    grid = torch.reshape(
        images,
        (batch_size, channels, grid_h, patch_size, grid_w, patch_size),
    )
    # Put the patch grid before channel/pixel data so tokens are in row-major grid order.
    grid = torch.permute(grid, (0, 2, 4, 1, 3, 5))
    # Flatten each patch into one token: (B, number_of_patches, C * P * P).
    return torch.reshape(
        grid,
        (batch_size, grid_h * grid_w, channels * patch_size * patch_size),
    )`,
    starterCode: `from __future__ import annotations

import torch

def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    # TODO:
    # 1. Validate the BCHW tensor and positive divisible patch size.
    # 2. Reshape into grid and within-patch axes.
    # 3. Permute to row-major tokens and flatten each patch.
    raise NotImplementedError("Implement patchify")

sample_images = torch.tensor([
    [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]],
])
print(patchify(sample_images, patch_size=2))`,
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'Computer Vision', 'Patch Embeddings'],
  },
  {
    id: 'rope-rotary-positional-embedding',
    order: 13,
    title: 'RoPE (Rotary Positional Embedding)',
    difficulty: 'Medium',
    summary:
      'Apply rotary positional embeddings across the last dimension of a batched attention tensor.',
    prompt: [
      'Implement rotary positional embeddings for a tensor of shape `(B, T, H, D)`, where `D` is even.',
      'Apply RoPE across the last dimension and return a tensor with the same shape. Treat the position as the `T` axis and rotate each adjacent pair of features with the standard `sin`/`cos` frequencies.',
    ],
    signature: `def apply_rope(x: torch.Tensor) -> torch.Tensor:
    ...`,
    requirements: [
      '`x` has shape `(B, T, H, D)`.',
      '`D` must be even.',
      'Return a tensor with the same shape as `x`.',
      'Apply RoPE across the last dimension.',
      'Raise `ValueError` on invalid input.',
    ],
    examples: [
      {
        label: 'Example 1',
        lines: [
          'x = [[[[1.0, 0.0, 1.0, 0.0]], [[1.0, 0.0, 1.0, 0.0]]]]',
        ],
        result:
          '[[[[1.0, 0.0, 1.0, 0.0]], [[0.54030, 0.84147, 0.99995, 0.01000]]]]',
      },
      {
        label: 'Example 2',
        lines: ['x = [[[[1.0, 0.0]], [[0.0, 1.0]]]]'],
        result: '[[[[1.0, 0.0]], [[-0.84147, 0.54030]]]]',
      },
    ],
    hint: [
      'Split the last dimension into even and odd coordinates, then rotate each pair together.',
      'Build position-dependent angles from the `T` index and a frequency vector derived from the pair index.',
      'Broadcast the `sin` and `cos` tables over batch and head dimensions.',
      'A helper like `rotate_half` can make the final formula easier to read.',
    ],
    solutionNotes: [
      'RoPE treats each adjacent pair of channels as a 2D vector and rotates it by an angle that depends on the token position. That preserves the vector norm while injecting relative position information into attention.',
      'The implementation is cleanest when you precompute one sine/cosine table per token position and frequency pair, then combine it with the input using the standard `rotate_half` pattern.',
    ],
    solutionCode: `from __future__ import annotations

import torch

def apply_rope(x: torch.Tensor) -> torch.Tensor:
    # RoPE uses continuous angles, so standardize the input to a floating dtype.
    x = torch.as_tensor(x, dtype=torch.float64)

    # The final feature axis is split into adjacent pairs, which requires an even width.
    if x.ndim != 4 or any(size <= 0 for size in x.shape):
        raise ValueError("x must have positive shape (B, T, H, D)")
    if x.shape[-1] % 2 != 0:
        raise ValueError("D must be even")

    batch_size, seq_len, num_heads, dim = x.shape
    # Build one inverse frequency for each adjacent feature pair.
    pair_indices = torch.arange(dim // 2, dtype=torch.float64)
    inverse_frequencies = 1.0 / torch.pow(
        torch.as_tensor(10000.0, dtype=torch.float64),
        (2.0 * pair_indices) / dim,
    )
    # Outer multiplication assigns every sequence position an angle per feature pair.
    positions = torch.arange(seq_len, dtype=torch.float64)[:, None]
    angles = positions * inverse_frequencies[None, :]

    # Add batch and head singleton axes so the same table broadcasts over both.
    sin = torch.sin(angles)[None, :, None, :]
    cos = torch.cos(angles)[None, :, None, :]
    x_even, x_odd = x[..., 0::2], x[..., 1::2]

    # Each adjacent feature pair is a 2D vector rotated by its position-dependent angle.
    out = torch.empty_like(x)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out`,
    starterCode: `from __future__ import annotations

import torch

def apply_rope(x: torch.Tensor) -> torch.Tensor:
    # TODO:
    # 1. Validate a positive (B, T, H, D) tensor with even D.
    # 2. Build position/frequency sine and cosine tables.
    # 3. Rotate each adjacent feature pair and preserve the original shape.
    raise NotImplementedError("Implement apply_rope")

sample_x = torch.tensor([
    [[[1.0, 0.0, 1.0, 0.0]], [[1.0, 0.0, 1.0, 0.0]]],
], dtype=torch.float64)
print(apply_rope(sample_x))`,
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'Attention', 'Transformers'],
  },
  {
    id: 'scaled-dot-product-self-attention',
    order: 14,
    title: 'Scaled dot-product self-attention',
    difficulty: 'Hard',
    summary:
      'Compute single-call multi-head self-attention with scaled dot-product scores, optional masking, and an output projection.',
    prompt: [
      'Implement single-call multi-head self-attention for a tensor of shape `(B, T, D_model)`.',
      'Project the input into query, key, and value spaces, split into `num_heads` heads, apply scaled dot-product attention with an optional mask, then project the concatenated heads back to `(B, T, D_model)`.',
    ],
    signature: `def self_attention(
    x: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    W_o: torch.Tensor,
    num_heads: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    ...`,
    requirements: [
      '`x` has shape `(B, T, D_model)`.',
      'Projection matrices `W_q`, `W_k`, `W_v`, and `W_o` all have shape `(D_model, D_model)`.',
      '`num_heads` divides `D_model`.',
      '`mask`, if provided, is broadcastable to `(B, H, T, T)` and contains `1` for allowed positions and `0` for blocked positions.',
      'Return an output of shape `(B, T, D_model)`.',
      'Raise `ValueError` on invalid input.',
    ],
    examples: [
      {
        label: 'Example 1',
        lines: [
          'x = [[[1.0, 0.0], [0.0, 1.0]]]',
          'W_q = W_k = W_v = W_o = [[1.0, 0.0], [0.0, 1.0]]',
          'num_heads = 1',
        ],
        result: '[[[0.66976, 0.33024], [0.33024, 0.66976]]]',
      },
      {
        label: 'Example 2',
        lines: [
          'x = [[[1.0, 0.0], [0.0, 1.0]]]',
          'W_q = W_k = W_v = W_o = [[1.0, 0.0], [0.0, 1.0]]',
          'num_heads = 1',
          'mask = [[[1, 0], [1, 1]]]',
        ],
        result: '[[[1.0, 0.0], [0.33024, 0.66976]]]',
      },
    ],
    hint: [
      'Reshape the projected tensors into `(B, H, T, D_head)` before computing attention scores.',
      'Use the scaled dot-product formula `Q K^T / sqrt(D_head)` and a numerically stable softmax over the last axis.',
      'If a mask is provided, broadcast it to the score tensor and zero out blocked positions before softmax.',
      'After attention, transpose the heads back and concatenate them before the final output projection.',
    ],
    solutionNotes: [
      'The workflow is the standard Transformer block: project to queries, keys, and values; split the channel dimension into heads; compute masked scaled dot-product attention; then merge the heads and apply the output projection.',
      'Broadcasted masking and a stable softmax are the two details that make the implementation robust. The mask keeps blocked positions from contributing, while the final projection preserves the original model width.',
    ],
    solutionCode: `from __future__ import annotations

import torch

def _stable_masked_softmax(logits: torch.Tensor) -> torch.Tensor:
    # Masked attention represents forbidden positions as -inf, so record the usable entries.
    finite = torch.isfinite(logits)
    # Replace non-finite values temporarily so an all-masked row has a finite maximum.
    safe_logits = torch.where(finite, logits, torch.zeros_like(logits))
    # Subtracting this row-wise maximum is the usual stable softmax shift.
    maximum = torch.amax(safe_logits, dim=-1, keepdim=True)
    # Exponentiate only valid entries; multiplying by finite restores exact zero for masked scores.
    exponentiated = torch.exp(safe_logits - maximum) * torch.as_tensor(
        finite,
        dtype=logits.dtype,
    )
    denominator = torch.sum(exponentiated, dim=-1, keepdim=True)
    # Use one for all-masked rows so division is defined before the final zeroing step.
    safe_denominator = torch.where(
        denominator > 0.0,
        denominator,
        torch.ones_like(denominator),
    )
    # An all-masked query has no valid distribution, so return an all-zero attention row.
    return torch.where(
        denominator > 0.0,
        exponentiated / safe_denominator,
        torch.zeros_like(exponentiated),
    )

def self_attention(
    x: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    W_o: torch.Tensor,
    num_heads: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    # Normalize all activations and projection matrices to one floating dtype.
    x = torch.as_tensor(x, dtype=torch.float64)
    W_q = torch.as_tensor(W_q, dtype=torch.float64)
    W_k = torch.as_tensor(W_k, dtype=torch.float64)
    W_v = torch.as_tensor(W_v, dtype=torch.float64)
    W_o = torch.as_tensor(W_o, dtype=torch.float64)

    # Self-attention consumes a non-empty batch of sequences with one model-width vector per token.
    if x.ndim != 3 or any(size <= 0 for size in x.shape):
        raise ValueError("x must have positive shape (B, T, D_model)")
    if isinstance(num_heads, bool) or not isinstance(num_heads, int) or num_heads <= 0:
        raise ValueError("num_heads must be a positive integer")

    batch_size, seq_len, model_dim = x.shape
    # Splitting model features evenly across heads requires an exact division.
    if model_dim % num_heads != 0:
        raise ValueError("num_heads must divide D_model")
    # Every projection maps D_model features back to D_model features.
    for matrix, name in ((W_q, "W_q"), (W_k, "W_k"), (W_v, "W_v"), (W_o, "W_o")):
        if matrix.ndim != 2 or matrix.shape != (model_dim, model_dim):
            raise ValueError(f"{name} must have shape (D_model, D_model)")

    head_dim = model_dim // num_heads
    # Project the same token sequence into query, key, and value feature spaces.
    q = torch.matmul(x, W_q)
    k = torch.matmul(x, W_k)
    v = torch.matmul(x, W_v)
    # Split D_model into heads, then move heads before sequence positions: (B, H, T, D_head).
    q = torch.permute(torch.reshape(q, (batch_size, seq_len, num_heads, head_dim)), (0, 2, 1, 3))
    k = torch.permute(torch.reshape(k, (batch_size, seq_len, num_heads, head_dim)), (0, 2, 1, 3))
    v = torch.permute(torch.reshape(v, (batch_size, seq_len, num_heads, head_dim)), (0, 2, 1, 3))

    # Each query compares with every key in its head; scale to keep logit magnitudes controlled.
    scores = torch.matmul(q, torch.transpose(k, -1, -2)) / (head_dim ** 0.5)
    if mask is not None:
        # Permit convenient mask shapes such as (T, T) while enforcing the final attention shape.
        mask = torch.as_tensor(mask)
        try:
            mask = torch.broadcast_to(mask, scores.shape)
        except ValueError as exc:
            raise ValueError("mask must be broadcastable to (B, H, T, T)") from exc
        if not bool(torch.all((mask == 0) | (mask == 1))):
            raise ValueError("mask must contain only 0 and 1")
        # Set forbidden scores to -inf so their softmax probability becomes exactly zero.
        scores = torch.where(
            mask != 0,
            scores,
            torch.full_like(scores, float("-inf")),
        )

    # Normalize key scores within each query row, then use them to mix value vectors.
    attention = _stable_masked_softmax(scores)
    context = torch.matmul(attention, v)
    # Return to token-major layout, concatenate heads, and apply the output projection.
    context = torch.permute(context, (0, 2, 1, 3))
    context = torch.reshape(context, (batch_size, seq_len, model_dim))
    return torch.matmul(context, W_o)`,
    starterCode: `from __future__ import annotations

import torch

def self_attention(
    x: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    W_o: torch.Tensor,
    num_heads: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    # TODO:
    # 1. Validate BCHD projection shapes and the head divisibility invariant.
    # 2. Project, reshape into heads, and compute scaled dot-product scores.
    # 3. Apply the optional binary mask, normalize stably, merge heads, and project out.
    raise NotImplementedError("Implement self_attention")

sample_x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float64)
sample_w = torch.eye(2, dtype=torch.float64)

print(self_attention(sample_x, sample_w, sample_w, sample_w, sample_w, num_heads=1))`,
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'Attention', 'Transformers'],
  },
  {
    id: 'cross-attention',
    order: 15,
    title: 'Cross-attention',
    difficulty: 'Hard',
    summary:
      'Compute multi-head cross-attention from a query sequence and a separate context sequence, with scaled dot-product scores and an output projection.',
    prompt: [
      'Implement multi-head cross-attention for a query tensor and a separate context tensor.',
      'Project the query sequence into queries, project the context sequence into keys and values, split into heads, apply scaled dot-product attention with an optional mask, then project the concatenated heads back to `(B, Tq, D_model)`.',
    ],
    signature: `def cross_attention(
    query_x: torch.Tensor,
    context_x: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    W_o: torch.Tensor,
    num_heads: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    ...`,
    requirements: [
      '`query_x` has shape `(B, Tq, D_model)`.',
      '`context_x` has shape `(B, Tk, D_model)`.',
      'Projection matrices `W_q`, `W_k`, `W_v`, and `W_o` all have shape `(D_model, D_model)`.',
      '`num_heads` divides `D_model`.',
      '`mask`, if provided, is broadcastable to `(B, H, Tq, Tk)` and contains `1` for allowed positions and `0` for blocked positions.',
      'Return an output of shape `(B, Tq, D_model)`.',
      'Raise `ValueError` on invalid input.',
    ],
    examples: [
      {
        label: 'Example 1',
        lines: [
          'query_x = [[[1.0, 0.0]]]',
          'context_x = [[[1.0, 0.0], [0.0, 1.0]]]',
          'W_q = W_k = W_v = W_o = [[1.0, 0.0], [0.0, 1.0]]',
          'num_heads = 1',
        ],
        result: '[[[0.66976, 0.33024]]]',
      },
      {
        label: 'Example 2',
        lines: [
          'query_x = [[[1.0, 0.0], [0.0, 1.0]]]',
          'context_x = [[[1.0, 0.0], [0.0, 1.0]]]',
          'W_q = W_k = W_v = W_o = [[1.0, 0.0], [0.0, 1.0]]',
          'num_heads = 1',
          'mask = [[[1, 0], [1, 1]]]',
        ],
        result: '[[[1.0, 0.0], [0.33024, 0.66976]]]',
      },
    ],
    hint: [
      'The only difference from self-attention is that queries come from `query_x`, while keys and values come from `context_x`.',
      'Reshape the projected tensors into `(B, H, Tq, D_head)` for queries and `(B, H, Tk, D_head)` for keys and values.',
      'Use the scaled dot-product formula `Q K^T / sqrt(D_head)` and a numerically stable softmax over the last axis.',
      'If a mask is provided, broadcast it to the score tensor and zero out blocked positions before softmax.',
    ],
    solutionNotes: [
      'Cross-attention is the same attention primitive as self-attention, except the query tokens and the key/value tokens come from different inputs. That makes it the right building block when one sequence needs to read information from another.',
      'The implementation follows the usual Transformer recipe: project queries, keys, and values; split channels into heads; compute masked scaled dot-product attention; then merge the heads and apply the output projection.',
    ],
    solutionCode: `from __future__ import annotations

import torch

def _stable_masked_softmax(logits: torch.Tensor) -> torch.Tensor:
    # Masked attention represents forbidden positions as -inf, so record the usable entries.
    finite = torch.isfinite(logits)
    # Replace non-finite values temporarily so an all-masked row has a finite maximum.
    safe_logits = torch.where(finite, logits, torch.zeros_like(logits))
    # Subtracting this row-wise maximum is the usual stable softmax shift.
    maximum = torch.amax(safe_logits, dim=-1, keepdim=True)
    # Exponentiate only valid entries; multiplying by finite restores exact zero for masked scores.
    exponentiated = torch.exp(safe_logits - maximum) * torch.as_tensor(
        finite,
        dtype=logits.dtype,
    )
    denominator = torch.sum(exponentiated, dim=-1, keepdim=True)
    # Use one for all-masked rows so division is defined before the final zeroing step.
    safe_denominator = torch.where(
        denominator > 0.0,
        denominator,
        torch.ones_like(denominator),
    )
    # An all-masked query has no valid distribution, so return an all-zero attention row.
    return torch.where(
        denominator > 0.0,
        exponentiated / safe_denominator,
        torch.zeros_like(exponentiated),
    )

def cross_attention(
    query_x: torch.Tensor,
    context_x: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    W_o: torch.Tensor,
    num_heads: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    # Query and context may have different lengths, but all attention arithmetic uses one float dtype.
    query_x = torch.as_tensor(query_x, dtype=torch.float64)
    context_x = torch.as_tensor(context_x, dtype=torch.float64)
    W_q = torch.as_tensor(W_q, dtype=torch.float64)
    W_k = torch.as_tensor(W_k, dtype=torch.float64)
    W_v = torch.as_tensor(W_v, dtype=torch.float64)
    W_o = torch.as_tensor(W_o, dtype=torch.float64)

    # Both inputs are batches of token vectors; only their sequence lengths may differ.
    if query_x.ndim != 3 or context_x.ndim != 3:
        raise ValueError("query_x and context_x must have shape (B, T, D_model)")
    if any(size <= 0 for size in query_x.shape) or any(size <= 0 for size in context_x.shape):
        raise ValueError("inputs must have positive dimensions")
    if query_x.shape[0] != context_x.shape[0] or query_x.shape[2] != context_x.shape[2]:
        raise ValueError("query_x and context_x must share batch size and model dimension")
    if isinstance(num_heads, bool) or not isinstance(num_heads, int) or num_heads <= 0:
        raise ValueError("num_heads must be a positive integer")

    batch_size, query_len, model_dim = query_x.shape
    context_len = context_x.shape[1]
    # Splitting model features evenly across heads requires an exact division.
    if model_dim % num_heads != 0:
        raise ValueError("num_heads must divide D_model")
    # Every projection maps D_model features back to D_model features.
    for matrix, name in ((W_q, "W_q"), (W_k, "W_k"), (W_v, "W_v"), (W_o, "W_o")):
        if matrix.ndim != 2 or matrix.shape != (model_dim, model_dim):
            raise ValueError(f"{name} must have shape (D_model, D_model)")

    head_dim = model_dim // num_heads
    # Queries come from query_x; keys and values come from the separate context sequence.
    q = torch.matmul(query_x, W_q)
    k = torch.matmul(context_x, W_k)
    v = torch.matmul(context_x, W_v)
    # Split features into heads and move heads before token positions.
    q = torch.permute(torch.reshape(q, (batch_size, query_len, num_heads, head_dim)), (0, 2, 1, 3))
    k = torch.permute(torch.reshape(k, (batch_size, context_len, num_heads, head_dim)), (0, 2, 1, 3))
    v = torch.permute(torch.reshape(v, (batch_size, context_len, num_heads, head_dim)), (0, 2, 1, 3))

    # Each query compares with every context key in its head; scale the dot products by sqrt(D_head).
    scores = torch.matmul(q, torch.transpose(k, -1, -2)) / (head_dim ** 0.5)
    if mask is not None:
        # Permit convenient mask shapes while enforcing the final (B, H, Tq, Tk) score layout.
        mask = torch.as_tensor(mask)
        try:
            mask = torch.broadcast_to(mask, scores.shape)
        except ValueError as exc:
            raise ValueError("mask must be broadcastable to (B, H, Tq, Tk)") from exc
        if not bool(torch.all((mask == 0) | (mask == 1))):
            raise ValueError("mask must contain only 0 and 1")
        # Set forbidden scores to -inf so their softmax probability becomes exactly zero.
        scores = torch.where(
            mask != 0,
            scores,
            torch.full_like(scores, float("-inf")),
        )

    # Normalize over context keys, then use those weights to mix context value vectors.
    attention = _stable_masked_softmax(scores)
    context = torch.matmul(attention, v)
    # Restore token-major layout, concatenate heads, and apply the output projection.
    context = torch.permute(context, (0, 2, 1, 3))
    context = torch.reshape(context, (batch_size, query_len, model_dim))
    return torch.matmul(context, W_o)`,
    starterCode: `from __future__ import annotations

import torch

def cross_attention(
    query_x: torch.Tensor,
    context_x: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    W_o: torch.Tensor,
    num_heads: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    # TODO:
    # 1. Validate query/context dimensions and head divisibility.
    # 2. Project queries from query_x and keys/values from context_x.
    # 3. Apply masked scaled dot-product attention and merge the heads.
    raise NotImplementedError("Implement cross_attention")

sample_query = torch.tensor([[[1.0, 0.0]]], dtype=torch.float64)
sample_context = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float64)
sample_w = torch.eye(2, dtype=torch.float64)

print(cross_attention(sample_query, sample_context, sample_w, sample_w, sample_w, sample_w, num_heads=1))`,
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'Attention', 'Transformers'],
  },
  {
    id: 'manual-backprop-for-a-2-layer-mlp',
    order: 16,
    title: 'Manual backprop for a 2-layer MLP',
    difficulty: 'Hard',
    summary:
      'Compute the loss and parameter gradients for a 2-layer ReLU MLP with softmax cross-entropy.',
    prompt: [
      'Implement forward and backward for a 2-layer MLP with one hidden ReLU layer.',
      'Given inputs `X`, labels `y`, and parameters `W1`, `b1`, `W2`, and `b2`, compute the mean softmax cross-entropy loss and the gradients for all four parameters.',
    ],
    signature: `def mlp_loss_and_grads(
    X: torch.Tensor,
    y: torch.Tensor,
    W1: torch.Tensor,
    b1: torch.Tensor,
    W2: torch.Tensor,
    b2: torch.Tensor,
) -> dict[str, torch.Tensor]:
    ...`,
    requirements: [
      '`X` has shape `(N, D_in)`.',
      '`y` has shape `(N,)` and contains integer class labels in the range `[0, C)`.',
      '`W1` has shape `(D_in, H)` and `b1` has shape `(H,)`.',
      '`W2` has shape `(H, C)` and `b2` has shape `(C,)`.',
      'Return the mean softmax cross-entropy loss and a dictionary with `dW1`, `db1`, `dW2`, and `db2`.',
      'Raise `ValueError` on invalid input.',
    ],
    examples: [
      {
        label: 'Example 1',
        lines: [
          'X = [[1.0, 2.0]]',
          'y = [1]',
          'W1 = [[1.0, 0.0], [0.0, 1.0]]',
          'b1 = [0.0, 0.0]',
          'W2 = [[1.0, 0.0], [0.0, 1.0]]',
          'b2 = [0.0, 0.0]',
        ],
        result: `{
  "loss": 0.31326,
  "dW1": [[0.26894, -0.26894], [0.53788, -0.53788]],
  "db1": [0.26894, -0.26894],
  "dW2": [[0.26894, -0.26894], [0.53788, -0.53788]],
  "db2": [0.26894, -0.26894]
}`,
      },
      {
        label: 'Example 2',
        lines: [
          'X = [[1.0, 0.0], [0.0, 1.0]]',
          'y = [0, 1]',
          'W1 = [[1.0, 0.0], [0.0, 1.0]]',
          'b1 = [1.0, 1.0]',
          'W2 = [[1.0, 0.0], [0.0, 1.0]]',
          'b2 = [0.0, 0.0]',
        ],
        result: `{
  "loss": 0.31326,
  "dW1": [[-0.13447, 0.13447], [0.13447, -0.13447]],
  "db1": [0.0, 0.0],
  "dW2": [[-0.13447, 0.13447], [0.13447, -0.13447]],
  "db2": [0.0, 0.0]
}`,
      },
    ],
    hint: [
      'Cache the hidden pre-activations so you can apply the ReLU derivative during backprop.',
      'Build stable probabilities with a max shift, exponentials, and row sums; then the logits gradient is `probs - one_hot(y)`, averaged over the batch.',
      'Backpropagate from the output layer into the hidden layer before multiplying by the ReLU mask.',
      'Return the gradients in a dictionary so the caller can inspect each parameter separately.',
    ],
    solutionNotes: [
      'The forward pass is affine, ReLU, affine, and a manually expanded stable softmax cross-entropy. After the max shift, exponentials, and row normalization produce `probs`, the logits gradient is the usual `probs - one_hot` term divided by the batch size.',
      'From there, the remaining gradients follow by the chain rule: the second affine layer gives `dW2` and `db2`, and the upstream gradient passes through the ReLU mask before producing `dW1` and `db1`.',
    ],
    solutionCode: `from __future__ import annotations

import torch

def _validate_mlp_inputs(X, y, W1, b1, W2, b2):
    # Use float tensors for activations/parameters, but check labels before converting them to indices.
    X = torch.as_tensor(X, dtype=torch.float64)
    y = torch.as_tensor(y)
    W1 = torch.as_tensor(W1, dtype=torch.float64)
    b1 = torch.as_tensor(b1, dtype=torch.float64)
    W2 = torch.as_tensor(W2, dtype=torch.float64)
    b2 = torch.as_tensor(b2, dtype=torch.float64)

    # X holds N examples, each with D_in input features; y supplies one class per example.
    if X.ndim != 2 or any(size <= 0 for size in X.shape):
        raise ValueError("X must have positive shape (N, D_in)")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y must have shape (N,)")
    if torch.is_floating_point(y):
        raise ValueError("y must contain integer class labels")
    y = torch.as_tensor(y, dtype=torch.long)

    # Check the first affine layer, then use its hidden width to validate subsequent tensors.
    input_dim = X.shape[1]
    if W1.ndim != 2 or W1.shape[0] != input_dim or W1.shape[1] == 0:
        raise ValueError("W1 must have shape (D_in, H)")
    hidden_dim = W1.shape[1]
    if b1.ndim != 1 or b1.shape[0] != hidden_dim:
        raise ValueError("b1 must have shape (H,)")
    if W2.ndim != 2 or W2.shape[0] != hidden_dim or W2.shape[1] == 0:
        raise ValueError("W2 must have shape (H, C)")
    num_classes = W2.shape[1]
    if b2.ndim != 1 or b2.shape[0] != num_classes:
        raise ValueError("b2 must have shape (C,)")
    # Gather and one-hot subtraction require targets in the final layer's class range.
    if bool(torch.any(y < 0)) or bool(torch.any(y >= num_classes)):
        raise ValueError("y contains labels outside the valid range")
    return X, y, W1, b1, W2, b2

def mlp_loss_and_grads(
    X: torch.Tensor,
    y: torch.Tensor,
    W1: torch.Tensor,
    b1: torch.Tensor,
    W2: torch.Tensor,
    b2: torch.Tensor,
) -> dict[str, torch.Tensor]:
    # Centralize coercion and all shape checks before computing a forward pass.
    X, y, W1, b1, W2, b2 = _validate_mlp_inputs(X, y, W1, b1, W2, b2)

    # First affine layer, ReLU activation, then second affine layer produce class logits.
    hidden_pre = torch.matmul(X, W1) + b1
    # ReLU retains positive activations and replaces the rest with zero.
    hidden = torch.where(hidden_pre > 0.0, hidden_pre, torch.zeros_like(hidden_pre))
    logits = torch.matmul(hidden, W2) + b2

    # Reuse the same max-shifted exponentials for the manual loss and logits gradient.
    # Shift logits row by row before exponentiating so large values cannot overflow.
    shifted_logits = logits - torch.amax(logits, dim=1, keepdim=True)
    # These positive weights are the numerator terms of softmax.
    exponentiated = torch.exp(shifted_logits)
    # Their class-wise total is the denominator for both probabilities and loss.
    normalizers = torch.sum(exponentiated, dim=1)
    # Restore a column axis on the total so it divides every class weight in a row.
    probabilities = exponentiated / normalizers[:, None]
    # Select each row's target logit with matching row and class-index vectors.
    row_indices = torch.arange(X.shape[0], dtype=torch.long)
    # log(denominator) minus the target score is the per-example cross-entropy.
    loss = torch.mean(torch.log(normalizers) - shifted_logits[row_indices, y])

    # Start dlogits at softmax probabilities, then subtract the one-hot target in each row.
    dlogits = torch.clone(probabilities)
    dlogits[torch.arange(X.shape[0]), y] -= 1.0
    dlogits = dlogits / X.shape[0]

    # Backpropagate through the output affine layer to obtain its parameter gradients.
    dW2 = torch.matmul(torch.transpose(hidden, 0, 1), dlogits)
    db2 = torch.sum(dlogits, dim=0)
    # Send the gradient through W2, then apply the ReLU derivative to the pre-activation.
    dhidden = torch.matmul(dlogits, torch.transpose(W2, 0, 1))
    dhidden_pre = dhidden * (hidden_pre > 0)
    # The remaining chain-rule terms are the first layer's weight and bias gradients.
    dW1 = torch.matmul(torch.transpose(X, 0, 1), dhidden_pre)
    db1 = torch.sum(dhidden_pre, dim=0)

    return {"loss": loss, "dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}`,
    starterCode: `from __future__ import annotations

import torch

def mlp_loss_and_grads(
    X: torch.Tensor,
    y: torch.Tensor,
    W1: torch.Tensor,
    b1: torch.Tensor,
    W2: torch.Tensor,
    b2: torch.Tensor,
) -> dict[str, torch.Tensor]:
    # TODO:
    # 1. Validate shapes, finite values, and integer label range.
    # 2. Run affine -> ReLU -> affine, then build stable probabilities from max, exp, sum, and log.
    # 3. Backpropagate the logits gradient through both affine layers.
    raise NotImplementedError("Implement mlp_loss_and_grads")

sample_X = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
sample_y = torch.tensor([1], dtype=torch.long)
sample_W1 = torch.eye(2, dtype=torch.float64)
sample_b1 = torch.zeros(2, dtype=torch.float64)
sample_W2 = torch.eye(2, dtype=torch.float64)
sample_b2 = torch.zeros(2, dtype=torch.float64)

print(mlp_loss_and_grads(sample_X, sample_y, sample_W1, sample_b1, sample_W2, sample_b2))`,
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'Backpropagation', 'Neural Networks'],
  },
  {
    id: 'classic-mlp-forward-backward',
    order: 17,
    title: 'Classic MLP forward + backward',
    difficulty: 'Hard',
    summary:
      'Run the forward and backward pass for a classic 2-layer ReLU MLP and return the loss plus parameter gradients.',
    prompt: [
      'This is the one I would especially practice.',
      'Implement forward and backward for a 2-layer MLP with one hidden ReLU layer, a softmax cross-entropy loss, and gradients for all trainable parameters.',
    ],
    signature: `def mlp_forward_backward(
    X: torch.Tensor,
    y: torch.Tensor,
    W1: torch.Tensor,
    b1: torch.Tensor,
    W2: torch.Tensor,
    b2: torch.Tensor,
) -> dict[str, torch.Tensor]:
    ...`,
    requirements: [
      '`X` has shape `(N, D_in)`.',
      '`y` has shape `(N,)` and contains integer class labels in the range `[0, C)`.',
      '`W1` has shape `(D_in, H)` and `b1` has shape `(H,)`.',
      '`W2` has shape `(H, C)` and `b2` has shape `(C,)`.',
      'Return the mean softmax cross-entropy loss and gradients for `W1`, `b1`, `W2`, and `b2`.',
      'Raise `ValueError` on invalid input.',
    ],
    examples: [
      {
        label: 'Example 1',
        lines: [
          'X = [[1.0, 2.0]]',
          'y = [1]',
          'W1 = [[1.0, 0.0], [0.0, 1.0]]',
          'b1 = [0.0, 0.0]',
          'W2 = [[1.0, 0.0], [0.0, 1.0]]',
          'b2 = [0.0, 0.0]',
        ],
        result: `{
  "loss": 0.31326,
  "dW1": [[0.26894, -0.26894], [0.53788, -0.53788]],
  "db1": [0.26894, -0.26894],
  "dW2": [[0.26894, -0.26894], [0.53788, -0.53788]],
  "db2": [0.26894, -0.26894]
}`,
      },
      {
        label: 'Example 2',
        lines: [
          'X = [[1.0, 0.0], [0.0, 1.0]]',
          'y = [0, 1]',
          'W1 = [[1.0, 0.0], [0.0, 1.0]]',
          'b1 = [1.0, 1.0]',
          'W2 = [[1.0, 0.0], [0.0, 1.0]]',
          'b2 = [0.0, 0.0]',
        ],
        result: `{
  "loss": 0.31326,
  "dW1": [[-0.13447, 0.13447], [0.13447, -0.13447]],
  "db1": [0.0, 0.0],
  "dW2": [[-0.13447, 0.13447], [0.13447, -0.13447]],
  "db2": [0.0, 0.0]
}`,
      },
    ],
    hint: [
      'Cache the hidden pre-activations so you can apply the ReLU derivative during backprop.',
      'Build stable probabilities with a max shift, exponentials, and row sums; then the logits gradient is `probs - one_hot(y)`, averaged over the batch.',
      'Backpropagate from the output layer into the hidden layer before multiplying by the ReLU mask.',
      'Return the gradients in a dictionary so the caller can inspect each parameter separately.',
    ],
    solutionNotes: [
      'The forward pass is affine, ReLU, affine, and a manually expanded stable softmax cross-entropy. After the max shift, exponentials, and row normalization produce `probs`, the logits gradient is the usual `probs - one_hot` term divided by the batch size.',
      'From there, the remaining gradients follow by the chain rule: the second affine layer gives `dW2` and `db2`, and the upstream gradient passes through the ReLU mask before producing `dW1` and `db1`.',
    ],
    solutionCode: `from __future__ import annotations

import torch

def mlp_forward_backward(
    X: torch.Tensor,
    y: torch.Tensor,
    W1: torch.Tensor,
    b1: torch.Tensor,
    W2: torch.Tensor,
    b2: torch.Tensor,
) -> dict[str, torch.Tensor]:
    # Use floating tensors for matrix arithmetic; validate labels before converting them to indices.
    X = torch.as_tensor(X, dtype=torch.float64)
    y = torch.as_tensor(y)
    W1 = torch.as_tensor(W1, dtype=torch.float64)
    b1 = torch.as_tensor(b1, dtype=torch.float64)
    W2 = torch.as_tensor(W2, dtype=torch.float64)
    b2 = torch.as_tensor(b2, dtype=torch.float64)

    # X holds N examples with D_in features, and y supplies one integer class per example.
    if X.ndim != 2 or any(size <= 0 for size in X.shape):
        raise ValueError("X must have positive shape (N, D_in)")
    if y.ndim != 1 or y.shape[0] != X.shape[0] or torch.is_floating_point(y):
        raise ValueError("y must be an integer tensor of shape (N,)")
    y = torch.as_tensor(y, dtype=torch.long)

    # Validate each parameter against the layer width established by the previous tensor.
    input_dim = X.shape[1]
    if W1.ndim != 2 or W1.shape[0] != input_dim or W1.shape[1] == 0:
        raise ValueError("W1 must have shape (D_in, H)")
    hidden_dim = W1.shape[1]
    if b1.ndim != 1 or b1.shape[0] != hidden_dim:
        raise ValueError("b1 must have shape (H,)")
    if W2.ndim != 2 or W2.shape[0] != hidden_dim or W2.shape[1] == 0:
        raise ValueError("W2 must have shape (H, C)")
    num_classes = W2.shape[1]
    if b2.ndim != 1 or b2.shape[0] != num_classes:
        raise ValueError("b2 must have shape (C,)")
    if bool(torch.any(y < 0)) or bool(torch.any(y >= num_classes)):
        raise ValueError("y contains labels outside the valid range")

    # First affine layer, ReLU activation, and second affine layer form the forward pass.
    hidden_pre = torch.matmul(X, W1) + b1
    # ReLU preserves positive hidden units and zeros out inactive ones.
    hidden = torch.where(hidden_pre > 0.0, hidden_pre, torch.zeros_like(hidden_pre))
    logits = torch.matmul(hidden, W2) + b2

    # Reuse the same max-shifted exponentials for the manual loss and logits gradient.
    # Shift logits row by row before exponentiating so large values cannot overflow.
    shifted_logits = logits - torch.amax(logits, dim=1, keepdim=True)
    # These positive weights are the numerator terms of softmax.
    exponentiated = torch.exp(shifted_logits)
    # Their class-wise total is the denominator for both probabilities and loss.
    normalizers = torch.sum(exponentiated, dim=1)
    # Restore a column axis on the total so it divides every class weight in a row.
    probabilities = exponentiated / normalizers[:, None]
    # Select each row's target logit with matching row and class-index vectors.
    row_indices = torch.arange(X.shape[0], dtype=torch.long)
    # log(denominator) minus the target score is the per-example cross-entropy.
    loss = torch.mean(torch.log(normalizers) - shifted_logits[row_indices, y])

    # Start dlogits at softmax probabilities, then subtract the one-hot target in each row.
    dlogits = torch.clone(probabilities)
    dlogits[torch.arange(X.shape[0]), y] -= 1.0
    dlogits = dlogits / X.shape[0]

    # Apply the chain rule through the second affine layer.
    dW2 = torch.matmul(torch.transpose(hidden, 0, 1), dlogits)
    db2 = torch.sum(dlogits, dim=0)
    # Propagate through W2 and gate inactive ReLU units.
    dhidden = torch.matmul(dlogits, torch.transpose(W2, 0, 1))
    dhidden_pre = dhidden * (hidden_pre > 0)
    # Finish the chain rule for the first affine layer.
    dW1 = torch.matmul(torch.transpose(X, 0, 1), dhidden_pre)
    db1 = torch.sum(dhidden_pre, dim=0)

    return {"loss": loss, "dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}`,
    starterCode: `from __future__ import annotations

import torch

def mlp_forward_backward(
    X: torch.Tensor,
    y: torch.Tensor,
    W1: torch.Tensor,
    b1: torch.Tensor,
    W2: torch.Tensor,
    b2: torch.Tensor,
) -> dict[str, torch.Tensor]:
    # TODO:
    # 1. Validate shapes, finite values, and integer label range.
    # 2. Run affine -> ReLU -> affine, then build stable probabilities from max, exp, sum, and log.
    # 3. Backpropagate the logits gradient through the ReLU mask and both affine layers.
    raise NotImplementedError("Implement mlp_forward_backward")

sample_X = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
sample_y = torch.tensor([1], dtype=torch.long)
sample_W1 = torch.eye(2, dtype=torch.float64)
sample_b1 = torch.zeros(2, dtype=torch.float64)
sample_W2 = torch.eye(2, dtype=torch.float64)
sample_b2 = torch.zeros(2, dtype=torch.float64)

print(mlp_forward_backward(sample_X, sample_y, sample_W1, sample_b1, sample_W2, sample_b2))`,
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'Backpropagation', 'Neural Networks'],
  },
  {
    id: 'simple-n-gram-language-model',
    order: 18,
    title: 'Simple n-gram language model',
    difficulty: 'Medium',
    summary:
      'Build a tiny backoff n-gram model that learns token counts, then fit it on a real corpus slice such as Tiny Shakespeare.',
    prompt: [
      'Implement a simple n-gram language model class with `__init__`, `fit`, `next_token_probs`, and `generate` methods.',
      'Train on a list of tokens, return next-token probability distributions from observed counts, sample autoregressively, and back off gracefully when a context has not been seen before.',
      'Use the provided Tiny Shakespeare loader to pull in real text, tokenize it, and build the model from more than a hand-written toy sequence.',
    ],
    signature: `class NGramModel:
    def __init__(self, n: int):
        ...

    def fit(self, tokens: Iterable[Token]) -> NGramModel:
        ...

    def next_token_probs(self, context: Iterable[Token]) -> dict[Token, float]:
        ...

    def generate(self, max_tokens: int, seed: int | None = None) -> list[Token]:
        ...`,
    requirements: [
      '`n` is an integer with `n >= 1`.',
      '`fit(tokens)` trains on a 1D list of tokens, where each token is a string or int.',
      'The same `fit(tokens)` method should work for toy token lists and for tokens derived from a real text corpus.',
      'Store the observed counts needed to answer next-token queries for orders up to `n`.',
      '`next_token_probs(context)` returns a dictionary mapping candidate next tokens to probabilities that sum to `1.0`.',
      'If a context is unseen, back off to progressively shorter suffixes until a seen context is found.',
      '`generate(max_tokens, seed=None)` samples up to `max_tokens` tokens autoregressively.',
      'Sampling must be deterministic when `seed` is provided.',
      'Raise `ValueError` on invalid inputs.',
    ],
    examples: [
      {
        label: 'Example 1',
        lines: [
          'model = NGramModel(2)',
          'model.fit(["a", "b", "a", "c"])',
          'model.next_token_probs(["a"])',
        ],
        result: `{'b': 0.5, 'c': 0.5}`,
      },
      {
        label: 'Example 2',
        lines: [
          'model = NGramModel(2)',
          'model.fit(["a", "b", "a", "c"])',
          'model.next_token_probs(["z"])',
          'model.generate(5, seed=4)',
        ],
        result: `backoff probs = {'a': 0.5, 'b': 0.25, 'c': 0.25}
generated = ['a', 'b', 'a', 'b', 'a']`,
      },
      {
        label: 'Example 3',
        lines: [
          'text = load_tiny_shakespeare(max_chars=4000)',
          'tokens = tokenize_words(text)',
          'model = NGramModel(3)',
          'model.fit(tokens)',
          'model.next_token_probs(["Before", "we"])',
        ],
        result: `{'proceed': 1.0}`,
      },
    ],
    hint: [
      'A dictionary keyed by context tuples works for both string and integer tokens.',
      'During training, update counts for every suffix length from `0` up to `n - 1`, not just the longest context.',
      'To back off gracefully, keep shortening the context suffix until you find a context with observed counts.',
      'Use a dedicated seeded RNG inside `generate` so sampling is repeatable without touching global random state.',
      'Keep corpus loading and tokenization outside the class so the n-gram model stays reusable for any token source.',
    ],
    solutionNotes: [
      'The simplest clean design is to treat each context as a tuple and map it to a counter of next-token counts. While fitting, update every available suffix length for each position, which gives you unigram counts, bigram counts, and so on up to order `n` in one pass.',
      'At inference time, truncate the supplied context to at most `n - 1` tokens and repeatedly back off to shorter suffixes until a seen context appears. Normalizing the matching counter gives the probability distribution, and `generate` can repeatedly sample from that distribution with a local `random.Random(seed)` instance for deterministic behavior.',
      'Because `fit` only cares about tokens, a tiny helper can load a slice of Tiny Shakespeare, split it into word tokens, and pass those tokens directly into the same model implementation.',
    ],
    solutionCode: `from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
import random

# Each vocabulary item can be a word-like string or an integer token id.
Token = str | int

# Prefer the browser-mounted corpus, then the public source, then a small offline fallback.
TINY_SHAKESPEARE_PATH = "/datasets/tiny-shakespeare.txt"
TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
TINY_SHAKESPEARE_FALLBACK = """First Citizen: Before we proceed any further, hear me speak.
All: Speak, speak.
First Citizen: You are all resolved rather to die than to famish?
All: Resolved. resolved.
First Citizen: First, you know Caius Marcius is chief enemy to the people.
All: We know't, we know't.
"""

def load_tiny_shakespeare(max_chars: int = 12000) -> str:
    # Bound the downloaded/demo corpus to a positive, predictable amount of text.
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0:
        raise ValueError("max_chars must be a positive integer")

    try:
        # Pyodide exposes browser-served assets through open_url.
        from pyodide.http import open_url

        text = open_url(TINY_SHAKESPEARE_PATH).read()
    except Exception:
        try:
            # Outside the browser, fetch the same corpus directly from its public source.
            from urllib.request import urlopen

            with urlopen(TINY_SHAKESPEARE_URL, timeout=10) as response:
                text = response.read().decode("utf-8")
        except Exception:
            # Keep the exercise runnable even when neither the mounted asset nor network is available.
            text = TINY_SHAKESPEARE_FALLBACK

    # Convert to str defensively and trim only after a source has been selected.
    return str(text[:max_chars])

def tokenize_words(text: str) -> list[str]:
    # This minimal tokenizer treats whitespace-delimited chunks as word tokens.
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    return text.split()

def _coerce_tokens(values: Iterable[Token], name: str) -> list[Token]:
    # A bare string is iterable by character but is not a sequence of already-tokenized values.
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of string or int tokens")
    try:
        # Materialize iterables once because training and validation may make multiple passes.
        items = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of string or int tokens") from exc
    # bool is an int subclass, but treating True/False as vocabulary items is rarely intentional.
    if any(isinstance(token, bool) or not isinstance(token, (str, int)) for token in items):
        raise ValueError(f"{name} must contain only string or int tokens")
    return items

class NGramModel:
    def __init__(self, n: int):
        # An n-gram order of one is valid: it learns an unconditional token distribution.
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("n must be an integer >= 1")
        self.n = n
        # Map each context tuple to a Counter of the tokens observed immediately after it.
        self.counts: defaultdict[tuple[Token, ...], Counter[Token]] = defaultdict(Counter)

    def fit(self, tokens: Iterable[Token]) -> NGramModel:
        # Validate and materialize the token stream before rebuilding the model state.
        tokens = _coerce_tokens(tokens, "tokens")
        self.counts.clear()

        # Visit each observed next token once.
        for index, token in enumerate(tokens):
            # Store every available suffix so inference can back off without retraining.
            for context_len in range(min(self.n - 1, index) + 1):
                context = tuple(tokens[index - context_len:index]) if context_len else ()
                # Increment the count for seeing token immediately after that context.
                self.counts[context][token] += 1
        return self

    def next_token_probs(self, context: Iterable[Token]) -> dict[Token, float]:
        # Normalize any caller-supplied history to the same token representation as fit.
        context = _coerce_tokens(context, "context")
        # Try the longest usable suffix first, then progressively fall back to shorter contexts.
        for context_len in range(min(len(context), self.n - 1), -1, -1):
            key = tuple(context[-context_len:]) if context_len else ()
            counts = self.counts.get(key)
            if counts:
                # Divide counts by their total to expose a proper categorical distribution.
                total = sum(counts.values())
                return {token: count / total for token, count in counts.items()}
        # An unfitted or empty model has no next-token distribution.
        return {}

    def generate(self, max_tokens: int, seed: int | None = None) -> list[Token]:
        # Validate output length and optional seed without accepting bool as an integer argument.
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 0:
            raise ValueError("max_tokens must be a non-negative integer")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("seed must be an integer or None")

        # A local RNG makes a supplied seed reproducible without modifying global random state.
        rng = random.Random(seed)
        generated: list[Token] = []
        for _ in range(max_tokens):
            # The generated history is the context; next_token_probs automatically backs off.
            probs = self.next_token_probs(generated)
            if not probs:
                break
            # random.choices expects parallel item and probability-weight lists.
            tokens, weights = list(probs), list(probs.values())
            # A local RNG keeps seeded generation reproducible without global-state side effects.
            generated.append(rng.choices(tokens, weights=weights, k=1)[0])
        return generated

# Load a bounded corpus, tokenize it, and train a trigram model in one fluent expression.
text = load_tiny_shakespeare(max_chars=12000)
tokens = tokenize_words(text)
model = NGramModel(3).fit(tokens)

# Show corpus size, one conditional distribution, and a deterministic sampled continuation.
print(f"loaded {len(tokens)} tokens")
print(model.next_token_probs(["Before", "we"]))
print(" ".join(str(token) for token in model.generate(12, seed=7)))`,
    starterCode: `from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
import random

Token = str | int

TINY_SHAKESPEARE_PATH = "/datasets/tiny-shakespeare.txt"
TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
TINY_SHAKESPEARE_FALLBACK = """First Citizen: Before we proceed any further, hear me speak.
All: Speak, speak.
First Citizen: You are all resolved rather to die than to famish?
All: Resolved. resolved.
First Citizen: First, you know Caius Marcius is chief enemy to the people.
All: We know't, we know't.
"""

def load_tiny_shakespeare(max_chars: int = 12000) -> str:
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0:
        raise ValueError("max_chars must be a positive integer")

    try:
        from pyodide.http import open_url

        text = open_url(TINY_SHAKESPEARE_PATH).read()
    except Exception:
        try:
            from urllib.request import urlopen

            with urlopen(TINY_SHAKESPEARE_URL, timeout=10) as response:
                text = response.read().decode("utf-8")
        except Exception:
            text = TINY_SHAKESPEARE_FALLBACK

    return str(text[:max_chars])

def tokenize_words(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    return text.split()

def _coerce_tokens(values: Iterable[Token], name: str) -> list[Token]:
    # TODO: reject strings as containers and validate each token type.
    raise NotImplementedError("Implement _coerce_tokens")

class NGramModel:
    def __init__(self, n: int):
        # TODO: validate n and initialize the suffix-count mapping.
        raise NotImplementedError("Implement __init__")

    def fit(self, tokens: Iterable[Token]) -> NGramModel:
        # TODO: update counts for every suffix length from 0 through n - 1.
        raise NotImplementedError("Implement fit")

    def next_token_probs(self, context: Iterable[Token]) -> dict[Token, float]:
        # TODO: back off from the longest context suffix and normalize counts.
        raise NotImplementedError("Implement next_token_probs")

    def generate(self, max_tokens: int, seed: int | None = None) -> list[Token]:
        # TODO: use a local seeded RNG and sample autoregressively.
        raise NotImplementedError("Implement generate")

text = load_tiny_shakespeare(max_chars=12000)
tokens = tokenize_words(text)
model = NGramModel(3).fit(tokens)

print(f"loaded {len(tokens)} tokens")
print(model.next_token_probs(["Before", "we"]))
print(" ".join(str(token) for token in model.generate(12, seed=7)))`,
    tags: ['Language Models', 'Probability', 'Hash Maps'],
  },
] as const;
