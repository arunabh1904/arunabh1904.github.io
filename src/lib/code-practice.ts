import { ARCHITECTURE_CODE_PRACTICE_PROBLEMS } from './code-practice-architectures';
import {
  ATTENTION_CODE_PRACTICE_PROBLEMS,
  ATTENTION_PROBLEM_ENRICHMENTS,
} from './code-practice-attention';

export interface CodePracticeExample {
  label: string;
  lines: readonly string[];
  result: string;
}

export interface CodePracticeVisual {
  src: string;
  alt: string;
  caption: string;
}

export interface CodePracticeInterviewFormat {
  durationMinutes: number;
  evaluationCriteria: readonly string[];
  followUps: readonly string[];
}

export type CodePracticeReasoningAxis =
  | 'Inference efficiency'
  | 'Tensor reasoning'
  | 'Memory / computation tradeoff'
  | 'Cache update correctness';

export interface CodePracticeReasoningPoint {
  axis: CodePracticeReasoningAxis;
  detail: string;
}

export interface CodePracticeProblem {
  id: string;
  order: number;
  title: string;
  difficulty: 'Easy' | 'Medium' | 'Hard';
  summary: string;
  prompt: readonly string[];
  signature: string;
  requirements: readonly string[];
  examples: readonly CodePracticeExample[];
  hint: readonly string[];
  solutionNotes: readonly string[];
  solutionCode: string;
  /** Full commented reference used to explain the compact editor solution line by line. */
  walkthroughCode?: string;
  visual?: CodePracticeVisual;
  solutionDiagram?: string;
  starterCode: string;
  track?: 'fundamentals' | 'architecture';
  environment?: 'browser' | 'local-pytorch';
  editorStart?: 'blank' | 'scaffold';
  interview?: CodePracticeInterviewFormat;
  reasoning?: readonly CodePracticeReasoningPoint[];
  packages?: readonly string[];
  tags?: readonly string[];
}

export const CODE_PRACTICE_SECTION_SUMMARY =
  'Practice the way you would code in an ML interview: clarify the contract, design a small API, implement it cleanly, test shapes, and defend the tradeoffs.';

const PYTORCH_AND_NUMPY_PACKAGES = ['torch', 'numpy'] as const;

export function getCodePracticeProblemPath(problem: Pick<CodePracticeProblem, 'id'> | string) {
  const problemId = typeof problem === 'string' ? problem : problem.id;
  return `/code/${problemId}.html`;
}

export function getCodePracticeProblemById(problemId: string) {
  return codePracticeProblems.find((problem) => problem.id === problemId);
}

const RAW_CODE_PRACTICE_PROBLEMS = [
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
    visual: {
      src: '/assets/images/code-tensor-ops-broadcasting.gif',
      alt: 'Animated tensor diagrams showing broadcasting, expand, torch.cat, and torch.stack.',
      caption:
        'Tensor shape intuition: broadcasting aligns singleton axes, expand exposes a larger view, cat joins an existing axis, and stack creates a new axis. The same shape bookkeeping appears throughout the first PyTorch problems.',
    },
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
    id: 'class-weighted-cross-entropy',
    order: 35,
    title: 'Class-weighted cross-entropy',
    difficulty: 'Medium',
    summary:
      'Extend multiclass cross-entropy with one weight per class and PyTorch-compatible mean reduction.',
    prompt: [
      'Write `class_weighted_cross_entropy(logits, labels, class_weight)` for class scores shaped `(N, C)`, target class ids shaped `(N,)`, and one non-negative weight per class shaped `(C,)`.',
      'Compute cross-entropy from logits with a stable log-sum-exp path. Weight each example by its target class, then divide the weighted loss sum by the sum of the selected class weights.',
    ],
    signature: `def class_weighted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weight: torch.Tensor,
) -> torch.Tensor:
    ...`,
    requirements: [
      '`logits` has shape `(N, C)`, `labels` has shape `(N,)`, and `class_weight` has shape `(C,)`.',
      'Labels contain integer class ids in `[0, C - 1]`.',
      'Class weights are finite and non-negative, and the selected target weights have a positive sum.',
      'Use a numerically stable cross-entropy calculation from logits.',
      'Return `sum_i w[y_i] * loss_i / sum_i w[y_i]`, matching PyTorch weighted mean reduction.',
    ],
    examples: [
      {
        label: 'Example',
        lines: [
          'logits = [[2.0, 1.0, 0.1], [0.5, 1.5, -0.5]]',
          'labels = [0, 1]',
          'class_weight = [1.0, 2.0, 0.5]',
        ],
        result: '0.41075',
      },
    ],
    hint: [
      'Subtract each row maximum before exponentiating, as in stable softmax cross-entropy.',
      'Use `class_weight[labels]` to gather one scalar weight per batch row.',
      'Normalize by the gathered weights’ sum, not by `N`.',
    ],
    solutionNotes: [
      'For row `i`, first compute `loss_i = log(sum_c exp(z_i,c - m_i)) - (z_i,y_i - m_i)`, where `m_i` is the largest logit in that row. Subtracting `m_i` leaves the softmax unchanged and prevents overflow.',
      'The class vector has shape `(C,)`; indexing it with labels shaped `(N,)` produces `example_weight` shaped `(N,)`. The final loss is `sum_i w[y_i] loss_i / sum_i w[y_i]`. This denominator preserves PyTorch’s weighted-mean behavior when a batch contains different class mixtures.',
    ],
    solutionCode: `import torch

def class_weighted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weight: torch.Tensor,
) -> torch.Tensor:
    logits = torch.as_tensor(logits, dtype=torch.float64)
    labels = torch.as_tensor(labels)
    class_weight = torch.as_tensor(class_weight, dtype=logits.dtype)
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] == 0:
        raise ValueError("logits must have non-empty shape (N, C)")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("labels must have shape (N,)")
    if torch.is_floating_point(labels):
        raise ValueError("labels must contain integer class ids")
    if class_weight.shape != (logits.shape[1],):
        raise ValueError("class_weight must have shape (C,)")
    if not bool(torch.all(torch.isfinite(logits))) or not bool(torch.all(torch.isfinite(class_weight))):
        raise ValueError("logits and class_weight must be finite")
    if bool(torch.any(class_weight < 0)):
        raise ValueError("class_weight must be non-negative")
    labels = torch.as_tensor(labels, dtype=torch.long)
    if bool(torch.any(labels < 0)) or bool(torch.any(labels >= logits.shape[1])):
        raise ValueError("labels contain out-of-range class ids")

    shifted = logits - torch.amax(logits, dim=1, keepdim=True)
    log_normalizers = torch.log(torch.sum(torch.exp(shifted), dim=1))
    rows = torch.arange(logits.shape[0], dtype=torch.long)
    per_example_loss = log_normalizers - shifted[rows, labels]
    example_weight = class_weight[labels]
    weight_sum = torch.sum(example_weight)
    if float(weight_sum.item()) <= 0:
        raise ValueError("selected class weights must have a positive sum")
    return torch.sum(per_example_loss * example_weight) / weight_sum

logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 1.5, -0.5]])
labels = torch.tensor([0, 1])
class_weight = torch.tensor([1.0, 2.0, 0.5])
print(class_weighted_cross_entropy(logits, labels, class_weight).item())`,
    starterCode: `import torch

def class_weighted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weight: torch.Tensor,
) -> torch.Tensor:
    # TODO: compute stable per-example cross-entropy, gather target weights, and weighted-mean reduce.
    raise NotImplementedError("Implement class_weighted_cross_entropy")

logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 1.5, -0.5]])
labels = torch.tensor([0, 1])
class_weight = torch.tensor([1.0, 2.0, 0.5])
print(class_weighted_cross_entropy(logits, labels, class_weight).item())`,
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'Classification', 'Losses', 'Class Imbalance'],
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
      'For `x.shape == (N, D)` and `y.shape == (M, D)`, `x_norms` has shape `(N,)` and `y_norms` has shape `(M,)`. `x_norms[:, None]` changes the first vector to `(N, 1)`, while `y_norms[None, :]` changes the second to `(1, M)`.',
      'That makes `denominator = x_norms[:, None] * y_norms[None, :]` a broadcasted outer product: entry `[i, j]` is `||x[i]|| * ||y[j]||`, and the whole denominator has shape `(N, M)` to match `x @ y.T`. The key edge case is a zero vector, so those positions are explicitly returned as `0.0`.',
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
      'The main trick is to form all pairwise overlap rectangles with broadcasting. If `boxes1.shape == (N, 4)` and `boxes2.shape == (M, 4)`, then `boxes1[:, None, :2]` is `(N, 1, 2)` and `boxes2[None, :, :2]` is `(1, M, 2)`; `torch.maximum` broadcasts them to `(N, M, 2)`, one top-left corner for every box pair.',
      'The same pattern gives bottom-right corners `(N, M, 2)`, intersection areas `(N, M)`, and union `area1[:, None] + area2[None, :] - intersection`, where `(N, 1)` and `(1, M)` broadcast into an `(N, M)` matrix. Every reduction removes only the coordinate axis, so the final output still has one IoU per pair.',
      'Once the pairwise union is known, a `torch.where` denominator mask keeps the implementation stable and handles degenerate boxes cleanly.',
    ],
    solutionDiagram: `boxes1 (N, 4)      boxes2 (M, 4)
      │                    │
      ├─[:, None, :2]      └─[None, :, :2]
      │  (N, 1, 2)             (1, M, 2)
      └──────── broadcast maximum ────────┐
                                           ▼
                                     top_left (N, M, 2)

area1 (N, 1) + area2 (1, M) - intersection (N, M)
                         → union / IoU (N, M)`,
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
      'Imagine the training rows as points in a feature space: `train_X[i]` is one point and `train_y[i]` says which class owns it. Write `nearest_centroid_predict(train_X, train_y, test_X)` to label each new row in `test_X`.',
      'For each distinct class, average its training points to make one representative point—the class centroid. Compare every test point with every centroid using Euclidean distance, then return the label of the closest centroid. If distances tie, choose the smaller class label.',
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
      'The nearest-centroid rule compresses each class into one vector. `class_points` has shape `(number_of_points_in_class, D)`, summing over `dim=0` leaves a `(D,)` vector, and `torch.stack` turns all class vectors into `centroids.shape == (K, D)`.',
      'With `test_X.shape == (M, D)` and `centroids.shape == (K, D)`, `test_X[:, None, :]` is `(M, 1, D)` and `centroids[None, :, :]` is `(1, K, D)`. Subtraction broadcasts to `(M, K, D)`; summing over `D` gives one squared distance per test-point/class pair, `(M, K)`.',
      'Squared Euclidean distance preserves the same ordering as Euclidean distance, and keeping the class labels sorted makes `argmin` deterministic when two centroids are equally close.',
    ],
    solutionDiagram: `train_X (N, D) + train_y (N,)
        └─ group rows by class → one centroid per class
           centroids (K, D)

test_X (M, 1, D) - centroids (1, K, D)
                 → deltas (M, K, D)
                 → sum over D → distances (M, K)
                 → argmin over K → predictions (M,)`,
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
      'The intended distribution is `p_i(T) = exp(z_i / T) / sum_j exp(z_j / T)`. Use a numerically stable implementation, validate the inputs, and make sure each row of the output sums to `1`.',
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
      'Temperature scaling applies `p_i(T) = exp(z_i / T) / Σ_j exp(z_j / T)`. Dividing by `T` changes the gaps between logits before softmax: at `T = 1` the distribution is unchanged, a low temperature makes the largest class more dominant and the distribution sharper, and a high temperature makes probabilities flatter and more uniform.',
      'The implementation subtracts the maximum scaled logit in each row before `exp`; this does not change the ratio because the same constant is subtracted from every class. After exponentiation, `normalizers.shape == (N, 1)` broadcasts across the class axis `(N, C)` during the final division.',
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
      'Sinusoidal positional encoding is a deterministic lookup table. For position `pos` and pair index `k`, `PE(pos, 2k) = sin(pos / 10000^(2k / dim))` and `PE(pos, 2k + 1) = cos(pos / 10000^(2k / dim))`; adjacent columns therefore share a frequency but use different phases.',
      'The implementation makes the shape arithmetic visible: positions has shape `(length, 1)`, the paired frequency table has shape `(1, ceil(dim / 2))`, and their broadcasted product creates one angle per position/frequency pair. Interleaving sine into `0::2` and cosine into `1::2` then returns `(length, dim)`, including the final unpaired sine column when `dim` is odd.',
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
    visual: {
      src: '/assets/images/code-patchify-layout.gif',
      alt: 'Animated 4 by 4 image diagram showing row-major patch tokens and the inverse reshape and permutation.',
      caption:
        'Read the layout from left to right: patch tokens are ordered by grid row, then grid column; each token contains all channels and its local P×P pixels. Unpatchify reverses those axis moves.',
    },
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
      'This problem is the inverse of patch extraction. Starting from `patches.shape == (B, N, C * P * P)`, reshape to `(B, grid_h, grid_w, C, P, P)` so the flat token index becomes explicit grid row and grid column axes.',
      'The permutation `(0, 3, 1, 4, 2, 5)` changes that layout to `(B, C, grid_h, P, grid_w, P)`: each grid axis now sits next to its local pixel axis. The final reshape collapses `(grid_h, P)` into `H` and `(grid_w, P)` into `W`, producing `(B, C, H, W)`.',
    ],
    solutionDiagram: `patches (B, N, C·P·P)
  reshape → (B, grid_h, grid_w, C, P, P)
  permute → (B, C, grid_h, P, grid_w, P)
  reshape → (B, C, H, W)

row-major token index: token = row * grid_w + column`,
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
    visual: {
      src: '/assets/images/code-patchify-layout.gif',
      alt: 'Animated 4 by 4 image diagram showing a row-major patch grid flattened into four patch tokens.',
      caption:
        'Patchify first exposes `(grid_h, P, grid_w, P)`, then permutes to put the grid axes first. Only after that permutation is each P×P patch flattened into one token.',
    },
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
      'The core trick is to expose the image grid as `(B, C, grid_h, P, grid_w, P)`. The original image shape `(B, C, H, W)` is only being re-labeled because `H = grid_h * P` and `W = grid_w * P`.',
      'The permutation `(0, 2, 4, 1, 3, 5)` produces `(B, grid_h, grid_w, C, P, P)`, which places the row-major patch grid before each patch’s channel and local-pixel data. Flattening the last four axes yields `(B, N, C * P * P)` without mixing neighboring patches.',
    ],
    solutionDiagram: `images (B, C, H, W)
  reshape → (B, C, grid_h, P, grid_w, P)
  permute → (B, grid_h, grid_w, C, P, P)
  reshape → (B, N, C·P·P)

N = grid_h · grid_w; token order is row-major over the grid`,
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
      'The angle line comes from the standard RoPE frequency schedule: `angles = torch.arange(seq_len, dtype=torch.float64)[:, None] * (10000.0 ** (-2 * pair / dim))[None, :]`. `torch.arange(seq_len)[:, None]` has shape `(T, 1)` and the inverse-frequency term has shape `(1, D / 2)`, so broadcasting creates `angles.shape == (T, D / 2)`: every position gets one angle for every adjacent feature pair.',
      'The slices use Python’s `start:stop:step` convention on the last axis. `x[..., 0::2]` starts at index `0` and takes every second value, so it selects indices `0, 2, 4, 6, 8, ...`—the even positions. `x[..., 1::2]` starts at index `1` and takes every second value, so it selects indices `1, 3, 5, 7, 9, ...`—the odd positions. Pairing those two views lets the code apply the 2D rotation formula to each neighboring pair.',
      'The sine and cosine tables are reshaped to `(1, T, 1, D / 2)` so the same position-dependent rotation broadcasts over batch and head dimensions.',
    ],
    solutionDiagram: `For each position t and pair k:
  angle[t, k] = t · 10000^(-2k / D)
  [x_even, x_odd] → [x_even cos - x_odd sin,
                     x_even sin + x_odd cos]

x[..., 0::2] = even channels: 0, 2, 4, ...
x[..., 1::2] = odd channels:  1, 3, 5, ...`,
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
      'Both exercises use the same core equation: `Attention(Q, K, V) = softmax(QKᵀ / √D_head) V`. The code projects inputs, reshapes `(B, T, D_model)` into `(B, H, T, D_head)`, computes scores `(B, H, query_length, key_length)`, applies the mask before softmax, mixes values, then permutes and reshapes back to `(B, T, D_model)`.',
      'For self-attention, the same sequence supplies all three inputs: `Q, K, V` come from `x`, so `query_length = key_length = T` and scores have shape `(B, H, T, T)`. A token can read from other tokens in that sequence, subject to the mask.',
      'The mask and stable softmax are the important implementation details: blocked scores become `-inf`, and subtracting a row maximum prevents overflow. The final output projection preserves the original model width.',
    ],
    solutionDiagram: `Self-attention:
x (B, T, D) ──┬─ Q (B, H, T, Dh)
              ├─ K (B, H, T, Dh) → scores (B, H, T, T)
              └─ V (B, H, T, Dh)

Cross-attention uses the same path, but query_x supplies Q and
context_x supplies K,V:
query (B, Tq, D) + context (B, Tk, D)
→ scores (B, H, Tq, Tk) → output (B, Tq, D)`,
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
      'Cross-attention is the same primitive as self-attention, with one deliberate change: `Q` comes from `query_x`, while `K` and `V` come from `context_x`. That lets one sequence read another sequence—for example, decoder tokens reading encoder outputs.',
      'The equation and most shapes are unchanged: `Attention(Q, K, V) = softmax(QKᵀ / √D_head) V`, but `query_x.shape == (B, Tq, D)` and `context_x.shape == (B, Tk, D)`. Therefore `Q` is `(B, H, Tq, D_head)`, `K,V` are `(B, H, Tk, D_head)`, scores are `(B, H, Tq, Tk)`, and the output is `(B, Tq, D)`.',
      'That shape contrast is the whole distinction: self-attention compares every token with the same sequence length `T`, while cross-attention compares each query token with all `Tk` context tokens. Masking still happens on the score matrix before the stable softmax.',
    ],
    solutionDiagram: `Self:  Q,K,V ← one sequence x
      scores: (B, H, T, T)

Cross: Q ← query_x (B, Tq, D)
       K,V ← context_x (B, Tk, D)
       scores: (B, H, Tq, Tk)

Only the source of K,V and the key length change.`,
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
  {
    id: 'l1-regression-loss',
    order: 19,
    title: 'L1 regression loss',
    difficulty: 'Easy',
    summary: 'Compute mean absolute error and explain why its linear penalty is robust to outliers.',
    prompt: [
      'Write `l1_loss(prediction, target)` to return the mean absolute error between two tensors.',
      'Keep the implementation vectorized and reject mismatched shapes.',
    ],
    signature: `def l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    ...`,
    requirements: [
      '`prediction` and `target` have the same shape.',
      'Return one scalar equal to the mean absolute error.',
      'Raise `ValueError` when the shapes differ.',
    ],
    examples: [
      {
        label: 'Example',
        lines: ['prediction = [1, 2, 10]', 'target = [2, 2, 7]'],
        result: '1.33333',
      },
    ],
    hint: [
      'Subtract the target from the prediction and take the absolute value.',
      'Reduce all residuals with a mean.',
    ],
    solutionNotes: [
      'L1 loss is `L = (1 / K) Σ_i |prediction_i - target_i|`, where `K` is the number of entries. The implementation computes the residual elementwise, takes its absolute value, then reduces every entry to one scalar with `torch.mean`.',
      'Its penalty grows linearly rather than quadratically, which makes a large residual matter without allowing it to dominate as strongly as L2 loss.',
    ],
    solutionCode: `import torch

def l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    prediction = torch.as_tensor(prediction, dtype=torch.float64)
    target = torch.as_tensor(target, dtype=torch.float64)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    return torch.mean(torch.abs(prediction - target))

prediction = torch.tensor([1.0, 2.0, 10.0])
target = torch.tensor([2.0, 2.0, 7.0])
print(l1_loss(prediction, target).item())`,
    starterCode: `import torch

def l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    # TODO: compute the mean absolute error.
    raise NotImplementedError("Implement l1_loss")

prediction = torch.tensor([1.0, 2.0, 10.0])
target = torch.tensor([2.0, 2.0, 7.0])
print(l1_loss(prediction, target).item())`,
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'Regression', 'Losses'],
  },
  {
    id: 'smooth-l1-huber-loss',
    order: 20,
    title: 'Smooth L1 / Huber loss',
    difficulty: 'Medium',
    summary: 'Implement a piecewise loss that is quadratic near zero and linear for large residuals.',
    prompt: [
      'Write `huber_loss(prediction, target, delta)` using the quadratic branch for `|error| <= delta` and the linear branch otherwise.',
      'Return the mean loss over all entries and validate the shape and positive transition parameter.',
    ],
    signature: `def huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    ...`,
    requirements: [
      'Use `0.5 * error**2` when `|error| <= delta`.',
      'Use `delta * (|error| - 0.5 * delta)` otherwise.',
      'Return the mean over all entries.',
      'Raise `ValueError` for mismatched shapes or non-positive `delta`.',
    ],
    examples: [
      {
        label: 'Example',
        lines: ['prediction = [0, 2, 4]', 'target = [0, 0, 0]', 'delta = 1'],
        result: '1.66667',
      },
    ],
    hint: [
      'Compute the quadratic and linear tensors independently.',
      '`torch.where` selects the branch elementwise without a Python loop.',
    ],
    solutionNotes: [
      'With error `e = prediction - target`, Huber loss is `L_delta(e) = 0.5 e²` when `|e| <= delta`, and `delta (|e| - 0.5 delta)` otherwise. The returned scalar is the mean of this piecewise value over all entries.',
      'Huber loss keeps L2’s smooth gradient for small errors but switches to L1-like growth after `delta`, limiting the influence of large mistakes. Compute both branches once, select with `torch.where`, and reduce at the end.',
    ],
    solutionCode: `import torch

def huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    prediction = torch.as_tensor(prediction, dtype=torch.float64)
    target = torch.as_tensor(target, dtype=torch.float64)
    if prediction.shape != target.shape or delta <= 0:
        raise ValueError("prediction and target must match and delta must be positive")
    error = prediction - target
    magnitude = torch.abs(error)
    quadratic = 0.5 * error ** 2
    linear = delta * (magnitude - 0.5 * delta)
    return torch.mean(torch.where(magnitude <= delta, quadratic, linear))

prediction = torch.tensor([0.0, 2.0, 4.0])
target = torch.zeros(3)
print(huber_loss(prediction, target).item())`,
    starterCode: `import torch

def huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    # TODO: implement the quadratic and linear branches of Huber loss.
    raise NotImplementedError("Implement huber_loss")

prediction = torch.tensor([0.0, 2.0, 4.0])
target = torch.zeros(3)
print(huber_loss(prediction, target).item())`,
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'Regression', 'Losses'],
  },
  {
    id: 'binary-cross-entropy-from-probabilities',
    order: 21,
    title: 'Binary cross-entropy from probabilities',
    difficulty: 'Easy',
    summary: 'Compute binary cross-entropy from probabilities while keeping logarithms finite.',
    prompt: [
      'Write `binary_cross_entropy(probability, target)` for elementwise binary targets in `{0, 1}`.',
      'Clamp probabilities before taking logarithms, then return the mean loss. In production, explain why logits plus `binary_cross_entropy_with_logits` are usually preferable.',
    ],
    signature: `def binary_cross_entropy(
    probability: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    ...`,
    requirements: [
      '`probability` and `target` have the same shape.',
      'Targets contain only `0` and `1`.',
      'Clamp probabilities to `[eps, 1 - eps]` before taking logs.',
      'Return the mean binary cross-entropy.',
    ],
    examples: [
      {
        label: 'Example',
        lines: ['probability = [0.9, 0.2]', 'target = [1, 0]'],
        result: '0.16425',
      },
    ],
    hint: [
      'Use `-(target * log(p) + (1 - target) * log(1 - p))`.',
      'Clamp once before both logarithms.',
    ],
    solutionNotes: [
      'Binary cross-entropy is `L = -(1 / K) Σ_i [y_i log(p_i) + (1 - y_i) log(1 - p_i)]`. When `y_i = 1`, only `-log(p_i)` remains; when `y_i = 0`, only `-log(1 - p_i)` remains.',
      'The clamp prevents `log(0)` before either logarithm is evaluated. For model training, logits are normally better because the fused logits loss avoids explicitly forming probabilities.',
    ],
    solutionCode: `import torch

def binary_cross_entropy(
    probability: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    probability = torch.as_tensor(probability, dtype=torch.float64)
    target = torch.as_tensor(target, dtype=torch.float64)
    if probability.shape != target.shape:
        raise ValueError("probability and target must have the same shape")
    if bool(torch.any((target != 0) & (target != 1))):
        raise ValueError("target must contain only 0 and 1")
    probability = torch.clamp(probability, min=eps, max=1 - eps)
    loss = -target * torch.log(probability) - (1 - target) * torch.log(1 - probability)
    return torch.mean(loss)

probability = torch.tensor([0.9, 0.2])
target = torch.tensor([1.0, 0.0])
print(binary_cross_entropy(probability, target).item())`,
    starterCode: `import torch

def binary_cross_entropy(
    probability: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    # TODO: clamp probabilities, apply the binary cross-entropy formula, and mean-reduce.
    raise NotImplementedError("Implement binary_cross_entropy")

probability = torch.tensor([0.9, 0.2])
target = torch.tensor([1.0, 0.0])
print(binary_cross_entropy(probability, target).item())`,
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'Classification', 'Losses'],
  },
  {
    id: 'single-box-iou',
    order: 22,
    title: 'Single-box IoU',
    difficulty: 'Easy',
    summary: 'Compute intersection-over-union for two `[x1, y1, x2, y2]` boxes.',
    prompt: [
      'Write `box_iou(box_a, box_b)` for two axis-aligned boxes in corner format.',
      'Clamp non-overlapping intersection dimensions to zero and define IoU as `0.0` when the union has zero area.',
    ],
    signature: `def box_iou(
    box_a: torch.Tensor,
    box_b: torch.Tensor,
) -> torch.Tensor:
    ...`,
    requirements: [
      'Both inputs have shape `(4,)` and valid corner ordering.',
      'Compute `intersection / union`.',
      'Return zero for non-overlap and zero-area unions.',
    ],
    examples: [
      {
        label: 'Example',
        lines: ['box_a = [0, 0, 2, 2]', 'box_b = [1, 1, 3, 3]'],
        result: '0.14286',
      },
    ],
    hint: [
      'The intersection uses maximum top-left and minimum bottom-right corners.',
      'Union is `area_a + area_b - intersection` because overlap was counted twice.',
    ],
    solutionNotes: [
      'The geometry is two reductions: intersect the corners, then compute each box area and subtract the overlap once from the sum.',
      'Clamping the intersection dimensions handles disjoint boxes; guarding the union gives a defined zero result for degenerate boxes.',
    ],
    solutionCode: `import torch

def box_iou(
    box_a: torch.Tensor,
    box_b: torch.Tensor,
) -> torch.Tensor:
    box_a = torch.as_tensor(box_a, dtype=torch.float64)
    box_b = torch.as_tensor(box_b, dtype=torch.float64)
    if box_a.shape != (4,) or box_b.shape != (4,):
        raise ValueError("boxes must have shape (4,)")
    if bool(torch.any(box_a[2:] < box_a[:2])) or bool(torch.any(box_b[2:] < box_b[:2])):
        raise ValueError("boxes must use x2 >= x1 and y2 >= y1")
    top_left = torch.maximum(box_a[:2], box_b[:2])
    bottom_right = torch.minimum(box_a[2:], box_b[2:])
    intersection_size = torch.clamp(bottom_right - top_left, min=0.0)
    intersection = intersection_size[0] * intersection_size[1]
    size_a = box_a[2:] - box_a[:2]
    size_b = box_b[2:] - box_b[:2]
    area_a = size_a[0] * size_a[1]
    area_b = size_b[0] * size_b[1]
    union = area_a + area_b - intersection
    safe_union = torch.where(union > 0, union, torch.ones_like(union))
    return torch.where(union > 0, intersection / safe_union, torch.zeros_like(union))

box_a = torch.tensor([0.0, 0.0, 2.0, 2.0])
box_b = torch.tensor([1.0, 1.0, 3.0, 3.0])
print(box_iou(box_a, box_b).item())`,
    starterCode: `import torch

def box_iou(
    box_a: torch.Tensor,
    box_b: torch.Tensor,
) -> torch.Tensor:
    # TODO: compute intersection area divided by union area.
    raise NotImplementedError("Implement box_iou")

box_a = torch.tensor([0.0, 0.0, 2.0, 2.0])
box_b = torch.tensor([1.0, 1.0, 3.0, 3.0])
print(box_iou(box_a, box_b).item())`,
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'Computer Vision', 'Bounding Boxes'],
  },
  {
    id: 'pairwise-squared-distance',
    order: 23,
    title: 'Pairwise squared distance',
    difficulty: 'Medium',
    summary: 'Build an `(N, M)` matrix of squared Euclidean distances between two point sets.',
    prompt: [
      'Write `pairwise_squared_distance(x, y)` for `x.shape == (N, D)` and `y.shape == (M, D)`.',
      'Use broadcasting or the equivalent norm-and-matmul identity, and return one distance for every pair of rows.',
    ],
    signature: `def pairwise_squared_distance(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    ...`,
    requirements: [
      'Return shape `(N, M)`.',
      'Compute `||x_i - y_j||^2` without Python loops.',
      'Raise `ValueError` when either input is not 2D or the feature dimensions differ.',
    ],
    examples: [
      {
        label: 'Example',
        lines: ['x = [[0, 0], [1, 1]]', 'y = [[1, 0], [2, 2]]'],
        result: '[[1.0, 8.0], [1.0, 2.0]]',
      },
    ],
    hint: [
      'Insert singleton axes to make `(N, 1, D)` and `(1, M, D)`.',
      'For a lower-memory version, use `||x||^2 + ||y||^2 - 2 x y^T`.',
    ],
    solutionNotes: [
      'The target is `D[i, j] = ||x_i - y_j||²`. Expanding the square gives `||x_i||² + ||y_j||² - 2 x_i · y_j`, which lets one matrix multiplication compute all pairwise dot products at once.',
      'Here `x_squared` has shape `(N, 1)`, `y_squared` is reshaped to `(1, M)`, and `x @ y.T` has shape `(N, M)`. Therefore `distances = x_squared + y_squared - 2 * x @ torch.transpose(y, 0, 1)` broadcasts the two norm terms into an `(N, M)` matrix and subtracts twice the dot product for each `(i, j)` pair.',
      'The norm identity avoids materializing the direct broadcasted difference `(N, M, D)`, so it is the better production implementation when the point sets are large. Small negative values from floating-point roundoff are clamped to zero.',
    ],
    solutionCode: `import torch

def pairwise_squared_distance(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    x = torch.as_tensor(x, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("x and y must be 2D with the same feature dimension")
    x_squared = torch.sum(x * x, dim=1, keepdim=True)
    y_squared = torch.sum(y * y, dim=1)[None, :]
    distances = x_squared + y_squared - 2 * x @ torch.transpose(y, 0, 1)
    return torch.clamp(distances, min=0.0)

x = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
y = torch.tensor([[1.0, 0.0], [2.0, 2.0]])
print(pairwise_squared_distance(x, y))`,
    starterCode: `import torch

def pairwise_squared_distance(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    # TODO: broadcast the pair dimension and reduce squared differences over D.
    raise NotImplementedError("Implement pairwise_squared_distance")

x = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
y = torch.tensor([[1.0, 0.0], [2.0, 2.0]])
print(pairwise_squared_distance(x, y))`,
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'Broadcasting', 'Geometry'],
  },
  {
    id: 'masked-mean',
    order: 24,
    title: 'Masked mean',
    difficulty: 'Easy',
    summary: 'Reduce padded `(B, N, D)` features while excluding invalid rows from each batch mean.',
    prompt: [
      'Write `masked_mean(features, mask)` for features shaped `(B, N, D)` and a validity mask shaped `(B, N)`.',
      'Return one `(B, D)` mean per batch item. A row with no valid entries should return zeros rather than divide by zero.',
    ],
    signature: `def masked_mean(
    features: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    ...`,
    requirements: [
      'Broadcast the mask across the feature dimension.',
      'Sum over `N` and divide by the number of valid rows per batch item.',
      'Clamp the count so an all-padding item returns zero.',
    ],
    examples: [
      {
        label: 'Example',
        lines: ['features = [[[1, 2], [3, 4], [0, 0]]]', 'mask = [[1, 1, 0]]'],
        result: '[[2.0, 3.0]]',
      },
    ],
    hint: [
      'Use `mask.unsqueeze(-1)` to turn `(B, N)` into `(B, N, 1)`.',
      'Sum masked features and masked counts over the same axis.',
    ],
    solutionNotes: [
      'The mask is a selector, not a new reduction rule: cast it to the feature dtype, multiply it into `(B, N, D)`, and reduce over `N`.',
      'Keeping the count as `(B, 1)` makes the final division broadcast across `D`; clamping only the denominator makes empty batches return zero.',
    ],
    solutionCode: `import torch

def masked_mean(
    features: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    features = torch.as_tensor(features, dtype=torch.float64)
    mask = torch.as_tensor(mask)
    if features.ndim != 3 or mask.shape != features.shape[:2]:
        raise ValueError("features must be (B, N, D) and mask must be (B, N)")
    weights = torch.unsqueeze(torch.as_tensor(mask, dtype=features.dtype), -1)
    total = torch.sum(features * weights, dim=1)
    count = torch.clamp(torch.sum(weights, dim=1), min=1.0)
    return total / count

features = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]]])
mask = torch.tensor([[1, 1, 0]])
print(masked_mean(features, mask))`,
    starterCode: `import torch

def masked_mean(
    features: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    # TODO: exclude masked rows from the mean without a Python loop.
    raise NotImplementedError("Implement masked_mean")

features = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]]])
mask = torch.tensor([[1, 1, 0]])
print(masked_mean(features, mask))`,
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'Masking', 'Reductions'],
  },
  {
    id: 'top-k-gather',
    order: 25,
    title: 'Top-k features with gather',
    difficulty: 'Medium',
    summary: 'Select the top-k rows of batched features using per-batch scores and explicit gather shapes.',
    prompt: [
      'Write `topk_features(scores, features, k)` where `scores.shape == (B, N)` and `features.shape == (B, N, D)`.',
      'Return the feature rows corresponding to each batch item’s top-k scores in descending score order.',
    ],
    signature: `def topk_features(
    scores: torch.Tensor,
    features: torch.Tensor,
    k: int,
) -> torch.Tensor:
    ...`,
    requirements: [
      'Return shape `(B, k, D)`.',
      'Each batch item may select different rows.',
      'Use `unsqueeze`, broadcasting or expansion, and `torch.gather` rather than a Python loop.',
      'Raise `ValueError` when shapes or `k` are invalid.',
    ],
    examples: [
      {
        label: 'Example',
        lines: ['scores = [[0.2, 0.9, 0.4]]', 'features = [[[2, 0], [9, 0], [4, 0]]]', 'k = 2'],
        result: '[[[9, 0], [4, 0]]]',
      },
    ],
    hint: [
      'Sort indices along `N` and slice the first `k` columns.',
      'Turn `(B, k)` indices into `(B, k, D)` indices before gathering along dimension `1`.',
    ],
    solutionNotes: [
      'The scores produce row indices, but `gather` needs an index for every feature channel. Adding a last axis and broadcasting it from `(B, k, 1)` to `(B, k, D)` supplies exactly that shape.',
      'The batch axis is preserved throughout, so each example can choose different rows without flattening or looping.',
    ],
    solutionCode: `import torch

def topk_features(
    scores: torch.Tensor,
    features: torch.Tensor,
    k: int,
) -> torch.Tensor:
    scores = torch.as_tensor(scores, dtype=torch.float64)
    features = torch.as_tensor(features, dtype=torch.float64)
    if scores.ndim != 2 or features.ndim != 3 or features.shape[:2] != scores.shape:
        raise ValueError("scores must be (B, N) and features must be (B, N, D)")
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= scores.shape[1]:
        raise ValueError("k must be between 1 and N")
    indices = torch.argsort(scores, dim=1, descending=True)[:, :k]
    gather_indices = torch.broadcast_to(indices[:, :, None], (scores.shape[0], k, features.shape[2]))
    return torch.gather(features, dim=1, index=gather_indices)

scores = torch.tensor([[0.2, 0.9, 0.4]])
features = torch.tensor([[[2.0, 0.0], [9.0, 0.0], [4.0, 0.0]]])
print(topk_features(scores, features, k=2))`,
    starterCode: `import torch

def topk_features(
    scores: torch.Tensor,
    features: torch.Tensor,
    k: int,
) -> torch.Tensor:
    # TODO: create (B, k, D) gather indices from (B, k) top-score indices.
    raise NotImplementedError("Implement topk_features")

scores = torch.tensor([[0.2, 0.9, 0.4]])
features = torch.tensor([[[2.0, 0.0], [9.0, 0.0], [4.0, 0.0]]])
print(topk_features(scores, features, k=2))`,
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'Indexing', 'Gather'],
  },
  {
    id: 'dice-loss',
    order: 26,
    title: 'Dice loss',
    difficulty: 'Medium',
    summary: 'Implement a batch-wise Dice loss that emphasizes overlap for imbalanced segmentation masks.',
    prompt: [
      'Write `dice_loss(prediction, target)` for batched segmentation probabilities and binary targets.',
      'Flatten spatial dimensions per example, compute the smoothed Dice coefficient, and return `1 - mean(Dice)`.',
    ],
    signature: `def dice_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    ...`,
    requirements: [
      'Preserve the first dimension as batch size and flatten the remaining dimensions.',
      'Use `2 * intersection / (prediction_sum + target_sum)` with `eps` smoothing.',
      'Return one scalar loss.',
    ],
    examples: [
      {
        label: 'Example',
        lines: ['prediction = [[[1, 0], [0, 1]]]', 'target = [[[1, 0], [1, 0]]]'],
        result: '0.5',
      },
    ],
    hint: [
      'Use `reshape(batch_size, -1)` to keep one row per image.',
      'Elementwise multiplication gives the soft intersection.',
    ],
    solutionNotes: [
      'For each batch item, Dice is `Dice = (2 · Σ(prediction · target) + eps) / (Σ prediction + Σ target + eps)`, and the loss is `1 - mean(Dice)`. The factor of `2` rewards shared foreground mass while the denominator counts the predicted and target mass separately.',
      'Dice measures overlap directly, so it is less dominated by abundant background pixels than raw accuracy. Flattening each example makes the reduction independent of spatial rank, and `eps` keeps empty or nearly empty masks finite.',
    ],
    solutionCode: `import torch

def dice_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    prediction = torch.as_tensor(prediction, dtype=torch.float64)
    target = torch.as_tensor(target, dtype=torch.float64)
    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("prediction and target must have the same batched shape")
    prediction = prediction.reshape(prediction.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    intersection = torch.sum(prediction * target, dim=1)
    denominator = torch.sum(prediction, dim=1) + torch.sum(target, dim=1)
    dice = (2 * intersection + eps) / (denominator + eps)
    return 1 - torch.mean(dice)

prediction = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
target = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
print(dice_loss(prediction, target).item())`,
    starterCode: `import torch

def dice_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    # TODO: flatten each batch item, compute smoothed Dice, and return 1 - mean(Dice).
    raise NotImplementedError("Implement dice_loss")

prediction = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
target = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
print(dice_loss(prediction, target).item())`,
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'Segmentation', 'Losses'],
  },
  {
    id: 'segmentation-iou-loss',
    order: 27,
    title: 'Segmentation IoU loss',
    difficulty: 'Medium',
    summary: 'Compute a differentiable soft IoU loss over batched segmentation masks.',
    prompt: [
      'Write `iou_loss(prediction, target)` using soft masks rather than thresholding the predictions.',
      'Flatten each batch item, compute intersection and union, and return the mean of `1 - IoU`.',
    ],
    signature: `def iou_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    ...`,
    requirements: [
      'Preserve the first dimension as batch size.',
      'Use `union = prediction_sum + target_sum - intersection`.',
      'Use `eps` to keep zero-union examples finite.',
    ],
    examples: [
      {
        label: 'Example',
        lines: ['prediction = [[[1, 0], [0, 1]]]', 'target = [[[1, 0], [1, 0]]]'],
        result: '0.66667',
      },
    ],
    hint: [
      'IoU is the same intersection-over-union geometry used for boxes, applied to mask entries.',
      'Keep the batch axis while flattening all remaining dimensions.',
    ],
    solutionNotes: [
      'For each batch item, soft IoU is `IoU = (Σ(prediction · target) + eps) / (Σ prediction + Σ target - Σ(prediction · target) + eps)`, and the loss is `mean(1 - IoU)`. The union subtracts the intersection once because the overlap appears in both sums.',
      'Soft IoU replaces set membership with probabilities, so the loss remains usable in gradient-based training. Flattening keeps one score per example, and smoothing only prevents an undefined empty union.',
    ],
    solutionCode: `import torch

def iou_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    prediction = torch.as_tensor(prediction, dtype=torch.float64)
    target = torch.as_tensor(target, dtype=torch.float64)
    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("prediction and target must have the same batched shape")
    prediction = prediction.reshape(prediction.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    intersection = torch.sum(prediction * target, dim=1)
    union = torch.sum(prediction, dim=1) + torch.sum(target, dim=1) - intersection
    return torch.mean(1 - (intersection + eps) / (union + eps))

prediction = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
target = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
print(iou_loss(prediction, target).item())`,
    starterCode: `import torch

def iou_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    # TODO: compute soft intersection-over-union per batch item.
    raise NotImplementedError("Implement iou_loss")

prediction = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
target = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
print(iou_loss(prediction, target).item())`,
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'Segmentation', 'Losses'],
  },
  {
    id: 'focal-loss',
    order: 28,
    title: 'Focal loss',
    difficulty: 'Medium',
    summary: 'Implement binary focal loss so easy negatives contribute less under extreme class imbalance.',
    prompt: [
      'Write `focal_loss(logits, target, gamma)` for binary targets using logits as input.',
      'Compute `p_t`, downweight it by `(1 - p_t) ** gamma`, and use a stable enough probability path for ordinary finite logits.',
    ],
    signature: `def focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    ...`,
    requirements: [
      'Targets contain only `0` and `1` and match `logits` shape.',
      'Use `p_t = p` for positive targets and `1 - p` for negative targets.',
      'Return the mean `-(1 - p_t)^gamma * log(p_t)`.',
      'Raise `ValueError` for negative `gamma` or invalid targets.',
    ],
    examples: [
      {
        label: 'Example',
        lines: ['logits = [4.0, -0.4]', 'target = [1, 0]', 'gamma = 2'],
        result: 'approximately 0.092',
      },
    ],
    hint: [
      'A sigmoid converts logits to probabilities; clamp the result before `log`.',
      'The modulating factor is close to zero for an easy example with `p_t` near one.',
    ],
    solutionNotes: [
      'For a sigmoid probability `p = sigmoid(logit)`, define `p_t = p` when `target = 1` and `p_t = 1 - p` when `target = 0`. Focal loss is `L = -(1 / K) Σ_i (1 - p_t,i)^gamma log(p_t,i)`.',
      'The factor `(1 - p_t)^gamma` is close to zero for a correct, easy example with `p_t` near one, while hard examples retain more weight. In production, use a fused logits-based implementation to avoid explicitly materializing probabilities.',
    ],
    solutionCode: `import torch

def focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    logits = torch.as_tensor(logits, dtype=torch.float64)
    target = torch.as_tensor(target, dtype=torch.float64)
    if logits.shape != target.shape or gamma < 0:
        raise ValueError("logits and target must match and gamma must be non-negative")
    if bool(torch.any((target != 0) & (target != 1))):
        raise ValueError("target must contain only 0 and 1")
    probability = 1 / (1 + torch.exp(-torch.clamp(logits, min=-60.0, max=60.0)))
    p_t = torch.where(target == 1, probability, 1 - probability)
    p_t = torch.clamp(p_t, min=1e-8, max=1.0)
    return torch.mean(-((1 - p_t) ** gamma) * torch.log(p_t))

logits = torch.tensor([4.0, -0.4])
target = torch.tensor([1.0, 0.0])
print(focal_loss(logits, target).item())`,
    starterCode: `import torch

def focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    # TODO: compute p_t and apply the focal modulating factor.
    raise NotImplementedError("Implement focal_loss")

logits = torch.tensor([4.0, -0.4])
target = torch.tensor([1.0, 0.0])
print(focal_loss(logits, target).item())`,
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'Detection', 'Class Imbalance'],
  },
  {
    id: 'weighted-box-regression-loss',
    order: 29,
    title: 'Weighted bounding-box regression loss',
    difficulty: 'Easy',
    summary: 'Combine per-coordinate regression errors with explicit weights for position, size, angle, and velocity.',
    prompt: [
      'Write `weighted_box_regression_loss(prediction, target, weights)` as a weighted mean absolute error over the final coordinate dimension.',
      'The weights may reflect different semantic groups such as position, size, yaw, and velocity.',
    ],
    signature: `def weighted_box_regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    ...`,
    requirements: [
      '`prediction` and `target` share shape `(..., D)`.',
      '`weights` has shape `(D,)` and contains non-negative values.',
      'Return the mean weighted L1 error.',
      'Raise `ValueError` on incompatible shapes or negative weights.',
    ],
    examples: [
      {
        label: 'Example',
        lines: ['prediction = [[1, 2, 3]]', 'target = [[0, 0, 0]]', 'weights = [1, 2, 0.5]'],
        result: '2.16667',
      },
    ],
    hint: [
      'Subtract and take absolute values before multiplying by `weights`.',
      'The last-axis weight vector broadcasts across every leading dimension.',
    ],
    solutionNotes: [
      'Different box coordinates can have different units and importance, so a single unweighted error can let one group dominate. A final-axis weight vector makes that tradeoff explicit.',
      'The code keeps the weighting separate from the reduction: broadcast the weights over the coordinate axis, then mean all weighted residuals.',
    ],
    solutionCode: `import torch

def weighted_box_regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    prediction = torch.as_tensor(prediction, dtype=torch.float64)
    target = torch.as_tensor(target, dtype=torch.float64)
    weights = torch.as_tensor(weights, dtype=torch.float64)
    if prediction.shape != target.shape or prediction.ndim == 0 or weights.shape != prediction.shape[-1:]:
        raise ValueError("prediction, target, and weights have incompatible shapes")
    if bool(torch.any(weights < 0)):
        raise ValueError("weights must be non-negative")
    return torch.mean(torch.abs(prediction - target) * weights)

prediction = torch.tensor([[1.0, 2.0, 3.0]])
target = torch.zeros_like(prediction)
weights = torch.tensor([1.0, 2.0, 0.5])
print(weighted_box_regression_loss(prediction, target, weights).item())`,
    starterCode: `import torch

def weighted_box_regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    # TODO: compute weighted absolute error over the coordinate dimension.
    raise NotImplementedError("Implement weighted_box_regression_loss")

prediction = torch.tensor([[1.0, 2.0, 3.0]])
target = torch.zeros_like(prediction)
weights = torch.tensor([1.0, 2.0, 0.5])
print(weighted_box_regression_loss(prediction, target, weights).item())`,
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'Regression', 'Detection'],
  },
  {
    id: 'wrapped-angular-difference',
    order: 30,
    title: 'Wrapped angular difference',
    difficulty: 'Easy',
    summary: 'Compute the shortest signed difference between angles without a discontinuity at ±π.',
    prompt: [
      'Write `angular_difference(prediction, target)` for angles in radians.',
      'A prediction of `179°` and target of `-179°` should differ by about `2°`, not `358°`.',
    ],
    signature: `def angular_difference(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    ...`,
    requirements: [
      'Inputs have the same shape.',
      'Return values in `[-π, π]` using the shortest signed rotation.',
      'Do not use a naive subtraction as the final answer.',
    ],
    examples: [
      {
        label: 'Example',
        lines: ['prediction = 179°', 'target = -179°'],
        result: 'approximately -2°',
      },
    ],
    hint: [
      'First compute the raw difference.',
      '`atan2(sin(difference), cos(difference))` wraps it to the principal interval.',
    ],
    solutionNotes: [
      'Angles live on a circle, so ordinary subtraction mistakes the seam at `±π` for a long physical rotation.',
      'The sine and cosine preserve the direction on that circle, and `atan2` recovers the equivalent angle in the principal interval.',
    ],
    solutionCode: `import torch

def angular_difference(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    prediction = torch.as_tensor(prediction, dtype=torch.float64)
    target = torch.as_tensor(target, dtype=torch.float64)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    difference = prediction - target
    return torch.atan2(torch.sin(difference), torch.cos(difference))

prediction = torch.tensor(179.0 * 3.141592653589793 / 180)
target = torch.tensor(-179.0 * 3.141592653589793 / 180)
print(angular_difference(prediction, target) * 180 / 3.141592653589793)`,
    starterCode: `import torch

def angular_difference(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    # TODO: wrap the raw difference to the shortest signed angle.
    raise NotImplementedError("Implement angular_difference")

prediction = torch.tensor(179.0 * 3.141592653589793 / 180)
target = torch.tensor(-179.0 * 3.141592653589793 / 180)
print(angular_difference(prediction, target) * 180 / 3.141592653589793)`,
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'Geometry', 'Autonomous Driving'],
  },
  {
    id: 'average-precision-from-matches',
    order: 31,
    title: 'Average precision from ranked matches',
    difficulty: 'Hard',
    summary: 'Compute one-class average precision from confidence-ranked true-positive matches.',
    prompt: [
      'Write `average_precision(scores, is_true_positive, num_ground_truth)` for detections from one class.',
      'Sort by confidence, sweep the ranked list, and sum precision at each newly recovered ground-truth object.',
    ],
    signature: `def average_precision(
    scores: torch.Tensor,
    is_true_positive: torch.Tensor,
    num_ground_truth: int,
) -> torch.Tensor:
    ...`,
    requirements: [
      '`scores` and `is_true_positive` are 1D and have the same length.',
      'Sort predictions by descending score before accumulating counts.',
      'Compute AP as the mean precision over true-positive ranks, normalized by `num_ground_truth`.',
      'Raise `ValueError` for invalid shapes or no ground-truth objects.',
    ],
    examples: [
      {
        label: 'Example',
        lines: ['scores = [0.9, 0.8, 0.7]', 'is_true_positive = [1, 0, 1]', 'num_ground_truth = 2'],
        result: '0.83333',
      },
    ],
    hint: [
      'Use descending `argsort` to create the evaluation order.',
      'A true positive contributes the precision at its rank; false positives only increase the denominator.',
    ],
    solutionNotes: [
      'After sorting detections by descending confidence, let `TP(k)` be the number of true positives in the first `k` predictions and let `m_k` be `1` only when rank `k` is a true positive. Then `precision(k) = TP(k) / k`, and `AP = (1 / num_ground_truth) Σ_k m_k · precision(k)`.',
      'The code follows that equation directly: `cumulative_tp` has one count per rank, `ranks` is `1..N`, and the boolean match mask selects only the ranks that contribute precision. False positives still increase the rank denominator, so they lower later precision even though they add no numerator.',
      'The ground-truth count normalizes the sum and makes the metric comparable across images or classes. Full mAP additionally averages this AP across classes and IoU thresholds.',
    ],
    solutionCode: `import torch

def average_precision(
    scores: torch.Tensor,
    is_true_positive: torch.Tensor,
    num_ground_truth: int,
) -> torch.Tensor:
    scores = torch.as_tensor(scores, dtype=torch.float64)
    is_true_positive = torch.as_tensor(is_true_positive, dtype=torch.bool)
    if scores.ndim != 1 or is_true_positive.shape != scores.shape or num_ground_truth <= 0:
        raise ValueError("scores and matches must be 1D with positive ground-truth count")
    order = torch.argsort(scores, descending=True)
    matches = is_true_positive[order]
    cumulative_tp = torch.cumsum(torch.as_tensor(matches, dtype=torch.float64), dim=0)
    ranks = torch.arange(scores.shape[0], dtype=torch.float64) + 1
    precision = cumulative_tp / ranks
    return torch.sum(precision * torch.as_tensor(matches, dtype=torch.float64)) / num_ground_truth

scores = torch.tensor([0.9, 0.8, 0.7])
matches = torch.tensor([1, 0, 1], dtype=torch.bool)
print(average_precision(scores, matches, num_ground_truth=2).item())`,
    starterCode: `import torch

def average_precision(
    scores: torch.Tensor,
    is_true_positive: torch.Tensor,
    num_ground_truth: int,
) -> torch.Tensor:
    # TODO: sort by confidence and average precision at true-positive ranks.
    raise NotImplementedError("Implement average_precision")

scores = torch.tensor([0.9, 0.8, 0.7])
matches = torch.tensor([1, 0, 1], dtype=torch.bool)
print(average_precision(scores, matches, num_ground_truth=2).item())`,
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'Detection', 'Metrics'],
  },
  {
    id: 'greedy-detection-matching',
    order: 32,
    title: 'Greedy prediction-to-ground-truth matching',
    difficulty: 'Hard',
    summary: 'Match confidence-ranked predictions to at most one ground-truth box using an IoU threshold.',
    prompt: [
      'Write `match_predictions(predictions, scores, ground_truth, iou_threshold)` and return the matched ground-truth index for each prediction, or `-1` for a false positive.',
      'Process predictions in descending confidence order; a ground-truth box can be matched only once.',
    ],
    signature: `def match_predictions(
    predictions: torch.Tensor,
    scores: torch.Tensor,
    ground_truth: torch.Tensor,
    iou_threshold: float,
) -> list[int]:
    ...`,
    requirements: [
      'Inputs have shapes `(N, 4)`, `(N,)`, and `(M, 4)`.',
      'For each prediction, choose the unmatched ground truth with highest IoU.',
      'Mark a match only when the best IoU is at least the threshold.',
      'Return `-1` for unmatched predictions and preserve prediction-index order in the output.',
    ],
    examples: [
      {
        label: 'Example',
        lines: [
          'predictions = [[0, 0, 2, 2], [0, 0, 2, 2]]',
          'scores = [0.9, 0.8]',
          'ground_truth = [[0, 0, 2, 2]]',
          'iou_threshold = 0.5',
        ],
        result: '[0, -1]',
      },
    ],
    hint: [
      'Compute a vectorized `(N, M)` IoU matrix first.',
      'Use a `set` of already claimed ground-truth indices while traversing sorted predictions.',
      'Do not let a second high-overlap prediction become a second true positive.',
    ],
    solutionNotes: [
      'Detection evaluation is not independent per prediction: once a ground-truth object is claimed by the highest-confidence matching prediction, later duplicates are false positives.',
      'The vectorized IoU calculation handles geometry; the small greedy loop handles the one-to-one assignment rule and keeps the confidence ordering explicit.',
    ],
    solutionCode: `import torch

def match_predictions(
    predictions: torch.Tensor,
    scores: torch.Tensor,
    ground_truth: torch.Tensor,
    iou_threshold: float,
) -> list[int]:
    predictions = torch.as_tensor(predictions, dtype=torch.float64)
    scores = torch.as_tensor(scores, dtype=torch.float64)
    ground_truth = torch.as_tensor(ground_truth, dtype=torch.float64)
    if predictions.ndim != 2 or predictions.shape[1] != 4 or ground_truth.ndim != 2 or ground_truth.shape[1] != 4:
        raise ValueError("predictions and ground_truth must have shape (N, 4) and (M, 4)")
    if scores.shape != (predictions.shape[0],) or not 0 <= iou_threshold <= 1:
        raise ValueError("scores or iou_threshold is invalid")

    top_left = torch.maximum(predictions[:, None, :2], ground_truth[None, :, :2])
    bottom_right = torch.minimum(predictions[:, None, 2:], ground_truth[None, :, 2:])
    size = torch.clamp(bottom_right - top_left, min=0.0)
    intersection = size[..., 0] * size[..., 1]
    area_pred = (predictions[:, 2] - predictions[:, 0]) * (predictions[:, 3] - predictions[:, 1])
    area_gt = (ground_truth[:, 2] - ground_truth[:, 0]) * (ground_truth[:, 3] - ground_truth[:, 1])
    union = area_pred[:, None] + area_gt[None, :] - intersection
    ious = torch.where(union > 0, intersection / torch.where(union > 0, union, torch.ones_like(union)), torch.zeros_like(union))

    matches = [-1] * predictions.shape[0]
    used = set()
    for prediction_index in torch.argsort(scores, descending=True).tolist():
        if ground_truth.shape[0] == 0:
            continue
        available = torch.as_tensor([ground_truth_index not in used for ground_truth_index in range(ground_truth.shape[0])], dtype=torch.bool)
        if not torch.any(available):
            continue
        candidate_ious = torch.where(available, ious[prediction_index], torch.full_like(ious[prediction_index], float('-inf')))
        best_gt = int(torch.argmax(candidate_ious).item())
        best_iou = float(candidate_ious[best_gt].item())
        if best_iou >= iou_threshold:
            matches[prediction_index] = best_gt
            used.add(best_gt)
    return matches

predictions = torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0]])
scores = torch.tensor([0.9, 0.8])
ground_truth = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
print(match_predictions(predictions, scores, ground_truth, 0.5))`,
    starterCode: `import torch

def match_predictions(
    predictions: torch.Tensor,
    scores: torch.Tensor,
    ground_truth: torch.Tensor,
    iou_threshold: float,
) -> list[int]:
    # TODO: compute pairwise IoU, then greedily claim each ground-truth box once.
    raise NotImplementedError("Implement match_predictions")

predictions = torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0]])
scores = torch.tensor([0.9, 0.8])
ground_truth = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
print(match_predictions(predictions, scores, ground_truth, 0.5))`,
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'Detection', 'Greedy Matching'],
  },
  {
    id: 'homogeneous-coordinate-transform',
    order: 33,
    title: '3D coordinate transform',
    difficulty: 'Medium',
    summary: 'Apply one 4×4 homogeneous transform to a batch of 3D points.',
    prompt: [
      'Write `transform_points(points, transform)` for `points.shape == (N, 3)` and a homogeneous transform shaped `(4, 4)`.',
      'Append a padding coordinate of one to every point, multiply by the transform, and return the transformed XYZ coordinates. The extra coordinate is what lets one matrix multiplication represent both rotation and translation.',
    ],
    signature: `def transform_points(
    points: torch.Tensor,
    transform: torch.Tensor,
) -> torch.Tensor:
    ...`,
    requirements: [
      'Return shape `(N, 3)`.',
      'Use one matrix multiplication for rotation and translation.',
      'Reject malformed point or transform shapes.',
    ],
    examples: [
      {
        label: 'Example',
        lines: ['points = [[1, 2, 3]]', 'transform = identity with translation [10, 20, 30]'],
        result: '[[11, 22, 33]]',
      },
    ],
    hint: [
      'Concatenate a column of ones to get `(N, 4)`.',
      'Multiply `transform @ points_h.T`, transpose back, and discard the fourth coordinate.',
    ],
    solutionNotes: [
      'The ordinary affine equation is `p′ = R p + t`. Homogeneous coordinates rewrite it as `[p′; 1] = [[R, t], [0, 1]] [p; 1]`, so translation becomes part of the same matrix multiplication as rotation.',
      'For `points.shape == (N, 3)`, `ones.shape == (N, 1)` and `torch.cat([points, ones], dim=1)` creates `homogeneous.shape == (N, 4)`. The transform expects points as columns, so `transform (4, 4) @ homogeneous.T (4, N)` produces `(4, N)`; transposing back gives `(N, 4)`, and `[:, :3]` removes the padding coordinate.',
      'The padding value is `1`, not `0`: multiplying the last transform column by one adds the translation vector `t` to every point. The final homogeneous coordinate is discarded because the task asks for XYZ only.',
    ],
    solutionDiagram: `point row:        [x, y, z]
pad one column:  [x, y, z, 1]   ← homogeneous point

transform (4,4) @ homogeneous.T (4,N)
      → transformed.T (N,4)
      → transformed[:, :3] (N,3)

last input value 1 activates the translation column`,
    solutionCode: `import torch

def transform_points(
    points: torch.Tensor,
    transform: torch.Tensor,
) -> torch.Tensor:
    points = torch.as_tensor(points, dtype=torch.float64)
    transform = torch.as_tensor(transform, dtype=torch.float64)
    if points.ndim != 2 or points.shape[1] != 3 or transform.shape != (4, 4):
        raise ValueError("points must be (N, 3) and transform must be (4, 4)")
    ones = torch.ones(points.shape[0], 1, dtype=points.dtype)
    homogeneous = torch.cat([points, ones], dim=1)
    transformed = torch.transpose(transform @ torch.transpose(homogeneous, 0, 1), 0, 1)
    return transformed[:, :3]

points = torch.tensor([[1.0, 2.0, 3.0]])
transform = torch.tensor([[1.0, 0, 0, 10], [0, 1.0, 0, 20], [0, 0, 1.0, 30], [0, 0, 0, 1.0]])
print(transform_points(points, transform))`,
    starterCode: `import torch

def transform_points(
    points: torch.Tensor,
    transform: torch.Tensor,
) -> torch.Tensor:
    # TODO: append homogeneous ones, transform, and return XYZ.
    raise NotImplementedError("Implement transform_points")

points = torch.tensor([[1.0, 2.0, 3.0]])
transform = torch.tensor([[1.0, 0, 0, 10], [0, 1.0, 0, 20], [0, 0, 1.0, 30], [0, 0, 0, 1.0]])
print(transform_points(points, transform))`,
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', '3D Geometry', 'Autonomous Driving'],
  },
  {
    id: 'batched-best-iou-match',
    order: 34,
    title: 'Batched best-IoU matching',
    difficulty: 'Hard',
    summary: 'For every predicted box in every batch item, find the ground-truth box with maximum IoU.',
    prompt: [
      'Write `best_iou_match(predictions, ground_truth)` for tensors shaped `(B, N, 4)` and `(B, M, 4)`.',
      'Return both the best IoU and the best ground-truth index for each predicted box.',
    ],
    signature: `def best_iou_match(
    predictions: torch.Tensor,
    ground_truth: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...`,
    requirements: [
      'Return `best_iou` and `best_gt` with shape `(B, N)`.',
      'Use broadcasting to form pairwise IoUs within each batch item.',
      'Use `argmax` over the ground-truth axis and gather the corresponding IoUs.',
      'Raise `ValueError` for incompatible batch or box shapes.',
    ],
    examples: [
      {
        label: 'Example',
        lines: [
          'predictions.shape = [B, N, 4]',
          'ground_truth.shape = [B, M, 4]',
        ],
        result: 'best_iou.shape = [B, N]; best_gt.shape = [B, N]',
      },
    ],
    hint: [
      'Use `predictions[:, :, None, :]` and `ground_truth[:, None, :, :]` to create `(B, N, M, 4)` pairs.',
      'After `argmax(dim=-1)`, add a final singleton axis before `torch.gather`.',
    ],
    solutionNotes: [
      'This is the full pattern in one exercise: batch-aware broadcasting creates every prediction–ground-truth pair, and the last axis is reduced to the best match.',
      'The shapes make the logic auditable: IoUs are `(B, N, M)`, so `argmax(-1)` and gather both naturally return `(B, N)`.',
    ],
    solutionCode: `import torch

def best_iou_match(
    predictions: torch.Tensor,
    ground_truth: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    predictions = torch.as_tensor(predictions, dtype=torch.float64)
    ground_truth = torch.as_tensor(ground_truth, dtype=torch.float64)
    if predictions.ndim != 3 or ground_truth.ndim != 3 or predictions.shape[0] != ground_truth.shape[0] or predictions.shape[2] != 4 or ground_truth.shape[2] != 4:
        raise ValueError("predictions and ground_truth must be (B, N, 4) and (B, M, 4)")
    top_left = torch.maximum(predictions[:, :, None, :2], ground_truth[:, None, :, :2])
    bottom_right = torch.minimum(predictions[:, :, None, 2:], ground_truth[:, None, :, 2:])
    size = torch.clamp(bottom_right - top_left, min=0.0)
    intersection = size[..., 0] * size[..., 1]
    area_pred = (predictions[..., 2] - predictions[..., 0]) * (predictions[..., 3] - predictions[..., 1])
    area_gt = (ground_truth[..., 2] - ground_truth[..., 0]) * (ground_truth[..., 3] - ground_truth[..., 1])
    union = area_pred[:, :, None] + area_gt[:, None, :] - intersection
    ious = torch.where(union > 0, intersection / torch.where(union > 0, union, torch.ones_like(union)), torch.zeros_like(union))
    best_gt = torch.argmax(ious, dim=-1)
    best_iou = torch.gather(ious, dim=-1, index=best_gt[..., None]).squeeze(-1)
    return best_iou, best_gt

predictions = torch.tensor([[[0.0, 0.0, 2.0, 2.0]]])
ground_truth = torch.tensor([[[1.0, 1.0, 3.0, 3.0]]])
print(best_iou_match(predictions, ground_truth))`,
    starterCode: `import torch

def best_iou_match(
    predictions: torch.Tensor,
    ground_truth: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO: broadcast pairwise IoU over B, N, and M, then argmax over M.
    raise NotImplementedError("Implement best_iou_match")

predictions = torch.tensor([[[0.0, 0.0, 2.0, 2.0]]])
ground_truth = torch.tensor([[[1.0, 1.0, 3.0, 3.0]]])
print(best_iou_match(predictions, ground_truth))`,
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'Broadcasting', 'Detection'],
  },
] as const;

// Keep the learner-facing references close to the short, shape-first interview solutions.
// Requirements and walkthroughs carry validation and edge-case discussion; these snippets show
// the tensor path an interviewer is actually testing.
const COMPACT_REFERENCE_SOLUTIONS: Readonly<Record<string, string>> = {
  'stable-softmax-cross-entropy': `import torch

def softmax_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shifted = logits - torch.amax(logits, dim=1, keepdim=True)
    exp_logits = torch.exp(shifted)
    normalizers = torch.sum(exp_logits, dim=1)
    rows = torch.arange(logits.shape[0], dtype=torch.long)
    return torch.mean(torch.log(normalizers) - shifted[rows, labels])`,
  'class-weighted-cross-entropy': `import torch

def class_weighted_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, class_weight: torch.Tensor) -> torch.Tensor:
    shifted = logits - torch.amax(logits, dim=1, keepdim=True)
    log_normalizers = torch.log(torch.sum(torch.exp(shifted), dim=1))
    rows = torch.arange(logits.shape[0], dtype=torch.long)
    losses = log_normalizers - shifted[rows, labels]
    example_weight = class_weight[labels]
    return torch.sum(losses * example_weight) / torch.sum(example_weight)`,
  'non-maximum-suppression': `import torch

def _pairwise_iou(box, boxes):
    top_left = torch.maximum(box[:2], boxes[:, :2])
    bottom_right = torch.minimum(box[2:], boxes[:, 2:])
    size = torch.clamp(bottom_right - top_left, min=0)
    intersection = size[:, 0] * size[:, 1]
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - intersection
    return intersection / torch.clamp(union, min=1e-8)

def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> list[int]:
    order = torch.argsort(scores, descending=True, stable=True).tolist()
    keep = []
    while order:
        current, order = order[0], order[1:]
        keep.append(current)
        if order:
            remaining = torch.as_tensor(order, dtype=torch.long)
            order = [int(i) for i in remaining[_pairwise_iou(boxes[current], boxes[remaining]) <= iou_threshold].tolist()]
    return keep`,
  'causal-attention-mask': `import torch

def make_causal_attention_mask(seq_lens: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    seq_lens = torch.as_tensor(seq_lens, dtype=torch.long)
    length = int(torch.amax(seq_lens).item()) if max_len is None else max(int(torch.amax(seq_lens).item()), max_len)
    positions = torch.arange(length, dtype=torch.long)
    valid = positions[None, :] < seq_lens[:, None]
    causal = positions[:, None] >= positions[None, :]
    return torch.as_tensor(causal[None] & valid[:, :, None] & valid[:, None, :], dtype=torch.int64)`,
  'binary-classification-metrics': `import torch

def binary_classification_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict[str, int | float]:
    tp = int(torch.sum((y_true == 1) & (y_pred == 1)).item())
    tn = int(torch.sum((y_true == 0) & (y_pred == 0)).item())
    fp = int(torch.sum((y_true == 0) & (y_pred == 1)).item())
    fn = int(torch.sum((y_true == 1) & (y_pred == 0)).item())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': (tp + tn) / y_true.numel()}`,
  'pairwise-cosine-similarity': `import torch

def pairwise_cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    numerator = x @ torch.transpose(y, 0, 1)
    x_norm = torch.sqrt(torch.sum(x * x, dim=1))
    y_norm = torch.sqrt(torch.sum(y * y, dim=1))
    denominator = x_norm[:, None] * y_norm[None, :]
    return torch.where(denominator > 0, numerator / torch.clamp(denominator, min=1e-8), torch.zeros_like(numerator))`,
  'top-k-accuracy': `import torch

def top_k_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> torch.Tensor:
    top_k = min(k, logits.shape[1])
    ranked = torch.argsort(logits, dim=1, descending=True)
    candidate_indices = ranked[:, :top_k]
    hits = torch.any(candidate_indices == labels[:, None], dim=1)
    return torch.mean(torch.as_tensor(hits, dtype=torch.float64))`,
  'iou-matrix': `import torch

def box_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    size = torch.clamp(bottom_right - top_left, min=0)
    intersection = size[..., 0] * size[..., 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - intersection
    return intersection / torch.clamp(union, min=1e-8)`,
  'nearest-centroid-classifier': `import torch

def nearest_centroid_predict(train_X: torch.Tensor, train_y: torch.Tensor, test_X: torch.Tensor) -> torch.Tensor:
    labels = torch.unique(train_y, sorted=True)
    centroids = []
    for label in labels:
        class_points = train_X[train_y == label]
        centroids.append(torch.sum(class_points, dim=0) / class_points.shape[0])
    centroids = torch.stack(centroids)
    distances = torch.sum((test_X[:, None, :] - centroids[None, :, :]) ** 2, dim=-1)
    return labels[torch.argmin(distances, dim=1)]`,
  'temperature-scaling-of-logits': `import torch

def temperature_scaled_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    scaled = logits / temperature
    shifted = scaled - torch.amax(scaled, dim=-1, keepdim=True)
    exp_logits = torch.exp(shifted)
    return exp_logits / torch.sum(exp_logits, dim=-1, keepdim=True)`,
  'sinusoidal-positional-encoding': `import math
import torch

def sinusoidal_positional_encoding(length: int, dim: int) -> torch.Tensor:
    positions = torch.arange(length, dtype=torch.float64)[:, None]
    indices = torch.arange(0, dim, 2, dtype=torch.float64)
    frequencies = torch.exp(-math.log(10000.0) * indices / dim)
    angles = positions * frequencies[None, :]
    encoding = torch.zeros(length, dim, dtype=torch.float64)
    encoding[:, 0::2] = torch.sin(angles)
    encoding[:, 1::2] = torch.cos(angles[:, :encoding[:, 1::2].shape[1]])
    return encoding`,
  'unpatchify-back-to-image': `import torch

def unpatchify(patches: torch.Tensor, image_shape: tuple[int, int, int], patch_size: int) -> torch.Tensor:
    channels, height, width = image_shape
    grid_h, grid_w = height // patch_size, width // patch_size
    batch_size = patches.shape[0]
    grid = patches.reshape(batch_size, grid_h, grid_w, channels, patch_size, patch_size)
    grid = grid.permute(0, 3, 1, 4, 2, 5)
    return grid.reshape(batch_size, channels, height, width)`,
  '2d-patchify-for-images': `import torch

def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    batch_size, channels, height, width = images.shape
    grid_h, grid_w = height // patch_size, width // patch_size
    grid = images.reshape(batch_size, channels, grid_h, patch_size, grid_w, patch_size)
    grid = grid.permute(0, 2, 4, 1, 3, 5)
    return grid.reshape(batch_size, grid_h * grid_w, channels * patch_size * patch_size)`,
  'rope-rotary-positional-embedding': `import torch

def apply_rope(x: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len, num_heads, dim = x.shape
    pair = torch.arange(dim // 2, dtype=torch.float64)
    angles = torch.arange(seq_len, dtype=torch.float64)[:, None] * (10000.0 ** (-2 * pair / dim))[None, :]
    sin, cos = torch.sin(angles)[None, :, None, :], torch.cos(angles)[None, :, None, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    output = torch.empty_like(x)
    output[..., 0::2] = even * cos - odd * sin
    output[..., 1::2] = even * sin + odd * cos
    return output`,
  'scaled-dot-product-self-attention': `import torch

def _masked_softmax(scores):
    valid = torch.isfinite(scores)
    scores = torch.where(valid, scores, torch.zeros_like(scores))
    scores = scores - torch.amax(scores, dim=-1, keepdim=True)
    weights = torch.exp(scores) * torch.as_tensor(valid, dtype=scores.dtype)
    return weights / torch.clamp(torch.sum(weights, dim=-1, keepdim=True), min=1e-8)

def self_attention(x, W_q, W_k, W_v, W_o, num_heads, mask=None):
    batch_size, seq_len, model_dim = x.shape
    head_dim = model_dim // num_heads
    split = lambda z: z.reshape(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
    q, k, v = split(x @ W_q), split(x @ W_k), split(x @ W_v)
    scores = q @ k.transpose(-1, -2) / (head_dim ** 0.5)
    if mask is not None:
        scores = torch.where(mask != 0, scores, torch.full_like(scores, float('-inf')))
    context = _masked_softmax(scores) @ v
    context = context.permute(0, 2, 1, 3).reshape(batch_size, seq_len, model_dim)
    return context @ W_o`,
  'cross-attention': `import torch

def _masked_softmax(scores):
    valid = torch.isfinite(scores)
    scores = torch.where(valid, scores, torch.zeros_like(scores))
    scores = scores - torch.amax(scores, dim=-1, keepdim=True)
    weights = torch.exp(scores) * torch.as_tensor(valid, dtype=scores.dtype)
    return weights / torch.clamp(torch.sum(weights, dim=-1, keepdim=True), min=1e-8)

def cross_attention(query_x, context_x, W_q, W_k, W_v, W_o, num_heads, mask=None):
    batch_size, query_len, model_dim = query_x.shape
    context_len = context_x.shape[1]
    head_dim = model_dim // num_heads
    def split(z, length):
        return z.reshape(batch_size, length, num_heads, head_dim).permute(0, 2, 1, 3)
    q = split(query_x @ W_q, query_len)
    k = split(context_x @ W_k, context_len)
    v = split(context_x @ W_v, context_len)
    scores = q @ k.transpose(-1, -2) / (head_dim ** 0.5)
    if mask is not None:
        scores = torch.where(mask != 0, scores, torch.full_like(scores, float('-inf')))
    context = _masked_softmax(scores) @ v
    context = context.permute(0, 2, 1, 3).reshape(batch_size, query_len, model_dim)
    return context @ W_o`,
  'manual-backprop-for-a-2-layer-mlp': `import torch

def mlp_loss_and_grads(X, y, W1, b1, W2, b2):
    hidden_pre = X @ W1 + b1
    hidden = torch.clamp(hidden_pre, min=0)
    logits = hidden @ W2 + b2
    shifted = logits - torch.amax(logits, dim=1, keepdim=True)
    exp_logits = torch.exp(shifted)
    normalizers = torch.sum(exp_logits, dim=1)
    probs = exp_logits / normalizers[:, None]
    rows = torch.arange(X.shape[0], dtype=torch.long)
    loss = torch.mean(torch.log(normalizers) - shifted[rows, y])
    dlogits = probs.clone()
    dlogits[rows, y] -= 1
    dlogits /= X.shape[0]
    dW2 = hidden.T @ dlogits
    db2 = torch.sum(dlogits, dim=0)
    dhidden = dlogits @ W2.T * (hidden_pre > 0)
    dW1 = X.T @ dhidden
    db1 = torch.sum(dhidden, dim=0)
    return {'loss': loss, 'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}`,
  'classic-mlp-forward-backward': `import torch

def mlp_forward_backward(X, y, W1, b1, W2, b2):
    hidden_pre = X @ W1 + b1
    hidden = torch.clamp(hidden_pre, min=0)
    logits = hidden @ W2 + b2
    shifted = logits - torch.amax(logits, dim=1, keepdim=True)
    exp_logits = torch.exp(shifted)
    normalizers = torch.sum(exp_logits, dim=1)
    probs = exp_logits / normalizers[:, None]
    rows = torch.arange(X.shape[0], dtype=torch.long)
    loss = torch.mean(torch.log(normalizers) - shifted[rows, y])
    dlogits = probs.clone()
    dlogits[rows, y] -= 1
    dlogits /= X.shape[0]
    dW2 = hidden.T @ dlogits
    db2 = torch.sum(dlogits, dim=0)
    dhidden = dlogits @ W2.T * (hidden_pre > 0)
    dW1 = X.T @ dhidden
    db1 = torch.sum(dhidden, dim=0)
    return {'loss': loss, 'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}`,
  'simple-n-gram-language-model': `from collections import Counter, defaultdict
import random

def _coerce_tokens(values, name):
    return list(values)

class NGramModel:
    def __init__(self, n):
        self.n = n
        self.counts = defaultdict(Counter)

    def fit(self, tokens):
        tokens = _coerce_tokens(tokens, 'tokens')
        self.counts.clear()
        for index, token in enumerate(tokens):
            for size in range(min(self.n - 1, index) + 1):
                context = tuple(tokens[index - size:index])
                self.counts[context][token] += 1
        return self

    def next_token_probs(self, context):
        context = list(context)
        for size in range(min(self.n - 1, len(context)), -1, -1):
            key = tuple(context[-size:]) if size else ()
            counts = self.counts.get(key)
            if counts:
                total = sum(counts.values())
                return {token: count / total for token, count in counts.items()}
        return {}

    def generate(self, max_tokens, seed=None):
        rng = random.Random(seed)
        output = []
        for _ in range(max_tokens):
            probs = self.next_token_probs(output)
            if not probs:
                break
            tokens, weights = zip(*probs.items())
            output.append(rng.choices(tokens, weights=weights, k=1)[0])
        return output`,
};

const PROGRESSIVE_ORDER: Readonly<Record<string, number>> = {
  'l1-regression-loss': 1,
  'binary-cross-entropy-from-probabilities': 2,
  'masked-mean': 3,
  'binary-classification-metrics': 4,
  'top-k-accuracy': 5,
  'single-box-iou': 6,
  'wrapped-angular-difference': 7,
  'smooth-l1-huber-loss': 8,
  'stable-softmax-cross-entropy': 9,
  'class-weighted-cross-entropy': 10,
  'temperature-scaling-of-logits': 11,
  'pairwise-squared-distance': 12,
  'pairwise-cosine-similarity': 13,
  'nearest-centroid-classifier': 14,
  'iou-matrix': 15,
  'non-maximum-suppression': 16,
  'weighted-box-regression-loss': 17,
  'dice-loss': 18,
  'segmentation-iou-loss': 19,
  'focal-loss': 20,
  'top-k-gather': 21,
  'homogeneous-coordinate-transform': 22,
  '2d-patchify-for-images': 23,
  'unpatchify-back-to-image': 24,
  'sinusoidal-positional-encoding': 25,
  'causal-attention-mask': 26,
  'rope-rotary-positional-embedding': 27,
  'scaled-dot-product-self-attention': 28,
  'incremental-kv-cache': 29,
  'grouped-query-and-multi-query-attention': 30,
  'cross-attention': 31,
  'simple-n-gram-language-model': 32,
  'average-precision-from-matches': 33,
  'greedy-detection-matching': 34,
  'batched-best-iou-match': 35,
  'manual-backprop-for-a-2-layer-mlp': 36,
  'classic-mlp-forward-backward': 37,
};

const PROGRESSIVE_DIFFICULTY: Readonly<Record<string, CodePracticeProblem['difficulty']>> = {
  'l1-regression-loss': 'Easy',
  'binary-cross-entropy-from-probabilities': 'Easy',
  'masked-mean': 'Easy',
  'binary-classification-metrics': 'Easy',
  'top-k-accuracy': 'Easy',
  'single-box-iou': 'Easy',
  'wrapped-angular-difference': 'Easy',
  'smooth-l1-huber-loss': 'Medium',
  'stable-softmax-cross-entropy': 'Medium',
  'class-weighted-cross-entropy': 'Medium',
  'temperature-scaling-of-logits': 'Medium',
  'pairwise-squared-distance': 'Medium',
  'pairwise-cosine-similarity': 'Medium',
  'nearest-centroid-classifier': 'Medium',
  'iou-matrix': 'Medium',
  'non-maximum-suppression': 'Medium',
  'weighted-box-regression-loss': 'Medium',
  'dice-loss': 'Medium',
  'segmentation-iou-loss': 'Medium',
  'focal-loss': 'Medium',
  'top-k-gather': 'Medium',
  'homogeneous-coordinate-transform': 'Medium',
  '2d-patchify-for-images': 'Medium',
  'unpatchify-back-to-image': 'Medium',
  'sinusoidal-positional-encoding': 'Medium',
  'causal-attention-mask': 'Medium',
  'rope-rotary-positional-embedding': 'Medium',
  'scaled-dot-product-self-attention': 'Hard',
  'incremental-kv-cache': 'Hard',
  'grouped-query-and-multi-query-attention': 'Hard',
  'cross-attention': 'Hard',
  'simple-n-gram-language-model': 'Hard',
  'average-precision-from-matches': 'Hard',
  'greedy-detection-matching': 'Hard',
  'batched-best-iou-match': 'Hard',
  'manual-backprop-for-a-2-layer-mlp': 'Hard',
  'classic-mlp-forward-backward': 'Hard',
};

const ALL_CODE_PRACTICE_PROBLEMS: readonly CodePracticeProblem[] = [
  ...RAW_CODE_PRACTICE_PROBLEMS,
  ...ATTENTION_CODE_PRACTICE_PROBLEMS,
  ...ARCHITECTURE_CODE_PRACTICE_PROBLEMS,
];

export const codePracticeProblems: readonly CodePracticeProblem[] = ALL_CODE_PRACTICE_PROBLEMS
  .map((problem) => ({
    ...problem,
    ...ATTENTION_PROBLEM_ENRICHMENTS[problem.id],
    track: problem.track ?? 'fundamentals',
    environment: problem.environment ?? 'browser',
    editorStart:
      problem.editorStart ?? (problem.signature.trimStart().startsWith('def ') ? 'blank' : 'scaffold'),
    order: PROGRESSIVE_ORDER[problem.id] ?? problem.order,
    difficulty: PROGRESSIVE_DIFFICULTY[problem.id] ?? problem.difficulty,
    walkthroughCode: problem.walkthroughCode ?? problem.solutionCode,
    solutionCode: COMPACT_REFERENCE_SOLUTIONS[problem.id] ?? problem.solutionCode,
  }))
  .sort((left, right) => left.order - right.order);
