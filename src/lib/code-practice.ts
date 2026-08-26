import { ARCHITECTURE_CODE_PRACTICE_PROBLEMS } from './code-practice-architectures';
import {
  ATTENTION_CODE_PRACTICE_PROBLEMS,
  ATTENTION_PROBLEM_ENRICHMENTS,
} from './code-practice-attention';
import codePracticeVisualSpecs from './code-practice-visuals.json';

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

export interface CodePracticeNumpyAlternative {
  /** Compact, standalone NumPy reference for interview recall. */
  code: string;
  /** Small runnable example appended when the reference is shown. */
  exampleCode: string;
  /** One or two syntax or shape cues worth memorizing. */
  memory: readonly string[];
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
  numpyAlternative?: CodePracticeNumpyAlternative;
  starterCode: string;
  track?: 'fundamentals' | 'architecture';
  editorStart?: 'blank' | 'scaffold';
  interview?: CodePracticeInterviewFormat;
  reasoning?: readonly CodePracticeReasoningPoint[];
  packages?: readonly string[];
  tags?: readonly string[];
}

export const CODE_PRACTICE_SECTION_SUMMARY =
  'Practice the way you would code in an ML interview: clarify the contract, design a small API, implement it cleanly, test shapes, and defend the tradeoffs.';

const PYTORCH_AND_NUMPY_PACKAGES = ['torch', 'numpy'] as const;

const CODE_PRACTICE_VISUALS: Readonly<Record<string, CodePracticeVisual>> = Object.fromEntries(
  codePracticeVisualSpecs.map((spec) => [
    spec.id,
    {
      src: `/assets/images/code-glance-${spec.id}.svg`,
      alt: `${spec.headline}.`,
      caption: spec.caption,
    },
  ]),
);

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
      'Shift every row by its largest logit before exponentiating:\n`shifted = logits - row_max`\nSoftmax is unchanged because the same constant is removed from every class.',
      'Then compute log-sum-exp and subtract the target logit:\n`loss_i = log(Σ_c exp(shifted_i,c)) - shifted_i,target`\nAverage the per-row losses at the end.',
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
      'First compute the stable cross-entropy for each row:\n`loss_i = log(Σ_c exp(z_i,c - m_i)) - (z_i,y_i - m_i)`\nHere `m_i` is the row maximum, so subtracting it prevents overflow without changing softmax.',
      'Indexing class weights `(C,)` with labels `(N,)` gives one weight per example `(N,)`. Use PyTorch’s weighted-mean denominator:\n`L = Σ_i w[y_i] loss_i / Σ_i w[y_i]`',
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
      'NMS is a greedy loop:\n`sort by score → keep best box → suppress high-IoU boxes → repeat`',
      'Compute IoU between the kept box and all remaining boxes at once. Keep candidates whose IoU is at most the threshold; index-based tie-breaking makes equal scores deterministic.',
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
      'Combine two boolean rules:\n`causal: key_index <= query_index`\n`valid: query_index < length and key_index < length`',
      'Build the lower-triangular causal template once, then broadcast it against each example’s validity mask to produce `(B, T, T)` without Python loops.',
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
      'Build the confusion matrix first:\n`TP: prediction = 1 and target = 1`\n`FP: prediction = 1 and target = 0`\n`FN: prediction = 0 and target = 1`\n`TN: prediction = 0 and target = 0`',
      'Then compute the ratios:\n`precision = TP / (TP + FP)`\n`recall = TP / (TP + FN)`\n`F1 = 2 · precision · recall / (precision + recall)`\nReturn `0.0` when a denominator is zero.',
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
      'Cosine similarity is a normalized dot product:\n`similarity[i, j] = (x_i · y_j) / (||x_i|| ||y_j||)`',
      'The norm views broadcast into every pair:\n`x_norms[:, None]: (N, 1)`\n`y_norms[None, :]: (1, M)`\nTheir product matches the dot-product matrix `(N, M)`.',
      'A zero vector makes the denominator zero. Return `0.0` for those pairs instead of dividing by zero.',
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
      'For each row, sort class scores from largest to smallest and keep the first `k` indices:\n`correct_i = target_i appears in top_k_indices_i`',
      'The final accuracy is the mean of that boolean vector:\n`top_k_accuracy = mean(correct)`',
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
      'Insert singleton axes so every box in the first set meets every box in the second:\n`boxes1[:, None, :2]: (N, 1, 2)`\n`boxes2[None, :, :2]: (1, M, 2)`\n`pairwise top-left corners: (N, M, 2)`',
      'Reducing only the coordinate axis keeps one value per pair:\n`intersection: (N, M)`\n`area1[:, None]: (N, 1)`\n`area2[None, :]: (1, M)`\n`union and IoU: (N, M)`',
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
      'Implement a `NearestCentroidClassifier` with separate `fit(train_X, train_y)` and `predict(test_X)` methods.',
      'Fitting stores one mean vector per class. Prediction compares new rows with those learned centroids and chooses the nearest class; ties go to the smaller label.',
    ],
    signature: `@dataclass(slots=True)
class NearestCentroidClassifier:
    labels: torch.Tensor | None = None
    centroids: torch.Tensor | None = None

    def fit(self, train_X: torch.Tensor, train_y: torch.Tensor) -> NearestCentroidClassifier:
        ...

    def predict(self, test_X: torch.Tensor) -> torch.Tensor:
        ...`,
    requirements: [
      '`train_X` is an `(N, D)` array or list.',
      '`train_y` is a 1D array or list of length `N` containing class labels.',
      '`test_X` is an `(M, D)` array or list.',
      '`fit` stores sorted labels and a `(K, D)` centroid tensor, then returns `self`.',
      '`predict` returns a 1D label tensor and rejects use before fitting.',
      'If distances tie, choose the smaller class label.',
      'Raise `ValueError` for invalid shapes or invalid labels.',
    ],
    examples: [
      {
        label: 'Example 1',
        lines: [
          'model = NearestCentroidClassifier()',
          'train_X = [[0.0], [2.0], [10.0], [12.0]]',
          'train_y = [0, 0, 1, 1]',
          'model.fit(train_X, train_y).predict([[0.0], [6.0], [12.0]])',
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
      'A class is useful here because `fit` creates state that later `predict` calls reuse.',
      'Group `train_X` by label, then divide each class-wise feature sum by its count to form the centroids.',
      'Sort the unique labels so that ties fall to the smaller class label when you take an argmin.',
      'Broadcast `test_X` against the centroid matrix to compute all distances at once.',
      'Use squared Euclidean distance to avoid an unnecessary square root.',
    ],
    solutionNotes: [
      'This is stateful, unlike a one-shot loss: `fit` learns labels and centroids, while `predict` reuses them. A small dataclass makes that lifecycle explicit.',
      'Represent each class by the mean of its training points:\n`class points: (n_k, D)  →  centroid: (D,)`\nStacking all class means gives `centroids: (K, D)`.',
      'Broadcast every test point against every centroid:\n`test_X[:, None, :]: (M, 1, D)`\n`centroids[None, :, :]: (1, K, D)`\n`difference: (M, K, D)  →  squared distance: (M, K)`',
      'Squared distance has the same nearest-centroid ordering as Euclidean distance, so no square root is needed. Sorted class labels make ties deterministic.',
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

from dataclasses import dataclass
import torch

@dataclass(slots=True)
class NearestCentroidClassifier:
    labels: torch.Tensor | None = None
    centroids: torch.Tensor | None = None

    def fit(self, train_X: torch.Tensor, train_y: torch.Tensor) -> NearestCentroidClassifier:
        # TODO: compute and store one centroid per sorted class label.
        raise NotImplementedError("Implement fit")

    def predict(self, test_X: torch.Tensor) -> torch.Tensor:
        # TODO: reject an unfitted model, then return the nearest stored label.
        raise NotImplementedError("Implement predict")

def smoke_test() -> None:
    model = NearestCentroidClassifier().fit([[0.0], [2.0], [10.0], [12.0]], [0, 0, 1, 1])
    prediction = model.predict([[0.0], [6.0], [12.0]])
    assert prediction.tolist() == [0, 0, 1]
    print("Nearest-centroid smoke test passed:", prediction.tolist())

smoke_test()`,
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
      'Divide the logits by temperature before softmax:\n`p_i(T) = exp(z_i / T) / Σ_j exp(z_j / T)`\n`T = 1` changes nothing, `T < 1` sharpens the distribution, and `T > 1` flattens it.',
      'Subtract each row maximum before `exp`. Summing with `keepdim=True` leaves normalizers shaped `(N, 1)`, which broadcast across logits shaped `(N, C)`.',
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
      'Each adjacent channel pair shares a frequency but uses sine and cosine phases:\n`PE(pos, 2k) = sin(pos / 10000^(2k / dim))`\n`PE(pos, 2k + 1) = cos(pos / 10000^(2k / dim))`',
      'The broadcast is:\n`positions: (length, 1)`\n`frequencies: (1, ceil(dim / 2))`\n`angles: (length, ceil(dim / 2))`\nWrite sine into `0::2` and cosine into `1::2` to return `(length, dim)`.',
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
      'Undo patch extraction by making the grid axes explicit again:\n`(B, N, C·P·P)  →  (B, grid_h, grid_w, C, P, P)`',
      'Move each grid axis next to its local pixel axis, then collapse them:\n`(B, grid_h, grid_w, C, P, P)`\n`→ (B, C, grid_h, P, grid_w, P)`\n`→ (B, C, H, W)`',
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
      'Expose the patch grid without changing any values:\n`(B, C, H, W)  →  (B, C, grid_h, P, grid_w, P)`',
      'Move the two grid axes before the patch contents, then flatten:\n`(B, C, grid_h, P, grid_w, P)`\n`→ (B, grid_h, grid_w, C, P, P)`\n`→ (B, N, C·P·P)`\nThis preserves row-major patch order.',
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
      'Treat each adjacent channel pair as a 2D vector and rotate it by a position-dependent angle. Rotation preserves the pair’s norm while adding relative position information.',
      'Build one angle for every position and channel pair:\n`positions: (T, 1)`\n`inverse frequencies: (1, D / 2)`\n`angles: (T, D / 2)`',
      'Split the adjacent pairs with ordinary Python slicing:\n`x[..., 0::2]  →  even channels 0, 2, 4, ...`\n`x[..., 1::2]  →  odd channels 1, 3, 5, ...`',
      'Reshape sine and cosine to `(1, T, 1, D / 2)` so the same table broadcasts over batch and head axes.',
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
      'Implement `MultiHeadSelfAttention` as an `nn.Module` for inputs shaped `(B, T, D_model)`.',
      'Store the four learned projections on the module, split Q/K/V into heads, apply stable masked attention, merge the heads, and project back to `(B, T, D_model)`.',
    ],
    signature: `@dataclass(frozen=True, slots=True)
class MHAConfig:
    model_dim: int
    num_heads: int

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: MHAConfig):
        ...

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ...`,
    requirements: [
      '`x` has shape `(B, T, D_model)`.',
      '`MHAConfig` validates positive dimensions and requires `num_heads` to divide `model_dim`.',
      'The module owns learned Q, K, V, and output projections.',
      '`mask`, if provided, is broadcastable to `(B, H, T, T)` and contains `1` for allowed positions and `0` for blocked positions.',
      'Return an output of shape `(B, T, D_model)`.',
      'Include a runnable smoke test for shape, masking, and finite output.',
    ],
    examples: [
      {
        label: 'Example 1',
        lines: [
          'layer = MultiHeadSelfAttention(MHAConfig(model_dim=8, num_heads=2))',
          'x.shape = (2, 4, 8)',
        ],
        result: 'layer(x).shape == (2, 4, 8)',
      },
      {
        label: 'Example 2',
        lines: [
          'positions = torch.arange(4)',
          'causal_mask = positions[:, None] >= positions[None, :]',
          'output = layer(x, causal_mask)',
        ],
        result: 'output.shape == x.shape and every output value is finite',
      },
    ],
    hint: [
      'Put learned projections in `__init__`; tensor shape work belongs in `forward`.',
      'Reshape the projected tensors into `(B, H, T, D_head)` before computing attention scores.',
      'Use the scaled dot-product formula `Q K^T / sqrt(D_head)` and a numerically stable softmax over the last axis.',
      'If a mask is provided, broadcast it to the score tensor and zero out blocked positions before softmax.',
      'After attention, transpose the heads back and concatenate them before the final output projection.',
    ],
    solutionNotes: [
      'Scaled dot-product attention is:\n`Attention(Q, K, V) = softmax(QKᵀ / √D_head) V`',
      'Use an `nn.Module` here because Q, K, V, and output projections are learned state reused on every call. The frozen config keeps the head divisibility invariant next to construction.',
      'For self-attention, `Q`, `K`, and `V` all come from `x`:\n`x: (B, T, D_model)`\n`Q, K, V: (B, H, T, D_head)`\n`scores: (B, H, T, T)`\n`output: (B, T, D_model)`',
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

from dataclasses import dataclass
import torch
from torch import nn

@dataclass(frozen=True, slots=True)
class MHAConfig:
    model_dim: int
    num_heads: int

    def __post_init__(self) -> None:
        if self.model_dim <= 0 or self.num_heads <= 0 or self.model_dim % self.num_heads:
            raise ValueError("num_heads must divide a positive model_dim")

    @property
    def head_dim(self) -> int:
        return self.model_dim // self.num_heads

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: MHAConfig):
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.k_proj = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.v_proj = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.out_proj = nn.Linear(config.model_dim, config.model_dim, bias=False)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        # TODO: reshape (B, T, D_model) into (B, H, T, D_head).
        raise NotImplementedError("Implement _split_heads")

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # TODO: project, split heads, attend stably, merge heads, and project out.
        raise NotImplementedError("Implement forward")

def smoke_test() -> None:
    torch.manual_seed(0)
    layer = MultiHeadSelfAttention(MHAConfig(model_dim=8, num_heads=2))
    x = torch.randn(2, 4, 8)
    positions = torch.arange(4)
    output = layer(x, positions[:, None] >= positions[None, :])
    assert output.shape == x.shape and bool(torch.all(torch.isfinite(output)))
    print("MHA smoke test passed:", tuple(output.shape))

smoke_test()`,
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
      'Implement `CrossAttention` as an `nn.Module` for a query sequence and a separate context sequence.',
      'Store the learned projections on the module, build Q from the query and K/V from the context, then return `(B, Tq, D_model)`.',
    ],
    signature: `@dataclass(frozen=True, slots=True)
class CrossAttentionConfig:
    model_dim: int
    num_heads: int

class CrossAttention(nn.Module):
    def forward(
        self,
        query_x: torch.Tensor,
        context_x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ...`,
    requirements: [
      '`query_x` has shape `(B, Tq, D_model)`.',
      '`context_x` has shape `(B, Tk, D_model)`.',
      'The config requires `num_heads` to divide `model_dim`.',
      'The module owns Q, K, V, and output projections.',
      '`mask`, if provided, is broadcastable to `(B, H, Tq, Tk)` and contains `1` for allowed positions and `0` for blocked positions.',
      'Return an output of shape `(B, Tq, D_model)`.',
      'Include a runnable smoke test with different query and context lengths.',
    ],
    examples: [
      {
        label: 'Example 1',
        lines: [
          'layer = CrossAttention(CrossAttentionConfig(model_dim=8, num_heads=2))',
          'query_x.shape = (2, 3, 8)',
          'context_x.shape = (2, 5, 8)',
        ],
        result: 'layer(query_x, context_x).shape == (2, 3, 8)',
      },
      {
        label: 'Example 2',
        lines: [
          'mask.shape = (3, 5)',
          'output = layer(query_x, context_x, mask)',
        ],
        result: 'output.shape == query_x.shape and every value is finite',
      },
    ],
    hint: [
      'Put learned projections in `__init__`; keep query/context tensor flow in `forward`.',
      'The only difference from self-attention is that queries come from `query_x`, while keys and values come from `context_x`.',
      'Reshape the projected tensors into `(B, H, Tq, D_head)` for queries and `(B, H, Tk, D_head)` for keys and values.',
      'Use the scaled dot-product formula `Q K^T / sqrt(D_head)` and a numerically stable softmax over the last axis.',
      'If a mask is provided, broadcast it to the score tensor and zero out blocked positions before softmax.',
    ],
    solutionNotes: [
      'Cross-attention uses the same equation as self-attention:\n`Attention(Q, K, V) = softmax(QKᵀ / √D_head) V`\nThe difference is where the three inputs come from: `Q` uses `query_x`; `K` and `V` use `context_x`.',
      'Use a module because the projections are learned parameters reused across calls. The config owns the model/head divisibility invariant.',
      'The two sequence lengths stay separate:\n`Q: (B, H, Tq, D_head)`\n`K, V: (B, H, Tk, D_head)`\n`scores: (B, H, Tq, Tk)`\n`output: (B, Tq, D)`',
      'Each query token can read all `Tk` context tokens. Apply any mask to the score matrix before the stable softmax.',
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

from dataclasses import dataclass
import torch
from torch import nn

@dataclass(frozen=True, slots=True)
class CrossAttentionConfig:
    model_dim: int
    num_heads: int

    def __post_init__(self) -> None:
        if self.model_dim <= 0 or self.num_heads <= 0 or self.model_dim % self.num_heads:
            raise ValueError("num_heads must divide a positive model_dim")

    @property
    def head_dim(self) -> int:
        return self.model_dim // self.num_heads

class CrossAttention(nn.Module):
    def __init__(self, config: CrossAttentionConfig):
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.k_proj = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.v_proj = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.out_proj = nn.Linear(config.model_dim, config.model_dim, bias=False)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        # TODO: reshape (B, T, D_model) into (B, H, T, D_head).
        raise NotImplementedError("Implement _split_heads")

    def forward(self, query_x: torch.Tensor, context_x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # TODO: project two sources, split heads, attend stably, merge, and project out.
        raise NotImplementedError("Implement forward")

def smoke_test() -> None:
    torch.manual_seed(0)
    layer = CrossAttention(CrossAttentionConfig(model_dim=8, num_heads=2))
    query, context = torch.randn(2, 3, 8), torch.randn(2, 5, 8)
    output = layer(query, context, torch.ones(3, 5))
    assert output.shape == query.shape and bool(torch.all(torch.isfinite(output)))
    print("Cross-attention smoke test passed:", tuple(output.shape))

smoke_test()`,
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
      'Cache the forward path because every backward step needs one of these values:\n`X → z1 → ReLU → hidden → z2 → softmax → loss`',
      'Start backprop from the softmax-cross-entropy shortcut:\n`dlogits = (probs - one_hot(y)) / N`\nThen move backward through the second affine layer, the ReLU mask, and the first affine layer.',
      'The output-layer gradients follow directly from the affine rule:\n`dW2 = hidden.T @ dlogits`\n`db2 = sum(dlogits, axis=0)`\n`dhidden = dlogits @ W2.T`',
      'ReLU passes gradient only where its pre-activation was positive:\n`dz1 = dhidden * (z1 > 0)`\nThen finish with `dW1 = X.T @ dz1` and `db1 = sum(dz1, axis=0)`.',
      'The most common interview mistakes are forgetting the batch division in `dlogits`, using the post-ReLU tensor for the ReLU mask, or transposing the wrong operand. Check each gradient shape against its parameter shape.',
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
      'Cache the forward path because every backward step needs one of these values:\n`X → z1 → ReLU → hidden → z2 → softmax → loss`',
      'Start backprop from the softmax-cross-entropy shortcut:\n`dlogits = (probs - one_hot(y)) / N`\nThen move backward through the second affine layer, the ReLU mask, and the first affine layer.',
      'Keep a shape ledger beside the derivation:\n`X: (N, Din), W1: (Din, H), hidden: (N, H)`\n`W2: (H, C), logits: (N, C)`\nEvery returned gradient must have the same shape as its corresponding parameter.',
      'For an affine layer `Y = XW + b`, memorize the three backward rules:\n`dW = X.T @ dY`\n`db = sum(dY, axis=0)`\n`dX = dY @ W.T`',
      'Run the chain in reverse order and do not update parameters inside this function. The interview target is the derivation: stable forward loss, cached intermediates, and explicit gradients that can be compared with autograd later.',
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
      'Build a runnable backoff n-gram model that learns token counts and samples deterministic continuations.',
    prompt: [
      'Implement a simple n-gram language model class with `__init__`, `fit`, `next_token_probs`, and `generate` methods.',
      'Train on a list of tokens, return next-token probability distributions from observed counts, sample autoregressively, and back off gracefully when a context has not been seen before.',
      'Keep corpus loading outside the model. The supplied smoke test uses a tiny sequence so every expected probability is easy to verify in an interview.',
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
      'The same `fit(tokens)` method should work for any iterable of string or integer tokens.',
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
    ],
    hint: [
      'A dictionary keyed by context tuples works for both string and integer tokens.',
      'During training, update counts for every suffix length from `0` up to `n - 1`, not just the longest context.',
      'To back off gracefully, keep shortening the context suffix until you find a context with observed counts.',
      'Use a dedicated seeded RNG inside `generate` so sampling is repeatable without touching global random state.',
      'Keep corpus loading and tokenization outside the class so the model stays reusable and the interview solution runs offline.',
    ],
    solutionNotes: [
      'Map each context tuple to counts of the tokens observed after it:\n`context tuple → next-token counts → normalized probabilities`\nDuring fitting, update every suffix length up to order `n`.',
      'At inference time, keep at most `n - 1` context tokens and back off through shorter suffixes until one was seen during fitting. Normalize that context’s counts into probabilities.',
      'Generation repeats lookup and sampling. A local `random.Random(seed)` keeps the sequence deterministic without changing global random state.',
      'The class owns fitted counts because `next_token_probs` and `generate` reuse them. Corpus loading is deliberately outside this interview implementation; the smoke test uses a tiny token list so the whole solution runs offline and the probability checks are exact.',
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

def smoke_test() -> None:
    model = NGramModel(2).fit(["a", "b", "a", "c"])
    assert model.next_token_probs(["a"]) == {"b": 0.5, "c": 0.5}
    assert model.next_token_probs(["unseen"]) == {"a": 0.5, "b": 0.25, "c": 0.25}
    generated = model.generate(5, seed=4)
    assert len(generated) == 5
    print("n-gram smoke test passed:", generated)

smoke_test()`,
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
      'Take the absolute error at every position, then average all entries:\n`L = (1 / K) Σ_i |prediction_i - target_i|`\nHere `K` is the total number of elements, not just the batch size.',
      'The penalty grows linearly, so a large residual matters without dominating as strongly as it would under squared error.',
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
      'Let the elementwise error be:\n`e = prediction - target`\nThen use the piecewise penalty:\n`loss(e) = 0.5 e²                         if |e| <= delta`\n`loss(e) = delta (|e| - 0.5 delta)       otherwise`',
      'Small errors get L2’s smooth gradient; large errors switch to L1-like growth. Compute both branches, select with `torch.where`, then average all entries.',
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
      '`probability` already contains probabilities, so do not apply sigmoid inside this function. Validate the range, clamp only to protect the logarithms at 0 and 1, then return the mean loss.',
    ],
    signature: `def binary_cross_entropy(
    probability: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    ...`,
    requirements: [
      '`probability` and `target` have the same shape.',
      'Probabilities lie in `[0, 1]`.',
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
      'An invalid binary label satisfies `(target != 0) & (target != 1)`; reject when `torch.any` finds one.',
      'Clamp once before both logarithms.',
    ],
    solutionNotes: [
      'The target chooses which mistake to penalize:\n`y_i = 1  →  loss_i = -log(p_i)`\n`y_i = 0  →  loss_i = -log(1 - p_i)`',
      'Average those elementwise losses:\n`L = -(1 / K) Σ_i [y_i log(p_i) + (1 - y_i) log(1 - p_i)]`',
      'A target is invalid only when it is neither `0` nor `1`:\n`invalid = (target != 0) & (target != 1)`\nReject the input if any entry in this mask is true.',
      'This function receives probabilities, so do not apply sigmoid. Clamp `p` only to avoid `log(0)` at the endpoints.',
      'If the input is a raw logit `z`, use the stable logits form instead:\n`loss_i = max(z_i, 0) - z_i y_i + log(1 + exp(-|z_i|))`',
      'In PyTorch training, call `binary_cross_entropy_with_logits` for raw logits.',
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
    if bool(torch.any((probability < 0) | (probability > 1))):
        raise ValueError("probability must lie in [0, 1]")
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
    # TODO: validate probabilities and binary targets, clamp, apply BCE, and mean-reduce.
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
      'Intersect the boxes by taking the later top-left corner and earlier bottom-right corner:\n`intersection_wh = clamp(min(br1, br2) - max(tl1, tl2), min=0)`',
      'Then divide intersection by union:\n`IoU = intersection / (area1 + area2 - intersection)`\nReturn zero when the union is zero, so degenerate boxes stay defined.',
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
      'Expand squared distance so one matrix multiplication handles every pair:\n`D[i, j] = ||x_i||² + ||y_j||² - 2 x_i · y_j`',
      'The three terms have shapes:\n`x_squared: (N, 1)`\n`y_squared: (1, M)`\n`x @ y.T: (N, M)`\nBroadcasting produces the final distance matrix `(N, M)`.',
      'This avoids materializing direct differences shaped `(N, M, D)`. Clamp tiny negative results to zero because floating-point roundoff can make a theoretical squared distance slightly negative.',
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
      'Treat the mask as a selector. Expand it over features, multiply, then reduce only the item axis:\n`values: (B, N, D)`\n`mask[..., None]: (B, N, 1)`\n`masked sum: (B, D)`',
      'Keep the valid count shaped `(B, 1)` so division broadcasts across `D`. Clamp only the count; an all-false mask then returns a zero row.',
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
      'The scores produce row indices, but `gather` needs an index for every feature channel:\n`top_indices: (B, k)`\n`top_indices[..., None]: (B, k, 1)`\n`expanded indices: (B, k, D)`',
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
      'Compute one soft overlap score per batch item:\n`Dice = (2 · Σ(prediction · target) + eps) / (Σ prediction + Σ target + eps)`\nThen return `1 - mean(Dice)`.',
      'The factor `2` rewards shared foreground mass. Flatten each example so the same reduction works for any spatial rank; `eps` keeps empty masks finite.',
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
      'Compute one soft IoU per batch item:\n`intersection = Σ(prediction · target)`\n`union = Σ prediction + Σ target - intersection`\n`IoU = (intersection + eps) / (union + eps)`',
      'Return `mean(1 - IoU)`. Probabilities replace hard set membership, so the score remains differentiable; `eps` only prevents an undefined empty union.',
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
      'Convert each logit into the probability assigned to its true label:\n`p_t = p          when target = 1`\n`p_t = 1 - p      when target = 0`',
      'Down-weight easy examples with:\n`L = -(1 / K) Σ_i (1 - p_t,i)^gamma log(p_t,i)`\nWhen `p_t` is near one, the modulating factor is near zero. In production, prefer a fused logits-based implementation.',
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
      'Apply one weight per box coordinate:\n`weighted_error = coordinate_weight * |prediction - target|`',
      'A weight vector shaped `(D,)` broadcasts over every leading box dimension. Average only after weighting so the coordinate tradeoff stays explicit.',
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
      'Angles live on a circle, so ordinary subtraction mistakes the seam at `±π` for a long rotation.',
      'Wrap the raw difference back to the principal interval:\n`delta = atan2(sin(prediction - target), cos(prediction - target))`\nThe result lies in `[-π, π]`.',
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
      'Sort detections by descending confidence, then compute precision at every rank:\n`precision(k) = TP(k) / k`',
      'Only true-positive ranks contribute to AP:\n`AP = (1 / num_ground_truth) Σ_k m_k · precision(k)`\nA false positive adds no numerator but still increases later rank denominators.',
      'The ground-truth count normalizes the sum and makes the metric comparable across images or classes. Full mAP additionally averages this AP across classes and IoU thresholds.',
      'This exercise uses the non-interpolated precision-at-true-positive form. Real benchmarks may interpolate the precision curve or average multiple IoU thresholds, so state the convention before comparing AP numbers.',
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
      'Match in descending confidence order:\n`prediction → best unmatched ground truth → threshold check → claim or false positive`',
      'Once a ground-truth object is claimed, later duplicates are false positives. Vectorized IoU handles the geometry; the small loop enforces the one-to-one rule.',
      'The output must return to original prediction order even though matching happens in score order. Write each decision into `matches[prediction_index]` instead of appending decisions to a new list.',
      'Mask already-used ground truths before `argmax`. Choosing the best ground truth first and checking availability afterward can miss the next-best legal match.',
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
      'Homogeneous coordinates fold translation into the same multiplication as rotation:\n`p′ = R p + t`\n`[p′; 1] = [[R, t], [0, 1]] [p; 1]`',
      'The shape flow is:\n`points: (N, 3)  →  pad ones: (N, 4)`\n`transform (4, 4) @ points.T (4, N)  →  (4, N)`\n`transpose and slice  →  (N, 3)`',
      'Pad with `1`, not `0`: that activates the transform’s translation column for every point.',
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
      'The shape flow is:\n`pairwise IoU: (B, N, M)`\n`argmax over M: (B, N)`\n`gather best IoU: (B, N)`',
      'The singleton axes keep batches isolated:\n`predictions[:, :, None, :]: (B, N, 1, 4)`\n`ground_truth[:, None, :, :]: (B, 1, M, 4)`\nBroadcasting forms `(B, N, M, 4)` without comparing boxes across images.',
      '`argmax` returns the winning ground-truth index, while `gather` retrieves the IoU at that index. This is independent best matching; unlike greedy detection matching, it does not prevent several predictions from choosing the same ground truth.',
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
  'nearest-centroid-classifier': `from dataclasses import dataclass
import torch

@dataclass(slots=True)
class NearestCentroidClassifier:
    labels: torch.Tensor | None = None
    centroids: torch.Tensor | None = None

    def fit(self, train_X, train_y):
        train_X = torch.as_tensor(train_X, dtype=torch.float64)
        train_y = torch.as_tensor(train_y, dtype=torch.long)
        self.labels = torch.unique(train_y, sorted=True)
        rows = []
        for label in self.labels:
            class_points = train_X[train_y == label]
            rows.append(torch.sum(class_points, dim=0) / class_points.shape[0])
        self.centroids = torch.stack(rows)
        return self

    def predict(self, test_X):
        if self.labels is None or self.centroids is None:
            raise ValueError("fit must be called before predict")
        test_X = torch.as_tensor(test_X, dtype=self.centroids.dtype)
        distances = torch.sum((test_X[:, None] - self.centroids[None]) ** 2, dim=-1)
        return self.labels[torch.argmin(distances, dim=1)]

def smoke_test():
    model = NearestCentroidClassifier().fit([[0.0], [2.0], [10.0], [12.0]], [0, 0, 1, 1])
    prediction = model.predict([[0.0], [6.0], [12.0]])
    assert prediction.tolist() == [0, 0, 1]
    print("Nearest-centroid smoke test passed:", prediction.tolist())

smoke_test()`,
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
  'scaled-dot-product-self-attention': `from dataclasses import dataclass
import torch
from torch import nn

@dataclass(frozen=True, slots=True)
class MHAConfig:
    model_dim: int
    num_heads: int

    def __post_init__(self):
        if self.model_dim <= 0 or self.num_heads <= 0 or self.model_dim % self.num_heads:
            raise ValueError("num_heads must divide a positive model_dim")

    @property
    def head_dim(self):
        return self.model_dim // self.num_heads

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: MHAConfig):
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.k_proj = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.v_proj = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.out_proj = nn.Linear(config.model_dim, config.model_dim, bias=False)

    def _split_heads(self, tensor):
        batch, length, _ = tensor.shape
        tensor = tensor.reshape(batch, length, self.config.num_heads, self.config.head_dim)
        return tensor.permute(0, 2, 1, 3)

    def forward(self, x, mask=None):
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        scores = q @ k.transpose(-1, -2) / (self.config.head_dim ** 0.5)
        if mask is not None:
            mask = torch.broadcast_to(torch.as_tensor(mask), scores.shape)
            scores = torch.where(mask != 0, scores, torch.full_like(scores, float('-inf')))
        valid = torch.isfinite(scores)
        safe = torch.where(valid, scores, torch.zeros_like(scores))
        shifted = safe - torch.amax(safe, dim=-1, keepdim=True)
        weights = torch.exp(shifted) * torch.as_tensor(valid, dtype=scores.dtype)
        weights = weights / torch.clamp(torch.sum(weights, dim=-1, keepdim=True), min=1e-8)
        context = (weights @ v).permute(0, 2, 1, 3)
        context = context.reshape(x.shape[0], x.shape[1], self.config.model_dim)
        return self.out_proj(context)

def smoke_test():
    torch.manual_seed(0)
    layer = MultiHeadSelfAttention(MHAConfig(model_dim=8, num_heads=2))
    x = torch.randn(2, 4, 8)
    positions = torch.arange(4)
    mask = positions[:, None] >= positions[None, :]
    output = layer(x, mask)
    assert output.shape == x.shape and bool(torch.all(torch.isfinite(output)))
    print("MHA smoke test passed:", tuple(output.shape))

smoke_test()`,
  'cross-attention': `from dataclasses import dataclass
import torch
from torch import nn

@dataclass(frozen=True, slots=True)
class CrossAttentionConfig:
    model_dim: int
    num_heads: int

    def __post_init__(self):
        if self.model_dim <= 0 or self.num_heads <= 0 or self.model_dim % self.num_heads:
            raise ValueError("num_heads must divide a positive model_dim")

    @property
    def head_dim(self):
        return self.model_dim // self.num_heads

class CrossAttention(nn.Module):
    def __init__(self, config: CrossAttentionConfig):
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.k_proj = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.v_proj = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.out_proj = nn.Linear(config.model_dim, config.model_dim, bias=False)

    def _split_heads(self, tensor):
        batch, length, _ = tensor.shape
        tensor = tensor.reshape(batch, length, self.config.num_heads, self.config.head_dim)
        return tensor.permute(0, 2, 1, 3)

    def forward(self, query_x, context_x, mask=None):
        q = self._split_heads(self.q_proj(query_x))
        k = self._split_heads(self.k_proj(context_x))
        v = self._split_heads(self.v_proj(context_x))
        scores = q @ k.transpose(-1, -2) / (self.config.head_dim ** 0.5)
        if mask is not None:
            mask = torch.broadcast_to(torch.as_tensor(mask), scores.shape)
            scores = torch.where(mask != 0, scores, torch.full_like(scores, float('-inf')))
        valid = torch.isfinite(scores)
        safe = torch.where(valid, scores, torch.zeros_like(scores))
        shifted = safe - torch.amax(safe, dim=-1, keepdim=True)
        weights = torch.exp(shifted) * torch.as_tensor(valid, dtype=scores.dtype)
        weights = weights / torch.clamp(torch.sum(weights, dim=-1, keepdim=True), min=1e-8)
        context = (weights @ v).permute(0, 2, 1, 3)
        context = context.reshape(query_x.shape[0], query_x.shape[1], self.config.model_dim)
        return self.out_proj(context)

def smoke_test():
    torch.manual_seed(0)
    layer = CrossAttention(CrossAttentionConfig(model_dim=8, num_heads=2))
    query, context = torch.randn(2, 3, 8), torch.randn(2, 5, 8)
    output = layer(query, context, torch.ones(3, 5))
    assert output.shape == query.shape and bool(torch.all(torch.isfinite(output)))
    print("Cross-attention smoke test passed:", tuple(output.shape))

smoke_test()`,
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
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be an iterable of tokens")
    return list(values)

class NGramModel:
    def __init__(self, n):
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("n must be an integer >= 1")
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
        return output

def smoke_test():
    model = NGramModel(2).fit(["a", "b", "a", "c"])
    assert model.next_token_probs(["a"]) == {"b": 0.5, "c": 0.5}
    assert model.next_token_probs(["unseen"]) == {"a": 0.5, "b": 0.25, "c": 0.25}
    generated = model.generate(5, seed=4)
    assert len(generated) == 5
    print("n-gram smoke test passed:", generated)

smoke_test()`,
};

// NumPy is useful here when it reinforces the same formula with a smaller array API.
// Keep these alternatives standalone and short: they are syntax recall cards, not a
// second framework-specific implementation to memorize.
const NUMPY_ALTERNATIVES: Readonly<Record<string, CodePracticeNumpyAlternative>> = {
  'l1-regression-loss': {
    code: `import numpy as np

def l1_loss(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    return np.mean(np.abs(prediction - target))`,
    exampleCode: `prediction = np.array([1.0, 2.0, 10.0])
target = np.array([2.0, 2.0, 7.0])
print(l1_loss(prediction, target))`,
    memory: ['Mean absolute error is `np.mean(np.abs(prediction - target))`.'],
  },
  'binary-cross-entropy-from-probabilities': {
    code: `import numpy as np

def binary_cross_entropy(
    probability: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-8,
) -> float:
    probability = np.asarray(probability, dtype=float)
    target = np.asarray(target, dtype=float)
    if np.any((target != 0) & (target != 1)):
        raise ValueError("target must contain only 0 and 1")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("probability must lie in [0, 1]")
    probability = np.clip(probability, eps, 1 - eps)
    loss = -target * np.log(probability) - (1 - target) * np.log(1 - probability)
    return np.mean(loss)`,
    exampleCode: `probability = np.array([0.9, 0.2])
target = np.array([1.0, 0.0])
print(binary_cross_entropy(probability, target))`,
    memory: [
      'Reject a label when `(target != 0) & (target != 1)` is true; the two comparisons must both mean “not equal.”',
      'This version accepts probabilities. For logits, use the stable softplus form instead of sigmoid → clamp → log.',
    ],
  },
  'masked-mean': {
    code: `import numpy as np

def masked_mean(features: np.ndarray, mask: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    weights = np.asarray(mask, dtype=float)[..., None]
    total = np.sum(features * weights, axis=1)
    count = np.maximum(np.sum(weights, axis=1), 1)
    return total / count`,
    exampleCode: `features = np.array([[[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]]])
mask = np.array([[1, 1, 0]])
print(masked_mean(features, mask))`,
    memory: ['`mask[..., None]` changes `(B, N)` to `(B, N, 1)` so it broadcasts over features.'],
  },
  'binary-classification-metrics': {
    code: `import numpy as np

def binary_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1,
            'accuracy': (tp + tn) / y_true.size}`,
    exampleCode: `y_true = np.array([1, 0, 1, 0])
y_pred = np.array([1, 0, 0, 1])
print(binary_classification_metrics(y_true, y_pred))`,
    memory: ['Count each confusion-matrix cell with a boolean expression followed by `np.sum`.'],
  },
  'top-k-accuracy': {
    code: `import numpy as np

def top_k_accuracy(logits: np.ndarray, labels: np.ndarray, k: int) -> float:
    logits, labels = np.asarray(logits), np.asarray(labels)
    top_k = min(k, logits.shape[1])
    ranked = np.argsort(logits, axis=1)[:, ::-1]
    candidates = ranked[:, :top_k]
    return np.mean(np.any(candidates == labels[:, None], axis=1))`,
    exampleCode: `logits = np.array([[0.1, 0.9, 0.2], [3.0, 1.0, 2.0]])
labels = np.array([1, 2])
print(top_k_accuracy(logits, labels, k=1))`,
    memory: ['NumPy `argsort` is ascending, so use `[:, ::-1]` before slicing the first `k`.'],
  },
  'single-box-iou': {
    code: `import numpy as np

def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    box_a, box_b = np.asarray(box_a, dtype=float), np.asarray(box_b, dtype=float)
    top_left = np.maximum(box_a[:2], box_b[:2])
    bottom_right = np.minimum(box_a[2:], box_b[2:])
    intersection_size = np.clip(bottom_right - top_left, 0, None)
    intersection = np.prod(intersection_size)
    area_a = np.prod(box_a[2:] - box_a[:2])
    area_b = np.prod(box_b[2:] - box_b[:2])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0`,
    exampleCode: `box_a = np.array([0.0, 0.0, 2.0, 2.0])
box_b = np.array([1.0, 1.0, 3.0, 3.0])
print(box_iou(box_a, box_b))`,
    memory: ['IoU is `intersection / (area_a + area_b - intersection)`; clip overlap widths at zero.'],
  },
  'wrapped-angular-difference': {
    code: `import numpy as np

def angular_difference(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    difference = np.asarray(prediction) - np.asarray(target)
    return np.arctan2(np.sin(difference), np.cos(difference))`,
    exampleCode: `prediction = np.deg2rad(179.0)
target = np.deg2rad(-179.0)
print(np.rad2deg(angular_difference(prediction, target)))`,
    memory: ['Wrap an angle with `atan2(sin(delta), cos(delta))` instead of modulo casework.'],
  },
  'smooth-l1-huber-loss': {
    code: `import numpy as np

def huber_loss(
    prediction: np.ndarray,
    target: np.ndarray,
    delta: float = 1.0,
) -> float:
    error = np.asarray(prediction, dtype=float) - np.asarray(target, dtype=float)
    magnitude = np.abs(error)
    quadratic = 0.5 * error ** 2
    linear = delta * (magnitude - 0.5 * delta)
    return np.mean(np.where(magnitude <= delta, quadratic, linear))`,
    exampleCode: `prediction = np.array([0.0, 2.0, 4.0])
target = np.zeros(3)
print(huber_loss(prediction, target))`,
    memory: ['For elementwise branches, compute both sides and select with `np.where(condition, a, b)`.'],
  },
  'stable-softmax-cross-entropy': {
    code: `import numpy as np

def softmax_cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=int)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_normalizers = np.log(np.sum(np.exp(shifted), axis=1))
    return np.mean(log_normalizers - shifted[np.arange(logits.shape[0]), labels])`,
    exampleCode: `logits = np.array([[2.0, 1.0, 0.1]])
labels = np.array([0])
print(f"{softmax_cross_entropy(logits, labels):.5f}")`,
    memory: ['Stability comes from the row max; `keepdims=True` preserves `(N, 1)` for broadcasting.'],
  },
  'class-weighted-cross-entropy': {
    code: `import numpy as np

def class_weighted_cross_entropy(
    logits: np.ndarray,
    labels: np.ndarray,
    class_weight: np.ndarray,
) -> float:
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=int)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    losses = np.log(np.sum(np.exp(shifted), axis=1))
    losses -= shifted[np.arange(logits.shape[0]), labels]
    example_weight = np.asarray(class_weight)[labels]
    return np.sum(losses * example_weight) / np.sum(example_weight)`,
    exampleCode: `logits = np.array([[2.0, 1.0, 0.1], [0.5, 1.5, -0.5]])
labels = np.array([0, 1])
class_weight = np.array([1.0, 2.0, 0.5])
print(class_weighted_cross_entropy(logits, labels, class_weight))`,
    memory: ['Turn class weights into per-example weights with `class_weight[labels]`.'],
  },
  'temperature-scaling-of-logits': {
    code: `import numpy as np

def temperature_scaled_probs(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(logits, dtype=float) / temperature
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)`,
    exampleCode: `logits = np.array([[1000.0, 1001.0, 1002.0]])
print(temperature_scaled_probs(logits, temperature=1.0))`,
    memory: ['Temperature divides logits before the usual stable softmax.'],
  },
  'pairwise-squared-distance': {
    code: `import numpy as np

def pairwise_squared_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    x_squared = np.sum(x * x, axis=1, keepdims=True)
    y_squared = np.sum(y * y, axis=1)[None, :]
    distances = x_squared + y_squared - 2 * x @ y.T
    return np.maximum(distances, 0)`,
    exampleCode: `x = np.array([[0.0, 0.0], [1.0, 1.0]])
y = np.array([[1.0, 0.0], [2.0, 2.0]])
print(pairwise_squared_distance(x, y))`,
    memory: ['Use `||x||² + ||y||² - 2xyᵀ` to avoid allocating an `(N, M, D)` difference array.'],
  },
  'pairwise-cosine-similarity': {
    code: `import numpy as np

def pairwise_cosine_similarity(
    x: np.ndarray,
    y: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    numerator = x @ y.T
    x_norm = np.sqrt(np.sum(x * x, axis=1))
    y_norm = np.sqrt(np.sum(y * y, axis=1))
    denominator = x_norm[:, None] * y_norm[None, :]
    return np.divide(numerator, np.maximum(denominator, eps),
                     out=np.zeros_like(numerator), where=denominator > 0)`,
    exampleCode: `x = np.array([[1.0, 0.0], [0.0, 0.0]])
y = np.array([[1.0, 0.0], [1.0, 1.0]])
print(pairwise_cosine_similarity(x, y))`,
    memory: ['`x_norm[:, None] * y_norm[None, :]` broadcasts `(N,)` and `(M,)` into `(N, M)`.'],
  },
  'nearest-centroid-classifier': {
    code: `import numpy as np

class NearestCentroidClassifier:
    def __init__(self) -> None:
        self.labels: np.ndarray | None = None
        self.centroids: np.ndarray | None = None

    def fit(self, train_X: np.ndarray, train_y: np.ndarray) -> "NearestCentroidClassifier":
        train_X, train_y = np.asarray(train_X), np.asarray(train_y)
        self.labels = np.unique(train_y)
        self.centroids = np.stack([train_X[train_y == label].mean(axis=0) for label in self.labels])
        return self

    def predict(self, test_X: np.ndarray) -> np.ndarray:
        test_X = np.asarray(test_X)
        distances = np.sum((test_X[:, None] - self.centroids[None]) ** 2, axis=-1)
        return self.labels[np.argmin(distances, axis=1)]`,
    exampleCode: `train_X = np.array([[0.0], [2.0], [10.0], [12.0]])
train_y = np.array([0, 0, 1, 1])
test_X = np.array([[0.0], [6.0], [12.0]])
model = NearestCentroidClassifier().fit(train_X, train_y)
print(model.predict(test_X))`,
    memory: ['Use a class when `fit` creates centroids that later `predict` calls reuse.'],
  },
  'iou-matrix': {
    code: `import numpy as np

def box_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    boxes1, boxes2 = np.asarray(boxes1), np.asarray(boxes2)
    top_left = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    size = np.clip(bottom_right - top_left, 0, None)
    intersection = size[..., 0] * size[..., 1]
    area1 = np.prod(boxes1[:, 2:] - boxes1[:, :2], axis=1)
    area2 = np.prod(boxes2[:, 2:] - boxes2[:, :2], axis=1)
    union = area1[:, None] + area2[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(union, dtype=float), where=union > 0)`,
    exampleCode: `boxes1 = np.array([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 1.0, 1.0]])
boxes2 = np.array([[1.0, 1.0, 3.0, 3.0], [0.0, 0.0, 2.0, 2.0]])
print(box_iou_matrix(boxes1, boxes2))`,
    memory: ['Insert the pair axes first: `(N, 1, 2)` against `(1, M, 2)` produces `(N, M, 2)`.'],
  },
  'non-maximum-suppression': {
    code: `import numpy as np

def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    boxes, scores = np.asarray(boxes, dtype=float), np.asarray(scores)
    order = np.argsort(scores, kind='stable')[::-1]
    keep = []
    while order.size:
        current, order = order[0], order[1:]
        keep.append(int(current))
        top_left = np.maximum(boxes[current, :2], boxes[order, :2])
        bottom_right = np.minimum(boxes[current, 2:], boxes[order, 2:])
        size = np.clip(bottom_right - top_left, 0, None)
        intersection = size[:, 0] * size[:, 1]
        area_current = np.prod(boxes[current, 2:] - boxes[current, :2])
        area_other = np.prod(boxes[order, 2:] - boxes[order, :2], axis=1)
        iou = intersection / np.maximum(area_current + area_other - intersection, 1e-8)
        order = order[iou <= iou_threshold]
    return keep`,
    exampleCode: `boxes = np.array([
    [0.0, 0.0, 2.0, 2.0],
    [0.5, 0.5, 2.5, 2.5],
    [5.0, 5.0, 7.0, 7.0],
])
scores = np.array([0.9, 0.8, 0.7])
print(nms(boxes, scores, iou_threshold=0.3))`,
    memory: ['NMS is one stable descending sort followed by a greedy filter of boxes above the IoU threshold.'],
  },
  'dice-loss': {
    code: `import numpy as np

def dice_loss(
    probability: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-8,
) -> float:
    probability, target = np.asarray(probability), np.asarray(target)
    axes = tuple(range(1, probability.ndim))
    intersection = np.sum(probability * target, axis=axes)
    total = np.sum(probability, axis=axes) + np.sum(target, axis=axes)
    return np.mean(1 - (2 * intersection + eps) / (total + eps))`,
    exampleCode: `prediction = np.array([[[1.0, 0.0], [0.0, 1.0]]])
target = np.array([[[1.0, 0.0], [1.0, 0.0]]])
print(dice_loss(prediction, target))`,
    memory: ['Dice is `1 - (2 * overlap + eps) / (prediction mass + target mass + eps)`.'],
  },
  'segmentation-iou-loss': {
    code: `import numpy as np

def segmentation_iou_loss(
    probability: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-8,
) -> float:
    probability, target = np.asarray(probability), np.asarray(target)
    axes = tuple(range(1, probability.ndim))
    intersection = np.sum(probability * target, axis=axes)
    union = np.sum(probability + target - probability * target, axis=axes)
    return np.mean(1 - (intersection + eps) / (union + eps))`,
    exampleCode: `prediction = np.array([[[1.0, 0.0], [0.0, 1.0]]])
target = np.array([[[1.0, 0.0], [1.0, 0.0]]])
print(segmentation_iou_loss(prediction, target))`,
    memory: ['Soft IoU uses `union = prediction + target - prediction * target` before reducing.'],
  },
  'focal-loss': {
    code: `import numpy as np

def focal_loss(
    logits: np.ndarray,
    target: np.ndarray,
    gamma: float = 2.0,
    eps: float = 1e-8,
) -> float:
    logits = np.asarray(logits, dtype=float)
    target = np.asarray(target, dtype=float)
    if logits.shape != target.shape or gamma < 0:
        raise ValueError("logits and target must match and gamma must be non-negative")
    if np.any((target != 0) & (target != 1)):
        raise ValueError("target must contain only 0 and 1")
    probability = 1 / (1 + np.exp(-np.clip(logits, -60.0, 60.0)))
    p_t = np.where(target == 1, probability, 1 - probability)
    p_t = np.clip(p_t, eps, 1.0)
    return np.mean(-((1 - p_t) ** gamma) * np.log(p_t))`,
    exampleCode: `logits = np.array([4.0, -0.4])
target = np.array([1.0, 0.0])
print(focal_loss(logits, target))`,
    memory: ['Build `p_t` once, then focal loss is `-(1 - p_t)^gamma * log(p_t)`.'],
  },
  'top-k-gather': {
    code: `import numpy as np

def topk_features(scores: np.ndarray, features: np.ndarray, k: int) -> np.ndarray:
    scores, features = np.asarray(scores), np.asarray(features)
    indices = np.argsort(scores, axis=1)[:, ::-1][:, :k]
    gather_indices = np.broadcast_to(indices[:, :, None],
                                     (*indices.shape, features.shape[2]))
    return np.take_along_axis(features, gather_indices, axis=1)`,
    exampleCode: `scores = np.array([[0.2, 0.9, 0.4]])
features = np.array([[[2.0, 0.0], [9.0, 0.0], [4.0, 0.0]]])
print(topk_features(scores, features, k=2))`,
    memory: ['NumPy’s counterpart to `torch.gather` is `np.take_along_axis`.'],
  },
  'homogeneous-coordinate-transform': {
    code: `import numpy as np

def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points, transform = np.asarray(points), np.asarray(transform)
    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    return (transform @ homogeneous.T).T[:, :3]`,
    exampleCode: `points = np.array([[1.0, 2.0, 3.0]])
transform = np.array([
    [1.0, 0.0, 0.0, 10.0],
    [0.0, 1.0, 0.0, 20.0],
    [0.0, 0.0, 1.0, 30.0],
    [0.0, 0.0, 0.0, 1.0],
])
print(transform_points(points, transform))`,
    memory: ['Append ones, multiply `transform @ homogeneous.T`, transpose back, then keep XYZ.'],
  },
  '2d-patchify-for-images': {
    code: `import numpy as np

def patchify(images: np.ndarray, patch_size: int) -> np.ndarray:
    batch, channels, height, width = images.shape
    grid_h, grid_w = height // patch_size, width // patch_size
    grid = images.reshape(batch, channels, grid_h, patch_size, grid_w, patch_size)
    grid = grid.transpose(0, 2, 4, 1, 3, 5)
    return grid.reshape(batch, grid_h * grid_w, channels * patch_size ** 2)`,
    exampleCode: `images = np.array([[
    [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
]])
print(patchify(images, patch_size=2))`,
    memory: ['Patchify is `reshape -> transpose -> reshape`; write the six intermediate axes first.'],
  },
  'unpatchify-back-to-image': {
    code: `import numpy as np

def unpatchify(
    patches: np.ndarray,
    image_shape: tuple[int, int, int],
    patch_size: int,
) -> np.ndarray:
    channels, height, width = image_shape
    grid_h, grid_w = height // patch_size, width // patch_size
    batch = patches.shape[0]
    grid = patches.reshape(batch, grid_h, grid_w, channels, patch_size, patch_size)
    grid = grid.transpose(0, 3, 1, 4, 2, 5)
    return grid.reshape(batch, channels, height, width)`,
    exampleCode: `patches = np.array([[
    [1, 2, 3, 4], [5, 6, 7, 8],
    [9, 10, 11, 12], [13, 14, 15, 16],
]])
print(unpatchify(patches, image_shape=(1, 4, 4), patch_size=2))`,
    memory: ['Unpatchify reverses the patch axis order before the final image reshape.'],
  },
  'sinusoidal-positional-encoding': {
    code: `import numpy as np

def sinusoidal_positional_encoding(length: int, dim: int) -> np.ndarray:
    positions = np.arange(length)[:, None]
    indices = np.arange(0, dim, 2)
    frequencies = np.exp(-np.log(10000.0) * indices / dim)
    angles = positions * frequencies[None, :]
    encoding = np.zeros((length, dim))
    encoding[:, 0::2] = np.sin(angles)
    encoding[:, 1::2] = np.cos(angles[:, :encoding[:, 1::2].shape[1]])
    return encoding`,
    exampleCode: `print(sinusoidal_positional_encoding(length=4, dim=5))`,
    memory: ['Even columns use sine, odd columns use cosine, and positions broadcast against frequencies.'],
  },
  'causal-attention-mask': {
    code: `import numpy as np

def make_causal_attention_mask(
    seq_lens: np.ndarray,
    max_len: int | None = None,
) -> np.ndarray:
    seq_lens = np.asarray(seq_lens, dtype=int)
    length = int(seq_lens.max()) if max_len is None else max(int(seq_lens.max()), max_len)
    positions = np.arange(length)
    valid = positions[None, :] < seq_lens[:, None]
    causal = positions[:, None] >= positions[None, :]
    return (causal[None] & valid[:, :, None] & valid[:, None, :]).astype(int)`,
    exampleCode: `seq_lens = np.array([3, 1])
print(make_causal_attention_mask(seq_lens, max_len=4))`,
    memory: ['Compare row and column positions to build `(L, L)`, then combine it with batch validity masks.'],
  },
  'rope-rotary-positional-embedding': {
    code: `import numpy as np

def apply_rope(x: np.ndarray) -> np.ndarray:
    _, seq_len, _, dim = x.shape
    pair = np.arange(dim // 2)
    angles = np.arange(seq_len)[:, None] * (10000.0 ** (-2 * pair / dim))[None, :]
    sin, cos = np.sin(angles)[None, :, None, :], np.cos(angles)[None, :, None, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    output = np.empty_like(x, dtype=float)
    output[..., 0::2] = even * cos - odd * sin
    output[..., 1::2] = even * sin + odd * cos
    return output`,
    exampleCode: `x = np.array([[
    [[1.0, 0.0, 1.0, 0.0]],
    [[1.0, 0.0, 1.0, 0.0]],
]])
print(apply_rope(x))`,
    memory: ['Treat every adjacent feature pair as a 2D rotation: `(x_even, x_odd)` times sine and cosine.'],
  },
  'scaled-dot-product-self-attention': {
    code: `import numpy as np

class MultiHeadSelfAttention:
    def __init__(self, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray, W_o: np.ndarray, num_heads: int):
        self.W_q, self.W_k, self.W_v, self.W_o = W_q, W_k, W_v, W_o
        self.num_heads = num_heads

    def __call__(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        batch, length, model_dim = x.shape
        head_dim = model_dim // self.num_heads
        split = lambda z: z.reshape(batch, length, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        q, k, v = split(x @ self.W_q), split(x @ self.W_k), split(x @ self.W_v)
        scores = q @ k.swapaxes(-1, -2) / np.sqrt(head_dim)
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        shifted = scores - np.max(scores, axis=-1, keepdims=True)
        weights = np.exp(shifted)
        weights /= np.sum(weights, axis=-1, keepdims=True)
        context = (weights @ v).transpose(0, 2, 1, 3).reshape(batch, length, model_dim)
        return context @ self.W_o`,
    exampleCode: `x = np.array([[[1.0, 0.0], [0.0, 1.0]]])
weight = np.eye(2)
layer = MultiHeadSelfAttention(weight, weight, weight, weight, num_heads=1)
print(layer(x))`,
    memory: ['A class owns reusable projection weights; `__call__` still follows project, split, attend, merge.'],
  },
  'average-precision-from-matches': {
    code: `import numpy as np

def average_precision(
    scores: np.ndarray,
    is_true_positive: np.ndarray,
    num_ground_truth: int,
) -> float:
    scores = np.asarray(scores)
    matches = np.asarray(is_true_positive, dtype=float)[np.argsort(scores)[::-1]]
    cumulative_tp = np.cumsum(matches)
    ranks = np.arange(1, scores.size + 1)
    precision = cumulative_tp / ranks
    return np.sum(precision * matches) / num_ground_truth`,
    exampleCode: `scores = np.array([0.9, 0.8, 0.7])
matches = np.array([True, False, True])
print(average_precision(scores, matches, num_ground_truth=2))`,
    memory: ['Sort by confidence, compute cumulative precision, and add precision only at true-positive ranks.'],
  },
  'manual-backprop-for-a-2-layer-mlp': {
    code: `import numpy as np

def mlp_loss_and_grads(
    X: np.ndarray,
    y: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
) -> dict[str, np.ndarray | float]:
    hidden_pre = X @ W1 + b1
    hidden = np.maximum(hidden_pre, 0)
    logits = hidden @ W2 + b2
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    rows = np.arange(X.shape[0])
    loss = np.mean(-np.log(probs[rows, y]))
    dlogits = probs.copy()
    dlogits[rows, y] -= 1
    dlogits /= X.shape[0]
    dW2, db2 = hidden.T @ dlogits, np.sum(dlogits, axis=0)
    dhidden = (dlogits @ W2.T) * (hidden_pre > 0)
    dW1, db1 = X.T @ dhidden, np.sum(dhidden, axis=0)
    return {'loss': loss, 'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}`,
    exampleCode: `X = np.array([[1.0, 2.0]])
y = np.array([1])
W1 = np.eye(2)
b1 = np.zeros(2)
W2 = np.eye(2)
b2 = np.zeros(2)
print(mlp_loss_and_grads(X, y, W1, b1, W2, b2))`,
    memory: ['Backprop order is output gradient, second layer, ReLU mask, then first layer.'],
  },
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
  .map((problem) => {
    const numpyAlternative = NUMPY_ALTERNATIVES[problem.id];
    const referenceSolution = COMPACT_REFERENCE_SOLUTIONS[problem.id] ?? problem.solutionCode;
    const tags = problem.tags ?? [];

    return {
      ...problem,
      ...ATTENTION_PROBLEM_ENRICHMENTS[problem.id],
      track: problem.track ?? 'fundamentals',
      editorStart: problem.editorStart ?? 'blank',
      order: PROGRESSIVE_ORDER[problem.id] ?? problem.order,
      difficulty: PROGRESSIVE_DIFFICULTY[problem.id] ?? problem.difficulty,
      walkthroughCode: referenceSolution,
      solutionCode: referenceSolution,
      visual: CODE_PRACTICE_VISUALS[problem.id] ?? problem.visual,
      numpyAlternative,
      tags: numpyAlternative && !tags.includes('NumPy') ? [...tags, 'NumPy'] : tags,
    };
  })
  .sort((left, right) => left.order - right.order);
