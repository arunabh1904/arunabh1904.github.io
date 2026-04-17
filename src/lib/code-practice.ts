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
    solutionCode: `import numpy as np  # Import NumPy for vectorized array operations.

def softmax_cross_entropy(logits, labels):  # Define the batch softmax cross-entropy helper.
    logits = np.asarray(logits, dtype=np.float64)  # Convert logits to a floating-point NumPy array.
    labels = np.asarray(labels)  # Convert labels to a NumPy array for validation and indexing.

    if logits.ndim != 2:  # Reject anything that is not a matrix of shape (N, C).
        raise ValueError("logits must be a 2D array of shape (N, C)")
    if labels.ndim != 1:  # Reject labels that are not a flat vector.
        raise ValueError("labels must be a 1D array of shape (N,)")

    batch_size, num_classes = logits.shape  # Unpack the batch size and class count.
    if batch_size == 0:  # Require at least one sample so the mean is defined.
        raise ValueError("logits must contain at least one sample")
    if num_classes == 0:  # Require at least one class so classification is meaningful.
        raise ValueError("logits must contain at least one class")
    if labels.shape[0] != batch_size:  # Enforce one label per example.
        raise ValueError("labels must have the same batch size as logits")
    if not np.issubdtype(labels.dtype, np.integer):  # Require integer class ids.
        raise ValueError("labels must contain integer class ids")
    if np.any(labels < 0) or np.any(labels >= num_classes):  # Reject labels outside the valid class range.
        raise ValueError("labels contain out-of-range class ids")

    shifted = logits - np.max(logits, axis=1, keepdims=True)  # Shift each row by its max for numerical stability.
    logsumexp = np.log(np.sum(np.exp(shifted), axis=1))  # Compute log(sum(exp(.))) on the stabilized logits.
    correct_class_logits = shifted[np.arange(batch_size), labels]  # Pick the shifted logit for the true class in each row.
    losses = logsumexp - correct_class_logits  # Convert each row into its cross-entropy loss.

    return float(np.mean(losses))  # Average the per-example losses and return a plain Python float.`,
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
    solutionCode: `import numpy as np  # Import NumPy so we can do vectorized box math.

def compute_iou(box, boxes):  # Define a helper that compares one box against many boxes.
    """  # Start the helper docstring.
    Compute IoU between one box and many boxes.

    Args:
        box: np.ndarray of shape (4,)
        boxes: np.ndarray of shape (M, 4)

    Returns:
        np.ndarray of shape (M,)
    """  # End the helper docstring.
    x1 = np.maximum(box[0], boxes[:, 0])  # Find the left edge of each overlap rectangle.
    y1 = np.maximum(box[1], boxes[:, 1])  # Find the top edge of each overlap rectangle.
    x2 = np.minimum(box[2], boxes[:, 2])  # Find the right edge of each overlap rectangle.
    y2 = np.minimum(box[3], boxes[:, 3])  # Find the bottom edge of each overlap rectangle.

    inter_w = np.maximum(0.0, x2 - x1)  # Clamp negative widths to zero when boxes do not overlap.
    inter_h = np.maximum(0.0, y2 - y1)  # Clamp negative heights to zero when boxes do not overlap.
    inter_area = inter_w * inter_h  # Multiply width and height to get the intersection area.

    box_area = (box[2] - box[0]) * (box[3] - box[1])  # Compute the area of the reference box.
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # Compute areas for all candidate boxes.

    union = box_area + boxes_area - inter_area  # Compute union as area_a + area_b - intersection.
    iou = np.where(union > 0.0, inter_area / union, 0.0)  # Divide safely and return 0.0 when union is zero.
    return iou  # Return the IoU values for every candidate box.


def nms(boxes, scores, iou_threshold):  # Define the main non-maximum suppression routine.
    """  # Start the function docstring.
    Perform non-maximum suppression.

    Args:
        boxes: np.ndarray of shape (N, 4)
        scores: np.ndarray of shape (N,)
        iou_threshold: float

    Returns:
        list[int]: selected indices in the order they are kept
    """  # End the function docstring.
    boxes = np.asarray(boxes, dtype=np.float64)  # Convert boxes to floating-point NumPy arrays.
    scores = np.asarray(scores, dtype=np.float64)  # Convert scores to floating-point NumPy arrays.

    if boxes.ndim != 2 or boxes.shape[1] != 4:  # Require a matrix with four coordinates per box.
        raise ValueError("boxes must have shape (N, 4)")
    if scores.ndim != 1 or scores.shape[0] != boxes.shape[0]:  # Require one score per box.
        raise ValueError("scores must have shape (N,)")
    if not (0.0 <= iou_threshold <= 1.0):  # Keep the threshold in the conventional IoU range.
        raise ValueError("iou_threshold must be in [0, 1]")

    if np.any(boxes[:, 2] < boxes[:, 0]) or np.any(boxes[:, 3] < boxes[:, 1]):  # Reject boxes with inverted corners.
        raise ValueError("invalid boxes detected")

    n = boxes.shape[0]  # Store the total number of boxes.
    if n == 0:  # Return early when there is nothing to suppress.
        return []

    order = sorted(range(n), key=lambda i: (-scores[i], i))  # Sort by score descending, then index ascending.
    keep = []  # Accumulate the indices that survive suppression.

    while order:  # Keep processing until no candidates remain.
        current = order[0]  # Take the best remaining box.
        keep.append(current)  # Record it as selected.

        if len(order) == 1:  # Stop when that was the last box.
            break

        remaining = np.array(order[1:], dtype=int)  # Convert the remaining candidate indices into a NumPy array.
        ious = compute_iou(boxes[current], boxes[remaining])  # Compute overlap against all remaining boxes at once.
        survivors = remaining[ious <= iou_threshold]  # Keep only boxes whose IoU is not strictly above the threshold.
        order = survivors.tolist()  # Convert back to a Python list for the next loop iteration.

    return keep  # Return the selected box indices in keep order.`,
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
    solutionCode: `import numpy as np  # Import NumPy for mask construction and broadcasting.

def make_causal_attention_mask(seq_lens, max_len=None):  # Define the batch causal mask builder.
    seq_lens = np.asarray(seq_lens)  # Convert the sequence lengths into a NumPy array.

    if seq_lens.ndim != 1:  # Require a flat list of lengths.
        raise ValueError("seq_lens must be a 1D array")
    if seq_lens.size == 0:  # Disallow empty batches.
        raise ValueError("seq_lens must not be empty")
    if not np.issubdtype(seq_lens.dtype, np.integer):  # Require integer lengths.
        raise ValueError("seq_lens must contain integers")
    if np.any(seq_lens < 0):  # Reject negative sequence lengths.
        raise ValueError("seq_lens must be non-negative")

    T = int(seq_lens.max())  # Start with the longest valid sequence length.
    if max_len is not None:  # Allow the caller to force a wider mask.
        if isinstance(max_len, bool) or not isinstance(max_len, (int, np.integer)):  # Reject non-integer max_len values.
            raise ValueError("max_len must be an integer or None")
        if max_len < 0:  # Disallow negative padding lengths.
            raise ValueError("max_len must be non-negative")
        T = max(T, int(max_len))  # Use whichever length is larger.

    valid = np.arange(T) < seq_lens[:, None]  # Mark positions that fall inside each example's valid length.
    causal = np.tri(T, dtype=np.int64)  # Build the lower-triangular causal template once.
    mask = causal[None, :, :] * valid[:, :, None] * valid[:, None, :]  # Zero out padded rows and columns per batch element.
    return mask.astype(np.int64)  # Return an integer mask as requested.`,
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
    solutionCode: `def _coerce_binary_labels(values, name):  # Normalize and validate a binary label sequence.
    if isinstance(values, (str, bytes)):  # Reject strings because they are not label sequences.
        raise ValueError(f"{name} must be a 1D sequence of binary labels")

    try:  # Try to materialize the input as a list so we can inspect every element.
        items = list(values)
    except TypeError as exc:  # Convert non-iterables into a clean validation error.
        raise ValueError(f"{name} must be a 1D sequence of binary labels") from exc

    if not items:  # Disallow empty inputs because the metrics would be undefined.
        raise ValueError(f"{name} must not be empty")

    for item in items:  # Inspect each item for shape and label validity.
        if hasattr(item, "__iter__") and not isinstance(item, (str, bytes)):  # Reject nested sequences.
            raise ValueError(f"{name} must be one-dimensional")
        if item not in (0, 1):  # Only binary class labels are allowed.
            raise ValueError(f"{name} must contain only 0 and 1")

    return items  # Return the validated flat list of binary labels.


def binary_classification_metrics(y_true, y_pred):  # Define the metric computation entry point.
    y_true = _coerce_binary_labels(y_true, "y_true")  # Validate and normalize the ground-truth labels.
    y_pred = _coerce_binary_labels(y_pred, "y_pred")  # Validate and normalize the predicted labels.

    if len(y_true) != len(y_pred):  # Require the two sequences to be aligned sample by sample.
        raise ValueError("y_true and y_pred must have the same length")

    tp = tn = fp = fn = 0  # Initialize the four confusion-matrix counts.
    for truth, pred in zip(y_true, y_pred):  # Walk through paired labels once.
        if truth == 1 and pred == 1:  # Count true positives.
            tp += 1
        elif truth == 0 and pred == 0:  # Count true negatives.
            tn += 1
        elif truth == 0 and pred == 1:  # Count false positives.
            fp += 1
        else:  # The remaining case is a false negative.
            fn += 1

    total = len(y_true)  # Store the number of evaluated examples.
    precision_den = tp + fp  # Precision divides by predicted positives.
    recall_den = tp + fn  # Recall divides by actual positives.

    precision = tp / precision_den if precision_den else 0.0  # Return 0.0 when precision is undefined.
    recall = tp / recall_den if recall_den else 0.0  # Return 0.0 when recall is undefined.
    f1_den = precision + recall  # F1 is based on the harmonic mean of precision and recall.
    f1 = (2.0 * precision * recall / f1_den) if f1_den else 0.0  # Return 0.0 when both are zero.
    accuracy = (tp + tn) / total  # Accuracy is the fraction of correct predictions.

    return {  # Package all counts and metrics into a dictionary.
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
    solutionCode: `import numpy as np  # Import NumPy for vectorized linear algebra.

def pairwise_cosine_similarity(x, y):  # Define the pairwise cosine similarity function.
    x = np.asarray(x, dtype=np.float64)  # Convert x to a floating-point matrix.
    y = np.asarray(y, dtype=np.float64)  # Convert y to a floating-point matrix.

    if x.ndim != 2 or y.ndim != 2:  # Require both inputs to be matrices.
        raise ValueError("x and y must be 2D arrays")
    if x.shape[1] != y.shape[1]:  # Require the same feature dimension on both sides.
        raise ValueError("x and y must have the same feature dimension")
    if x.shape[1] == 0:  # Disallow empty feature vectors.
        raise ValueError("feature dimension must be positive")

    x_norms = np.linalg.norm(x, axis=1)  # Compute the norm of each row in x.
    y_norms = np.linalg.norm(y, axis=1)  # Compute the norm of each row in y.
    similarities = x @ y.T  # Compute all pairwise dot products at once.
    denominator = x_norms[:, None] * y_norms[None, :]  # Broadcast the norm product into an (N, M) denominator.

    return np.divide(  # Divide safely so zero-norm rows produce zeros instead of warnings.
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
    solutionCode: `import numpy as np  # Import NumPy for sorting and vectorized comparisons.

def top_k_accuracy(logits, labels, k):  # Define the top-k accuracy routine.
    logits = np.asarray(logits, dtype=np.float64)  # Convert logits to a floating-point matrix.
    labels = np.asarray(labels)  # Convert labels to a NumPy array for validation.

    if logits.ndim != 2:  # Require a 2D logit matrix.
        raise ValueError("logits must be a 2D array of shape (N, C)")
    if labels.ndim != 1:  # Require a flat label vector.
        raise ValueError("labels must be a 1D array of shape (N,)")
    if logits.shape[0] == 0:  # Require at least one example.
        raise ValueError("logits must contain at least one sample")
    if logits.shape[1] == 0:  # Require at least one class.
        raise ValueError("logits must contain at least one class")
    if labels.shape[0] != logits.shape[0]:  # Require aligned batch sizes.
        raise ValueError("labels must have the same batch size as logits")
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):  # Reject non-integer k values.
        raise ValueError("k must be a positive integer")
    if k <= 0:  # Require k to be positive.
        raise ValueError("k must be a positive integer")
    if not np.issubdtype(labels.dtype, np.integer):  # Require integer class ids.
        raise ValueError("labels must contain integer class ids")
    if np.any(labels < 0) or np.any(labels >= logits.shape[1]):  # Reject labels outside the valid class range.
        raise ValueError("labels contain out-of-range class ids")

    top_k = min(int(k), logits.shape[1])  # Clamp k so it never exceeds the number of classes.
    ranked = np.argsort(-logits, axis=1)[:, :top_k]  # Take the top-k class indices per row.
    hits = np.any(ranked == labels[:, None], axis=1)  # Check whether the true label appears in each top-k set.
    return float(np.mean(hits))  # Convert the boolean hit rate into a Python float.`,
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
    solutionCode: `import numpy as np  # Import NumPy so we can broadcast box coordinates.

def _validate_boxes(boxes, name):  # Define a helper that validates box arrays.
    boxes = np.asarray(boxes, dtype=np.float64)  # Convert boxes to a floating-point matrix.

    if boxes.ndim != 2 or boxes.shape[1] != 4:  # Require one row per box with four coordinates.
        raise ValueError(f"{name} must have shape (N, 4)")
    if np.any(boxes[:, 2] < boxes[:, 0]) or np.any(boxes[:, 3] < boxes[:, 1]):  # Reject inverted boxes.
        raise ValueError(f"{name} contains invalid boxes")

    return boxes  # Return the validated box matrix.


def box_iou_matrix(boxes1, boxes2):  # Define the pairwise IoU matrix function.
    boxes1 = _validate_boxes(boxes1, "boxes1")  # Validate the first box set.
    boxes2 = _validate_boxes(boxes2, "boxes2")  # Validate the second box set.

    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])  # Compute the left edge of every pairwise overlap.
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])  # Compute the top edge of every pairwise overlap.
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])  # Compute the right edge of every pairwise overlap.
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])  # Compute the bottom edge of every pairwise overlap.

    inter_w = np.maximum(0.0, x2 - x1)  # Clamp negative overlap widths to zero.
    inter_h = np.maximum(0.0, y2 - y1)  # Clamp negative overlap heights to zero.
    inter_area = inter_w * inter_h  # Multiply width and height to get pairwise intersection areas.

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # Compute areas for boxes1.
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # Compute areas for boxes2.
    union = area1[:, None] + area2[None, :] - inter_area  # Compute pairwise unions from areas and intersections.

    return np.divide(  # Divide safely so degenerate pairs fall back to zero.
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
    solutionCode: `import numpy as np  # Import NumPy for centroid computation and distance math.

def nearest_centroid_predict(train_X, train_y, test_X):  # Define the nearest-centroid classifier.
    train_X = np.asarray(train_X, dtype=np.float64)  # Convert training features to floating-point arrays.
    train_y = np.asarray(train_y)  # Convert training labels to a NumPy array.
    test_X = np.asarray(test_X, dtype=np.float64)  # Convert test features to floating-point arrays.

    if train_X.ndim != 2:  # Require a 2D training feature matrix.
        raise ValueError("train_X must be a 2D array of shape (N, D)")
    if train_y.ndim != 1:  # Require a 1D label vector.
        raise ValueError("train_y must be a 1D array of shape (N,)")
    if test_X.ndim != 2:  # Require a 2D test feature matrix.
        raise ValueError("test_X must be a 2D array of shape (M, D)")
    if train_X.shape[0] == 0:  # Require at least one training sample.
        raise ValueError("train_X must contain at least one sample")
    if train_X.shape[0] != train_y.shape[0]:  # Require one label per training sample.
        raise ValueError("train_X and train_y must have the same number of samples")
    if train_X.shape[1] != test_X.shape[1]:  # Require matching feature dimensions.
        raise ValueError("train_X and test_X must have the same feature dimension")
    if not np.issubdtype(train_y.dtype, np.integer):  # Require integer class labels.
        raise ValueError("train_y must contain integer class labels")

    labels = np.unique(train_y)  # Sort the unique labels so ties resolve toward smaller labels.
    if labels.size == 0:  # Require at least one class in the training labels.
        raise ValueError("train_y must contain at least one class")

    centroids = np.vstack([train_X[train_y == label].mean(axis=0) for label in labels])  # Compute one centroid per class.
    deltas = test_X[:, None, :] - centroids[None, :, :]  # Broadcast test points against all centroids.
    squared_distances = np.sum(deltas * deltas, axis=2)  # Use squared Euclidean distance to avoid a square root.
    nearest_indices = np.argmin(squared_distances, axis=1)  # Pick the nearest centroid for each test point.
    return labels[nearest_indices]  # Map centroid indices back to class labels.`,
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
    solutionCode: `import numpy as np  # Import NumPy for stable softmax math.

def temperature_scaled_probs(logits, temperature):  # Define the temperature-scaled softmax helper.
    logits = np.asarray(logits, dtype=np.float64)  # Convert logits to a floating-point matrix.

    if isinstance(temperature, (bool, np.bool_)):  # Reject booleans, which are not meaningful temperatures.
        raise ValueError("temperature must be a positive float")
    try:  # Try to coerce the temperature to a Python float.
        temperature = float(temperature)
    except (TypeError, ValueError):  # Convert invalid scalars into a clean validation error.
        raise ValueError("temperature must be a positive float")

    if logits.ndim != 2:  # Require a 2D matrix of logits.
        raise ValueError("logits must be a 2D array of shape (N, C)")
    if logits.shape[0] == 0:  # Require at least one row.
        raise ValueError("logits must contain at least one sample")
    if logits.shape[1] == 0:  # Require at least one class.
        raise ValueError("logits must contain at least one class")
    if not np.isfinite(temperature) or temperature <= 0:  # Require a finite positive temperature.
        raise ValueError("temperature must be a positive float")

    scaled_logits = logits / temperature  # Rescale the logits before softmax.
    shifted = scaled_logits - np.max(scaled_logits, axis=1, keepdims=True)  # Stabilize by removing each row max.
    exp_shifted = np.exp(shifted)  # Exponentiate the shifted logits.
    return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)  # Normalize each row into probabilities.`,
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
    signature: `def sinusoidal_positional_encoding(length, dim):
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
      'Fill even columns with `np.sin` and odd columns with `np.cos` after computing the shared angles.',
      'If `dim` is odd, the last column still belongs to the even-column branch.',
    ],
    solutionNotes: [
      'Sinusoidal positional encoding is just a deterministic lookup table: each position gets a vector of sines and cosines at frequencies that decay geometrically across the embedding dimension.',
      'The implementation is compact if you compute one denominator per column pair and then broadcast positions across those frequencies. That also makes the odd-dimension case work naturally, because the final column is just the next even slot.',
    ],
    solutionCode: `import numpy as np  # Import NumPy for vectorized trigonometry and indexing.

def sinusoidal_positional_encoding(length, dim):  # Define the classic Transformer position embedding table.
    if isinstance(length, bool) or not isinstance(length, (int, np.integer)):  # Reject non-integer or boolean lengths.
        raise ValueError("length must be a positive integer")
    if isinstance(dim, bool) or not isinstance(dim, (int, np.integer)):  # Reject non-integer or boolean dimensions.
        raise ValueError("dim must be a positive integer")
    if length <= 0 or dim <= 0:  # Require positive sizes.
        raise ValueError("length and dim must be positive integers")

    positions = np.arange(length, dtype=np.float64)[:, None]  # Build the column vector of token positions.
    column_indices = np.arange(dim)  # Build the feature-column indices.
    even_indices = 2 * (column_indices // 2)  # Reuse the same frequency for each even/odd pair.
    angle_rates = np.power(10000.0, even_indices / dim)  # Compute the denominator for each column.
    angles = positions / angle_rates  # Broadcast positions against the per-column rates.

    encoding = np.empty((length, dim), dtype=np.float64)  # Allocate the output table.
    encoding[:, 0::2] = np.sin(angles[:, 0::2])  # Fill even columns with sine values.
    encoding[:, 1::2] = np.cos(angles[:, 1::2])  # Fill odd columns with cosine values.
    return encoding  # Return the completed positional encoding matrix.`,
    starterCode: `import numpy as np

def sinusoidal_positional_encoding(length, dim):
    # TODO:
    # 1. Validate that length and dim are positive integers.
    # 2. Build the sinusoidal table with the standard even/odd formulas.
    raise NotImplementedError("Implement sinusoidal_positional_encoding")

sample_length = 4
sample_dim = 5

print(sinusoidal_positional_encoding(sample_length, sample_dim))`,
    packages: ['numpy'],
    tags: ['NumPy', 'Sequence Modeling', 'Embeddings'],
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
    signature: `def unpatchify(patches, image_shape, patch_size):
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
    solutionCode: `import numpy as np  # Import NumPy for the reshape and transpose steps.

def unpatchify(patches, image_shape, patch_size):  # Define the inverse patch reconstruction routine.
    patches = np.asarray(patches)  # Convert the patch tensor to a NumPy array.
    image_shape = np.asarray(image_shape)  # Convert the image shape tuple into an array for validation.

    if patches.ndim != 3:  # Require a batch of patch sequences.
        raise ValueError("patches must have shape (B, N, C * P * P)")
    if image_shape.ndim != 1 or image_shape.size != 3:  # Require exactly three image shape values.
        raise ValueError("image_shape must have shape (3,)")
    if not np.issubdtype(image_shape.dtype, np.integer):  # Require integer shape metadata.
        raise ValueError("image_shape must contain integers")
    if isinstance(patch_size, bool) or not isinstance(patch_size, (int, np.integer)):  # Reject non-integer patch sizes.
        raise ValueError("patch_size must be a positive integer")
    if patch_size <= 0:  # Require a positive patch size.
        raise ValueError("patch_size must be a positive integer")

    C, H, W = (int(value) for value in image_shape)  # Unpack the channel, height, and width.
    if C <= 0 or H <= 0 or W <= 0:  # Require positive image dimensions.
        raise ValueError("image_shape must contain positive integers")
    if H % patch_size != 0 or W % patch_size != 0:  # Require the spatial dimensions to divide evenly into patches.
        raise ValueError("image dimensions must be divisible by patch_size")

    grid_h = H // patch_size  # Compute the number of patch rows.
    grid_w = W // patch_size  # Compute the number of patch columns.
    expected_num_patches = grid_h * grid_w  # Compute the total patch count per image.
    expected_patch_dim = C * patch_size * patch_size  # Compute the flattened size of one patch.

    if patches.shape[1] != expected_num_patches:  # Require the patch count to match the image grid.
        raise ValueError("patch count does not match image_shape and patch_size")
    if patches.shape[2] != expected_patch_dim:  # Require each flattened patch to have the expected size.
        raise ValueError("patch dimension does not match image_shape and patch_size")

    batch_size = patches.shape[0]  # Store the batch size for reshaping.
    reshaped = patches.reshape(batch_size, grid_h, grid_w, C, patch_size, patch_size)  # Recover the patch grid and patch pixels.
    reconstructed = reshaped.transpose(0, 3, 1, 4, 2, 5)  # Interleave patch-grid axes with within-patch axes.
    return reconstructed.reshape(batch_size, C, H, W)  # Collapse everything back into image tensors.`,
    starterCode: `import numpy as np

def unpatchify(patches, image_shape, patch_size):
    patches = np.asarray(patches)
    image_shape = np.asarray(image_shape)

    # TODO:
    # 1. Validate the tensor shapes and patch_size.
    # 2. Reshape and transpose the patch grid back into (B, C, H, W).
    raise NotImplementedError("Implement unpatchify")

sample_patches = np.array([
    [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
])
sample_image_shape = (1, 4, 4)

print(unpatchify(sample_patches, sample_image_shape, patch_size=2))`,
    packages: ['numpy'],
    tags: ['NumPy', 'Computer Vision', 'Transformers'],
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
    signature: `def patchify(images, patch_size):
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
    solutionCode: `import numpy as np  # Import NumPy for patch reshaping and transposition.

def patchify(images, patch_size):  # Define the patch extraction routine.
    images = np.asarray(images)  # Convert the input image batch to a NumPy array.

    if images.ndim != 4:  # Require a batch tensor with channel and spatial dimensions.
        raise ValueError("images must have shape (B, C, H, W)")
    if isinstance(patch_size, bool) or not isinstance(patch_size, (int, np.integer)):  # Reject non-integer patch sizes.
        raise ValueError("patch_size must be a positive integer")
    if patch_size <= 0:  # Require a positive patch size.
        raise ValueError("patch_size must be a positive integer")

    batch_size, channels, height, width = images.shape  # Unpack the image batch dimensions.
    if channels <= 0 or height <= 0 or width <= 0:  # Require positive channel and spatial sizes.
        raise ValueError("images must have positive channel and spatial dimensions")
    if height % patch_size != 0 or width % patch_size != 0:  # Require the image to divide evenly into patches.
        raise ValueError("image dimensions must be divisible by patch_size")

    grid_h = height // patch_size  # Compute the number of patch rows.
    grid_w = width // patch_size  # Compute the number of patch columns.
    reshaped = images.reshape(batch_size, channels, grid_h, patch_size, grid_w, patch_size)  # Expose the patch grid explicitly.
    patches = reshaped.transpose(0, 2, 4, 1, 3, 5)  # Move the grid axes ahead of the channel and pixel axes.
    return patches.reshape(batch_size, grid_h * grid_w, channels * patch_size * patch_size)  # Flatten each patch into a token.`,
    starterCode: `import numpy as np

def patchify(images, patch_size):
    images = np.asarray(images)

    # TODO:
    # 1. Validate the tensor shape and patch_size.
    # 2. Reshape and transpose the image grid into flattened patch tokens.
    raise NotImplementedError("Implement patchify")

sample_images = np.array([
    [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]
])
print(patchify(sample_images, patch_size=2))`,
    packages: ['numpy'],
    tags: ['NumPy', 'Computer Vision', 'Patch Embeddings'],
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
    signature: `def apply_rope(x):
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
    solutionCode: `import numpy as np  # Import NumPy for rotary embedding math.

def apply_rope(x):  # Define the rotary positional embedding transform.
    x = np.asarray(x, dtype=np.float64)  # Convert the input tensor to floating-point values.

    if x.ndim != 4:  # Require the expected (B, T, H, D) tensor layout.
        raise ValueError("x must have shape (B, T, H, D)")
    if np.any(np.array(x.shape) <= 0):  # Require all dimensions to be positive.
        raise ValueError("x must have positive dimensions")
    if x.shape[-1] % 2 != 0:  # Require an even feature dimension so pairs can be rotated.
        raise ValueError("D must be even")

    batch_size, seq_len, num_heads, dim = x.shape  # Unpack the tensor dimensions.
    half_dim = dim // 2  # Count the number of even/odd feature pairs.

    positions = np.arange(seq_len, dtype=np.float64)[:, None]  # Build the sequence-position axis.
    pair_indices = np.arange(half_dim, dtype=np.float64)  # Build the index for each feature pair.
    inv_freq = 1.0 / np.power(10000.0, (2.0 * pair_indices) / dim)  # Compute the RoPE inverse frequencies.
    angles = positions * inv_freq[None, :]  # Turn positions and frequencies into rotation angles.

    sin = np.sin(angles)[None, :, None, :]  # Broadcast sine values over batch and head dimensions.
    cos = np.cos(angles)[None, :, None, :]  # Broadcast cosine values over batch and head dimensions.

    x_even = x[..., 0::2]  # Select the even coordinates from each feature pair.
    x_odd = x[..., 1::2]  # Select the odd coordinates from each feature pair.

    out = np.empty_like(x)  # Allocate the output tensor.
    out[..., 0::2] = x_even * cos - x_odd * sin  # Rotate the even coordinates.
    out[..., 1::2] = x_even * sin + x_odd * cos  # Rotate the odd coordinates.
    return out  # Return the rotated tensor with the same shape as the input.`,
    starterCode: `import numpy as np

def apply_rope(x):
    x = np.asarray(x)

    # TODO:
    # 1. Validate the tensor shape and check that D is even.
    # 2. Build the RoPE sine/cosine tables and rotate each feature pair.
    raise NotImplementedError("Implement apply_rope")

sample_x = np.array([
    [[[1.0, 0.0, 1.0, 0.0]], [[1.0, 0.0, 1.0, 0.0]]]
])
print(apply_rope(sample_x))`,
    packages: ['numpy'],
    tags: ['NumPy', 'Attention', 'Transformers'],
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
    signature: `def self_attention(x, W_q, W_k, W_v, W_o, num_heads, mask=None):
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
    solutionCode: `import numpy as np  # Import NumPy for projection, masking, and attention math.

def _stable_softmax(logits):  # Define a numerically stable softmax helper.
    logits = np.asarray(logits, dtype=np.float64)  # Convert the logits to floating-point values.
    max_logits = np.max(logits, axis=-1, keepdims=True)  # Find the maximum score in each row.
    max_logits = np.where(np.isfinite(max_logits), max_logits, 0.0)  # Replace non-finite maxima with zero.
    shifted = logits - max_logits  # Shift the scores so the largest value becomes zero.
    exp_shifted = np.exp(shifted)  # Exponentiate the shifted scores.
    exp_shifted = np.where(np.isfinite(logits), exp_shifted, 0.0)  # Zero out any entries that were non-finite.
    denom = np.sum(exp_shifted, axis=-1, keepdims=True)  # Sum the exponentials row by row.
    return np.divide(exp_shifted, denom, out=np.zeros_like(exp_shifted), where=denom > 0)  # Normalize safely.


def self_attention(x, W_q, W_k, W_v, W_o, num_heads, mask=None):  # Define the multi-head self-attention block.
    x = np.asarray(x, dtype=np.float64)  # Convert the input sequence to floating-point values.
    W_q = np.asarray(W_q, dtype=np.float64)  # Convert the query projection matrix.
    W_k = np.asarray(W_k, dtype=np.float64)  # Convert the key projection matrix.
    W_v = np.asarray(W_v, dtype=np.float64)  # Convert the value projection matrix.
    W_o = np.asarray(W_o, dtype=np.float64)  # Convert the output projection matrix.

    if x.ndim != 3:  # Require the standard (B, T, D_model) shape.
        raise ValueError("x must have shape (B, T, D_model)")
    if np.any(np.array(x.shape) <= 0):  # Require every dimension to be positive.
        raise ValueError("x must have positive dimensions")
    if isinstance(num_heads, bool) or not isinstance(num_heads, (int, np.integer)):  # Reject invalid head counts.
        raise ValueError("num_heads must be a positive integer")
    if num_heads <= 0:  # Require at least one attention head.
        raise ValueError("num_heads must be a positive integer")

    batch_size, seq_len, model_dim = x.shape  # Unpack the batch, token, and model dimensions.
    if model_dim % num_heads != 0:  # Require the model width to split evenly across heads.
        raise ValueError("num_heads must divide D_model")

    for matrix, name in ((W_q, "W_q"), (W_k, "W_k"), (W_v, "W_v"), (W_o, "W_o")):  # Validate all projection matrices.
        if matrix.ndim != 2 or matrix.shape != (model_dim, model_dim):  # Require square matrices of the model width.
            raise ValueError(f"{name} must have shape (D_model, D_model)")

    head_dim = model_dim // num_heads  # Compute the per-head feature width.

    q = x @ W_q  # Project the input into queries.
    k = x @ W_k  # Project the input into keys.
    v = x @ W_v  # Project the input into values.

    q = q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)  # Split queries into heads.
    k = k.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)  # Split keys into heads.
    v = v.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)  # Split values into heads.

    scores = np.matmul(q, np.swapaxes(k, -1, -2)) / np.sqrt(head_dim)  # Compute scaled dot-product attention scores.

    if mask is not None:  # Apply an optional attention mask if provided.
        mask_arr = np.asarray(mask)  # Convert the mask to a NumPy array.
        try:  # Try to broadcast the mask to the score tensor.
            mask_broadcast = np.broadcast_to(mask_arr, scores.shape)
        except ValueError as exc:  # Recast broadcast failures as validation errors.
            raise ValueError("mask must be broadcastable to (B, H, T, T)") from exc
        if not np.all((mask_broadcast == 0) | (mask_broadcast == 1)):  # Require binary mask values.
            raise ValueError("mask must contain only 0 and 1 values")
        scores = np.where(mask_broadcast.astype(bool), scores, -np.inf)  # Block masked positions before softmax.

    attn = _stable_softmax(scores)  # Convert scores into attention weights.
    context = np.matmul(attn, v)  # Weight the values by the attention distribution.
    context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, model_dim)  # Recombine the heads.
    return context @ W_o  # Apply the final output projection.`,
    starterCode: `import numpy as np

def self_attention(x, W_q, W_k, W_v, W_o, num_heads, mask=None):
    x = np.asarray(x)
    W_q = np.asarray(W_q)
    W_k = np.asarray(W_k)
    W_v = np.asarray(W_v)
    W_o = np.asarray(W_o)

    # TODO:
    # 1. Validate shapes and make sure num_heads divides D_model.
    # 2. Project x into q/k/v, split into heads, apply masked attention, and combine the heads.
    raise NotImplementedError("Implement self_attention")

sample_x = np.array([[[1.0, 0.0], [0.0, 1.0]]])
sample_w = np.array([[1.0, 0.0], [0.0, 1.0]])

print(self_attention(sample_x, sample_w, sample_w, sample_w, sample_w, num_heads=1))`,
    packages: ['numpy'],
    tags: ['NumPy', 'Attention', 'Transformers'],
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
    signature: `def cross_attention(query_x, context_x, W_q, W_k, W_v, W_o, num_heads, mask=None):
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
    solutionCode: `import numpy as np  # Import NumPy for cross-attention math.

def _stable_softmax(logits):  # Define a numerically stable softmax helper.
    logits = np.asarray(logits, dtype=np.float64)  # Convert logits to floating-point values.
    max_logits = np.max(logits, axis=-1, keepdims=True)  # Find the max score in each row.
    max_logits = np.where(np.isfinite(max_logits), max_logits, 0.0)  # Replace non-finite maxima with zero.
    shifted = logits - max_logits  # Shift the logits before exponentiating.
    exp_shifted = np.exp(shifted)  # Exponentiate the shifted logits.
    exp_shifted = np.where(np.isfinite(logits), exp_shifted, 0.0)  # Clear out non-finite positions.
    denom = np.sum(exp_shifted, axis=-1, keepdims=True)  # Sum each softmax row.
    return np.divide(exp_shifted, denom, out=np.zeros_like(exp_shifted), where=denom > 0)  # Normalize safely.


def cross_attention(query_x, context_x, W_q, W_k, W_v, W_o, num_heads, mask=None):  # Define the cross-attention block.
    query_x = np.asarray(query_x, dtype=np.float64)  # Convert query inputs to floating-point values.
    context_x = np.asarray(context_x, dtype=np.float64)  # Convert context inputs to floating-point values.
    W_q = np.asarray(W_q, dtype=np.float64)  # Convert the query projection matrix.
    W_k = np.asarray(W_k, dtype=np.float64)  # Convert the key projection matrix.
    W_v = np.asarray(W_v, dtype=np.float64)  # Convert the value projection matrix.
    W_o = np.asarray(W_o, dtype=np.float64)  # Convert the output projection matrix.

    if query_x.ndim != 3 or context_x.ndim != 3:  # Require both inputs to be batched sequence tensors.
        raise ValueError("query_x and context_x must have shape (B, T, D_model)")
    if np.any(np.array(query_x.shape) <= 0) or np.any(np.array(context_x.shape) <= 0):  # Require positive dimensions.
        raise ValueError("inputs must have positive dimensions")
    if query_x.shape[0] != context_x.shape[0]:  # Require a shared batch size.
        raise ValueError("query_x and context_x must have the same batch size")
    if query_x.shape[2] != context_x.shape[2]:  # Require a shared model width.
        raise ValueError("query_x and context_x must have the same model dimension")
    if isinstance(num_heads, bool) or not isinstance(num_heads, (int, np.integer)):  # Reject invalid head counts.
        raise ValueError("num_heads must be a positive integer")
    if num_heads <= 0:  # Require at least one head.
        raise ValueError("num_heads must be a positive integer")

    batch_size, query_len, model_dim = query_x.shape  # Unpack the query tensor dimensions.
    context_len = context_x.shape[1]  # Store the context sequence length.
    if model_dim % num_heads != 0:  # Require the model width to split evenly across heads.
        raise ValueError("num_heads must divide D_model")

    for matrix, name in ((W_q, "W_q"), (W_k, "W_k"), (W_v, "W_v"), (W_o, "W_o")):  # Validate all projection matrices.
        if matrix.ndim != 2 or matrix.shape != (model_dim, model_dim):  # Require square matrices of model width.
            raise ValueError(f"{name} must have shape (D_model, D_model)")

    head_dim = model_dim // num_heads  # Compute the per-head feature width.

    q = query_x @ W_q  # Project the query sequence into query vectors.
    k = context_x @ W_k  # Project the context sequence into key vectors.
    v = context_x @ W_v  # Project the context sequence into value vectors.

    q = q.reshape(batch_size, query_len, num_heads, head_dim).transpose(0, 2, 1, 3)  # Split queries into heads.
    k = k.reshape(batch_size, context_len, num_heads, head_dim).transpose(0, 2, 1, 3)  # Split keys into heads.
    v = v.reshape(batch_size, context_len, num_heads, head_dim).transpose(0, 2, 1, 3)  # Split values into heads.

    scores = np.matmul(q, np.swapaxes(k, -1, -2)) / np.sqrt(head_dim)  # Compute scaled dot-product attention scores.

    if mask is not None:  # Apply an optional mask if the caller supplied one.
        mask_arr = np.asarray(mask)  # Convert the mask to a NumPy array.
        try:  # Try to broadcast the mask to the score tensor.
            mask_broadcast = np.broadcast_to(mask_arr, scores.shape)
        except ValueError as exc:  # Recast broadcast failures as validation errors.
            raise ValueError("mask must be broadcastable to (B, H, Tq, Tk)") from exc
        if not np.all((mask_broadcast == 0) | (mask_broadcast == 1)):  # Require binary mask values.
            raise ValueError("mask must contain only 0 and 1 values")
        scores = np.where(mask_broadcast.astype(bool), scores, -np.inf)  # Block masked positions before softmax.

    attn = _stable_softmax(scores)  # Turn scores into attention weights.
    context = np.matmul(attn, v)  # Apply the weights to the value vectors.
    context = context.transpose(0, 2, 1, 3).reshape(batch_size, query_len, model_dim)  # Merge heads back into the model width.
    return context @ W_o  # Apply the output projection.`,
    starterCode: `import numpy as np

def cross_attention(query_x, context_x, W_q, W_k, W_v, W_o, num_heads, mask=None):
    query_x = np.asarray(query_x)
    context_x = np.asarray(context_x)
    W_q = np.asarray(W_q)
    W_k = np.asarray(W_k)
    W_v = np.asarray(W_v)
    W_o = np.asarray(W_o)

    # TODO:
    # 1. Validate the shapes and make sure num_heads divides D_model.
    # 2. Project query_x into q, context_x into k/v, apply masked attention, and combine the heads.
    raise NotImplementedError("Implement cross_attention")

sample_query = np.array([[[1.0, 0.0]]])
sample_context = np.array([[[1.0, 0.0], [0.0, 1.0]]])
sample_w = np.array([[1.0, 0.0], [0.0, 1.0]])

print(cross_attention(sample_query, sample_context, sample_w, sample_w, sample_w, sample_w, num_heads=1))`,
    packages: ['numpy'],
    tags: ['NumPy', 'Attention', 'Transformers'],
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
    signature: `def mlp_loss_and_grads(X, y, W1, b1, W2, b2):
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
      'For softmax cross-entropy, the gradient with respect to the logits is `probs - one_hot(y)`, averaged over the batch.',
      'Backpropagate from the output layer into the hidden layer before multiplying by the ReLU mask.',
      'Return the gradients in a dictionary so the caller can inspect each parameter separately.',
    ],
    solutionNotes: [
      'The forward pass is just affine, ReLU, affine, and mean softmax cross-entropy. Once you have the probabilities, the gradient of the loss with respect to the logits is the usual `probs - one_hot` term divided by the batch size.',
      'From there, the remaining gradients follow by the chain rule: the second affine layer gives `dW2` and `db2`, and the upstream gradient passes through the ReLU mask before producing `dW1` and `db1`.',
    ],
    solutionCode: `import numpy as np  # Import NumPy for the forward and backward passes.

def _stable_softmax(logits):  # Define a numerically stable softmax helper.
    logits = np.asarray(logits, dtype=np.float64)  # Convert the logits to floating-point values.
    max_logits = np.max(logits, axis=-1, keepdims=True)  # Find the row-wise maximum.
    shifted = logits - max_logits  # Shift scores so the largest value in each row is zero.
    exp_shifted = np.exp(shifted)  # Exponentiate the shifted scores.
    denom = np.sum(exp_shifted, axis=-1, keepdims=True)  # Sum the exponentials row by row.
    return np.divide(exp_shifted, denom, out=np.zeros_like(exp_shifted), where=denom > 0)  # Normalize safely.


def mlp_loss_and_grads(X, y, W1, b1, W2, b2):  # Define the 2-layer MLP loss and gradient function.
    X = np.asarray(X, dtype=np.float64)  # Convert inputs to floating-point arrays.
    y = np.asarray(y, dtype=np.float64)  # Convert labels to floating-point arrays for validation.
    W1 = np.asarray(W1, dtype=np.float64)  # Convert the first layer weights.
    b1 = np.asarray(b1, dtype=np.float64)  # Convert the first layer bias.
    W2 = np.asarray(W2, dtype=np.float64)  # Convert the second layer weights.
    b2 = np.asarray(b2, dtype=np.float64)  # Convert the second layer bias.

    if X.ndim != 2:  # Require a 2D input matrix.
        raise ValueError("X must have shape (N, D_in)")
    if np.any(np.array(X.shape) <= 0):  # Require positive dimensions.
        raise ValueError("X must have positive dimensions")
    if y.ndim != 1:  # Require a flat label vector.
        raise ValueError("y must have shape (N,)")
    if y.shape[0] != X.shape[0]:  # Require one label per example.
        raise ValueError("X and y must have the same batch size")
    if not np.all(np.isfinite(y)) or not np.allclose(y, np.round(y)):  # Require integer-like class ids.
        raise ValueError("y must contain integer class labels")
    y = np.round(y).astype(np.int64)  # Convert the validated labels to integers.

    input_dim = X.shape[1]  # Store the input feature width.
    if W1.ndim != 2 or W1.shape[0] != input_dim:  # Require W1 to match the input width.
        raise ValueError("W1 must have shape (D_in, H)")
    hidden_dim = W1.shape[1]  # Store the hidden width.
    if hidden_dim <= 0:  # Require a positive hidden size.
        raise ValueError("W1 must have positive dimensions")
    if b1.ndim != 1 or b1.shape[0] != hidden_dim:  # Require b1 to match the hidden width.
        raise ValueError("b1 must have shape (H,)")
    if W2.ndim != 2 or W2.shape[0] != hidden_dim:  # Require W2 to start from the hidden width.
        raise ValueError("W2 must have shape (H, C)")
    num_classes = W2.shape[1]  # Store the output class count.
    if num_classes <= 0:  # Require at least one output class.
        raise ValueError("W2 must have positive dimensions")
    if b2.ndim != 1 or b2.shape[0] != num_classes:  # Require b2 to match the class count.
        raise ValueError("b2 must have shape (C,)")
    if np.any((y < 0) | (y >= num_classes)):  # Ensure every label is in range.
        raise ValueError("y contains labels outside the valid range")

    z1 = X @ W1 + b1  # Compute the hidden pre-activations.
    h = np.maximum(z1, 0.0)  # Apply the ReLU nonlinearity.
    logits = h @ W2 + b2  # Compute the output logits.
    probs = _stable_softmax(logits)  # Turn logits into probabilities.

    batch_size = X.shape[0]  # Store the batch size for averaging.
    loss = -np.log(probs[np.arange(batch_size), y]).mean()  # Compute the mean softmax cross-entropy.

    dlogits = probs.copy()  # Start the gradient at the output probabilities.
    dlogits[np.arange(batch_size), y] -= 1.0  # Subtract one on the true class for each row.
    dlogits /= batch_size  # Average the gradient over the batch.

    dW2 = h.T @ dlogits  # Backpropagate into the second layer weights.
    db2 = dlogits.sum(axis=0)  # Sum the output gradient over the batch for the bias.
    dh = dlogits @ W2.T  # Backpropagate into the hidden activations.
    dz1 = dh * (z1 > 0)  # Apply the ReLU derivative to the hidden pre-activations.
    dW1 = X.T @ dz1  # Backpropagate into the first layer weights.
    db1 = dz1.sum(axis=0)  # Sum the hidden gradient over the batch for the bias.

    return {  # Return the loss and each parameter gradient in a dictionary.
        "loss": float(loss),
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2,
    }`,
    starterCode: `import numpy as np

def mlp_loss_and_grads(X, y, W1, b1, W2, b2):
    X = np.asarray(X)
    y = np.asarray(y)
    W1 = np.asarray(W1)
    b1 = np.asarray(b1)
    W2 = np.asarray(W2)
    b2 = np.asarray(b2)

    # TODO:
    # 1. Validate shapes and label ranges.
    # 2. Run the forward pass, compute the softmax cross-entropy loss, and backpropagate gradients.
    raise NotImplementedError("Implement mlp_loss_and_grads")

sample_X = np.array([[1.0, 2.0]])
sample_y = np.array([1])
sample_W1 = np.array([[1.0, 0.0], [0.0, 1.0]])
sample_b1 = np.array([0.0, 0.0])
sample_W2 = np.array([[1.0, 0.0], [0.0, 1.0]])
sample_b2 = np.array([0.0, 0.0])

print(mlp_loss_and_grads(sample_X, sample_y, sample_W1, sample_b1, sample_W2, sample_b2))`,
    packages: ['numpy'],
    tags: ['NumPy', 'Backpropagation', 'Neural Networks'],
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
    signature: `hidden_pre = X @ W1 + b1
hidden = relu(hidden_pre)
logits = hidden @ W2 + b2
loss = mean softmax cross-entropy(logits, y)`,
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
      'For softmax cross-entropy, the gradient with respect to the logits is `probs - one_hot(y)`, averaged over the batch.',
      'Backpropagate from the output layer into the hidden layer before multiplying by the ReLU mask.',
      'Return the gradients in a dictionary so the caller can inspect each parameter separately.',
    ],
    solutionNotes: [
      'The forward pass is affine, ReLU, affine, and mean softmax cross-entropy. Once you have the probabilities, the gradient of the loss with respect to the logits is the usual `probs - one_hot` term divided by the batch size.',
      'From there, the remaining gradients follow by the chain rule: the second affine layer gives `dW2` and `db2`, and the upstream gradient passes through the ReLU mask before producing `dW1` and `db1`.',
    ],
    solutionCode: `import numpy as np  # Import NumPy for the forward and backward pass.

def _stable_softmax(logits):  # Define a numerically stable softmax helper.
    logits = np.asarray(logits, dtype=np.float64)  # Convert logits to floating-point values.
    max_logits = np.max(logits, axis=-1, keepdims=True)  # Find the row-wise maximum.
    shifted = logits - max_logits  # Shift each row so the largest value becomes zero.
    exp_shifted = np.exp(shifted)  # Exponentiate the shifted logits.
    denom = np.sum(exp_shifted, axis=-1, keepdims=True)  # Sum the exponentials row by row.
    return np.divide(exp_shifted, denom, out=np.zeros_like(exp_shifted), where=denom > 0)  # Normalize safely.


def mlp_forward_backward(X, y, W1, b1, W2, b2):  # Define the classic MLP loss and gradient function.
    X = np.asarray(X, dtype=np.float64)  # Convert inputs to floating-point arrays.
    y = np.asarray(y, dtype=np.float64)  # Convert labels to floating-point arrays for validation.
    W1 = np.asarray(W1, dtype=np.float64)  # Convert the first layer weights.
    b1 = np.asarray(b1, dtype=np.float64)  # Convert the first layer bias.
    W2 = np.asarray(W2, dtype=np.float64)  # Convert the second layer weights.
    b2 = np.asarray(b2, dtype=np.float64)  # Convert the second layer bias.

    if X.ndim != 2:  # Require a 2D input matrix.
        raise ValueError("X must have shape (N, D_in)")
    if np.any(np.array(X.shape) <= 0):  # Require positive dimensions.
        raise ValueError("X must have positive dimensions")
    if y.ndim != 1:  # Require a flat label vector.
        raise ValueError("y must have shape (N,)")
    if y.shape[0] != X.shape[0]:  # Require one label per example.
        raise ValueError("X and y must have the same batch size")
    if not np.all(np.isfinite(y)) or not np.allclose(y, np.round(y)):  # Require integer-like class labels.
        raise ValueError("y must contain integer class labels")
    y = np.round(y).astype(np.int64)  # Convert the validated labels to integers.

    input_dim = X.shape[1]  # Store the input feature width.
    if W1.ndim != 2 or W1.shape[0] != input_dim:  # Require W1 to match the input width.
        raise ValueError("W1 must have shape (D_in, H)")
    hidden_dim = W1.shape[1]  # Store the hidden width.
    if hidden_dim <= 0:  # Require a positive hidden size.
        raise ValueError("W1 must have positive dimensions")
    if b1.ndim != 1 or b1.shape[0] != hidden_dim:  # Require b1 to match the hidden width.
        raise ValueError("b1 must have shape (H,)")
    if W2.ndim != 2 or W2.shape[0] != hidden_dim:  # Require W2 to match the hidden width.
        raise ValueError("W2 must have shape (H, C)")
    num_classes = W2.shape[1]  # Store the number of classes.
    if num_classes <= 0:  # Require at least one class.
        raise ValueError("W2 must have positive dimensions")
    if b2.ndim != 1 or b2.shape[0] != num_classes:  # Require b2 to match the class count.
        raise ValueError("b2 must have shape (C,)")
    if np.any((y < 0) | (y >= num_classes)):  # Ensure every label falls within the valid class range.
        raise ValueError("y contains labels outside the valid range")

    hidden_pre = X @ W1 + b1  # Compute the hidden pre-activation values.
    hidden = np.maximum(hidden_pre, 0.0)  # Apply ReLU to get the hidden activations.
    logits = hidden @ W2 + b2  # Compute the output logits.
    probs = _stable_softmax(logits)  # Turn logits into class probabilities.

    batch_size = X.shape[0]  # Store the batch size for averaging.
    loss = -np.log(probs[np.arange(batch_size), y]).mean()  # Compute the mean softmax cross-entropy.

    dlogits = probs.copy()  # Start the gradient from the softmax probabilities.
    dlogits[np.arange(batch_size), y] -= 1.0  # Subtract one on the true class for each sample.
    dlogits /= batch_size  # Average the output gradient over the batch.

    dW2 = hidden.T @ dlogits  # Backpropagate into the second layer weights.
    db2 = dlogits.sum(axis=0)  # Sum the output gradient over the batch for the bias.
    dhidden = dlogits @ W2.T  # Backpropagate into the hidden activations.
    dhidden_pre = dhidden * (hidden_pre > 0)  # Apply the ReLU derivative to the hidden pre-activations.
    dW1 = X.T @ dhidden_pre  # Backpropagate into the first layer weights.
    db1 = dhidden_pre.sum(axis=0)  # Sum the hidden gradient over the batch for the bias.

    return {  # Return the loss and gradients in a dictionary.
        "loss": float(loss),
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2,
    }`,
    starterCode: `import numpy as np

def mlp_forward_backward(X, y, W1, b1, W2, b2):
    X = np.asarray(X)
    y = np.asarray(y)
    W1 = np.asarray(W1)
    b1 = np.asarray(b1)
    W2 = np.asarray(W2)
    b2 = np.asarray(b2)

    # TODO:
    # 1. Validate shapes and label ranges.
    # 2. Run the forward pass, compute the softmax cross-entropy loss, and backpropagate gradients.
    raise NotImplementedError("Implement mlp_forward_backward")

sample_X = np.array([[1.0, 2.0]])
sample_y = np.array([1])
sample_W1 = np.array([[1.0, 0.0], [0.0, 1.0]])
sample_b1 = np.array([0.0, 0.0])
sample_W2 = np.array([[1.0, 0.0], [0.0, 1.0]])
sample_b2 = np.array([0.0, 0.0])

print(mlp_forward_backward(sample_X, sample_y, sample_W1, sample_b1, sample_W2, sample_b2))`,
    packages: ['numpy'],
    tags: ['NumPy', 'Backpropagation', 'Neural Networks'],
  },
] as const;
