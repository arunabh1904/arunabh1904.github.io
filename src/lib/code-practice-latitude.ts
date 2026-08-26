import type { CodePracticeProblem } from './code-practice';

const PYTORCH_AND_NUMPY_PACKAGES = ['torch', 'numpy'] as const;

export const LATITUDE_CODE_PRACTICE_PROBLEMS = [
  {
    id: 'multiple-linear-regression',
    order: 17,
    title: 'Multiple linear regression',
    difficulty: 'Medium',
    summary: 'Fit an ordinary least-squares model with an intercept and predict a test matrix.',
    prompt: [
      'Write `multiple_linear_regression(train_X, train_y, test_X)` for dense floating-point feature matrices.',
      'Add the intercept explicitly, solve least squares without forming a matrix inverse, and return one prediction per test row.',
    ],
    signature: `def multiple_linear_regression(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    test_X: torch.Tensor,
) -> torch.Tensor:
    ...`,
    requirements: [
      '`train_X` has shape `(N, F)`, `train_y` has shape `(N,)`, and `test_X` has shape `(T, F)`.',
      'Prepend an intercept column of ones to both feature matrices.',
      'Use a least-squares solver rather than explicitly inverting `X.T @ X`.',
      'Return predictions with shape `(T,)` and reject empty or incompatible inputs.',
    ],
    examples: [
      {
        label: 'Exact plane',
        lines: [
          'train_X = [[0, 0], [1, 0], [0, 1], [1, 1]]',
          'train_y = [1, 3, 4, 6]',
          'test_X = [[2, 3]]',
        ],
        result: '[14.0]',
      },
    ],
    hint: [
      'The intercept changes `(N, F)` into `(N, F + 1)`.',
      '`torch.linalg.lstsq(design, y[:, None]).solution` handles rank deficiency more safely than a normal-equation inverse.',
      'Use the same column order for the train and test design matrices.',
    ],
    solutionNotes: [
      'The intercept is a learned constant, so it needs its own feature column:\n`X: (N, F) -> [1, X]: (N, F + 1)`\nWithout that column, the fitted hyperplane is forced through the origin.',
      'Least squares solves:\n`min_w ||X_aug w - y||²`\n`lstsq` uses a factorization and can return a useful solution when columns are correlated. Explicitly forming `(X.T @ X)^-1` squares the condition number and can fail on singular data.',
      'Prediction preserves the feature contract:\n`test_aug: (T, F + 1) @ weights: (F + 1,) -> predictions: (T,)`\nTrain and test columns must use the same order.',
      'Correlated features make individual coefficients unstable even when predictions remain reasonable. Ridge regression changes the objective to:\n`||Xw - y||² + lambda ||w||²`\nUsually the intercept is excluded from that penalty.',
    ],
    solutionCode: `import torch

def multiple_linear_regression(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    test_X: torch.Tensor,
) -> torch.Tensor:
    train_X = torch.as_tensor(train_X, dtype=torch.float64)
    train_y = torch.as_tensor(train_y, dtype=torch.float64)
    test_X = torch.as_tensor(test_X, dtype=torch.float64)
    if train_X.ndim != 2 or test_X.ndim != 2 or train_y.ndim != 1:
        raise ValueError("expected train_X (N,F), train_y (N,), test_X (T,F)")
    if train_X.shape[0] == 0 or train_X.shape[0] != train_y.shape[0]:
        raise ValueError("training rows must be non-empty and match train_y")
    if train_X.shape[1] != test_X.shape[1]:
        raise ValueError("train and test feature counts must match")
    train_design = torch.cat((torch.ones((train_X.shape[0], 1)), train_X), dim=1)
    test_design = torch.cat((torch.ones((test_X.shape[0], 1)), test_X), dim=1)
    weights = torch.linalg.lstsq(train_design, train_y[:, None]).solution[:, 0]
    return test_design @ weights

train_X = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
train_y = torch.tensor([1.0, 3.0, 4.0, 6.0])
print(multiple_linear_regression(train_X, train_y, torch.tensor([[2.0, 3.0]])))`,
    starterCode: `import torch

def multiple_linear_regression(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    test_X: torch.Tensor,
) -> torch.Tensor:
    raise NotImplementedError("Implement multiple_linear_regression")

train_X = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
train_y = torch.tensor([1.0, 3.0, 4.0, 6.0])
print(multiple_linear_regression(train_X, train_y, torch.tensor([[2.0, 3.0]])))`,
    numpyAlternative: {
      code: `import numpy as np

def multiple_linear_regression(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
) -> np.ndarray:
    train_X = np.asarray(train_X, dtype=float)
    train_y = np.asarray(train_y, dtype=float)
    test_X = np.asarray(test_X, dtype=float)
    train_design = np.column_stack((np.ones(train_X.shape[0]), train_X))
    test_design = np.column_stack((np.ones(test_X.shape[0]), test_X))
    weights, *_ = np.linalg.lstsq(train_design, train_y, rcond=None)
    return test_design @ weights`,
      exampleCode: `train_X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
train_y = np.array([1, 3, 4, 6])
print(multiple_linear_regression(train_X, train_y, np.array([[2, 3]])))`,
      memory: ['Add the intercept first; use `np.linalg.lstsq`, not a normal-equation inverse.'],
    },
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'NumPy', 'Regression', 'Linear Algebra'],
  },
  {
    id: 'polynomial-regression-office-prices',
    order: 18,
    title: 'Polynomial regression: office prices',
    difficulty: 'Medium',
    summary: 'Generate every degree-three monomial, fit least squares, and predict unseen rows.',
    prompt: [
      'Write `polynomial_regression(train_X, train_y, test_X, degree=3)` for normalized office features.',
      'Include powers and cross-feature interactions whose total degree is at most three, then fit the expanded design with least squares.',
    ],
    signature: `def polynomial_regression(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    test_X: torch.Tensor,
    degree: int = 3,
) -> torch.Tensor:
    ...`,
    requirements: [
      'Generate the intercept and every monomial with total degree from one through `degree`.',
      'Include interactions such as `x1 * x2`, not only independent powers.',
      'Use the same expansion order for train and test matrices.',
      'Require `degree >= 1`, matching row counts, and matching feature counts.',
    ],
    examples: [
      {
        label: 'Quadratic interaction',
        lines: ['train_X = [[0,0], [1,0], [0,1], [1,1], [2,0], [0,2]]', 'train_y = [1, 2, 3, 6, 3, 5]', 'degree = 2'],
        result: 'prediction for [2, 3] is approximately 21',
      },
    ],
    hint: [
      '`combinations_with_replacement(range(F), d)` enumerates powers and interactions of total degree `d`.',
      'Multiply the selected columns for each combination to create one feature.',
      'Build train and test features with the same deterministic combinations.',
    ],
    solutionNotes: [
      'For two inputs and degree two, the design contains:\n`[1, x1, x2, x1², x1*x2, x2²]`\nThe interaction term matters because changing both features can have an effect that neither one-dimensional power represents.',
      'With `F` inputs and maximum degree `d`, the number of columns including the intercept is:\n`C(F + d, d)`\nThat combinatorial growth is the main memory and overfitting cost.',
      'The feature generator must be deterministic:\n`train_phi: (N, P), test_phi: (T, P)`\nIf column order changes between the two calls, learned coefficients multiply the wrong monomials.',
      'Normalization is useful when raw feature scales differ because high powers amplify scale gaps. Raising the degree lowers training bias but raises variance; choose degree and regularization with held-out data rather than the training residual.',
    ],
    solutionCode: `from itertools import combinations_with_replacement
import torch

def _polynomial_features(x: torch.Tensor, degree: int) -> torch.Tensor:
    columns = [torch.ones(x.shape[0], dtype=x.dtype)]
    for power in range(1, degree + 1):
        for indices in combinations_with_replacement(range(x.shape[1]), power):
            columns.append(torch.prod(x[:, indices], dim=1))
    return torch.stack(columns, dim=1)

def polynomial_regression(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    test_X: torch.Tensor,
    degree: int = 3,
) -> torch.Tensor:
    train_X = torch.as_tensor(train_X, dtype=torch.float64)
    train_y = torch.as_tensor(train_y, dtype=torch.float64)
    test_X = torch.as_tensor(test_X, dtype=torch.float64)
    if degree < 1 or train_X.ndim != 2 or test_X.ndim != 2 or train_y.ndim != 1:
        raise ValueError("expected degree >= 1 and matrix/vector inputs")
    if train_X.shape[0] != train_y.shape[0] or train_X.shape[1] != test_X.shape[1]:
        raise ValueError("row and feature counts must match")
    train_phi = _polynomial_features(train_X, degree)
    test_phi = _polynomial_features(test_X, degree)
    weights = torch.linalg.lstsq(train_phi, train_y[:, None]).solution[:, 0]
    return test_phi @ weights

X = torch.tensor([[0.,0.], [1.,0.], [0.,1.], [1.,1.], [2.,0.], [0.,2.]])
print(polynomial_regression(X, torch.tensor([1.,2.,3.,6.,3.,5.]), torch.tensor([[2.,3.]]), 2))`,
    starterCode: `import torch

def polynomial_regression(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    test_X: torch.Tensor,
    degree: int = 3,
) -> torch.Tensor:
    raise NotImplementedError("Implement polynomial_regression")

X = torch.tensor([[0.,0.], [1.,0.], [0.,1.], [1.,1.], [2.,0.], [0.,2.]])
print(polynomial_regression(X, torch.tensor([1.,2.,3.,6.,3.,5.]), torch.tensor([[2.,3.]]), 2))`,
    numpyAlternative: {
      code: `from itertools import combinations_with_replacement
import numpy as np

def polynomial_regression(train_X: np.ndarray, train_y: np.ndarray,
                          test_X: np.ndarray, degree: int = 3) -> np.ndarray:
    train_X, test_X = np.asarray(train_X, float), np.asarray(test_X, float)
    terms = [()] + [term for d in range(1, degree + 1)
                    for term in combinations_with_replacement(range(train_X.shape[1]), d)]
    def expand(x):
        return np.column_stack([np.ones(x.shape[0]) if not term
                                else np.prod(x[:, term], axis=1) for term in terms])
    weights, *_ = np.linalg.lstsq(expand(train_X), np.asarray(train_y, float), rcond=None)
    return expand(test_X) @ weights`,
      exampleCode: `X = np.array([[0,0], [1,0], [0,1], [1,1], [2,0], [0,2]], dtype=float)
y = np.array([1, 2, 3, 6, 3, 5], dtype=float)
print(polynomial_regression(X, y, np.array([[2, 3]]), degree=2))`,
      memory: ['Enumerate combinations with replacement once, then reuse that monomial order.'],
    },
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'NumPy', 'Regression', 'Feature Engineering'],
  },
  {
    id: 'basic-statistics-warmup',
    order: 4,
    title: 'Basic statistics warmup',
    difficulty: 'Easy',
    summary: 'Compute deterministic summary statistics and a normal-approximation confidence interval.',
    prompt: [
      'Write `basic_statistics(values)` to return the mean, median, smallest mode, population standard deviation, and 95% confidence interval for a non-empty one-dimensional tensor.',
      'Use `1.96` for the normal-approximation interval and define the mode tie-break explicitly.',
    ],
    signature: `def basic_statistics(values: torch.Tensor) -> dict[str, float]:
    ...`,
    requirements: [
      'Use the numerically smallest value when several modes have the same count.',
      'Use population variance: divide the squared-error sum by `N`, not `N - 1`.',
      'Return the confidence interval `mean +/- 1.96 * population_std / sqrt(N)`.',
      'Raise `ValueError` for empty or non-vector input.',
    ],
    examples: [
      {
        label: 'Tied mode',
        lines: ['values = [1, 1, 2, 2, 4]'],
        result: 'mean=2.0, median=2.0, mode=1.0, population_std≈1.0954',
      },
    ],
    hint: [
      'Sort once for the median; `torch.unique(..., return_counts=True)` gives deterministic mode candidates.',
      'The two middle indices for even `N` are `N // 2 - 1` and `N // 2`.',
      'Compute the standard error only after the population standard deviation.',
    ],
    solutionNotes: [
      'Mean and population standard deviation use all `N` observations:\n`mean = sum(x) / N`\n`population_std = sqrt(sum((x - mean)²) / N)`\nSample standard deviation instead divides by `N - 1` to estimate an unseen population.',
      'The median needs sorted order, but the mean does not. Sorting costs `O(N log N)`; counting modes is expected `O(N)` with a hash map, while `torch.unique` may sort internally.',
      'Mode is a contract, not just a count:\n`mode = smallest value among the maximum-count values`\nThat rule makes ties deterministic. Empty input has no mean or mode, so this implementation raises instead of returning NaNs.',
      'The approximate interval is:\n`mean +/- 1.96 * population_std / sqrt(N)`\nIt assumes independent samples and uses the provided normal approximation. Production work may use a t-interval or bootstrap when those assumptions do not fit.',
    ],
    solutionCode: `import torch

def basic_statistics(values: torch.Tensor) -> dict[str, float]:
    values = torch.as_tensor(values, dtype=torch.float64)
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("values must be a non-empty vector")
    ordered = torch.sort(values).values
    count = values.numel()
    middle = count // 2
    median = ordered[middle] if count % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    unique, counts = torch.unique(values, sorted=True, return_counts=True)
    mode = unique[torch.argmax(counts)]
    mean = torch.mean(values)
    population_std = torch.sqrt(torch.mean((values - mean) ** 2))
    margin = 1.96 * population_std / torch.sqrt(torch.tensor(float(count)))
    return {
        "mean": float(mean.item()), "median": float(median.item()),
        "mode": float(mode.item()), "population_std": float(population_std.item()),
        "ci_lower": float((mean - margin).item()), "ci_upper": float((mean + margin).item()),
    }

print(basic_statistics(torch.tensor([1, 1, 2, 2, 4])))`,
    starterCode: `import torch

def basic_statistics(values: torch.Tensor) -> dict[str, float]:
    raise NotImplementedError("Implement basic_statistics")

print(basic_statistics(torch.tensor([1, 1, 2, 2, 4])))`,
    numpyAlternative: {
      code: `import numpy as np

def basic_statistics(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a non-empty vector")
    ordered = np.sort(values)
    unique, counts = np.unique(values, return_counts=True)
    mean = np.mean(values)
    std = np.sqrt(np.mean((values - mean) ** 2))
    margin = 1.96 * std / np.sqrt(values.size)
    return {"mean": mean, "median": np.median(ordered),
            "mode": unique[np.argmax(counts)], "population_std": std,
            "ci_lower": mean - margin, "ci_upper": mean + margin}`,
      exampleCode: `print(basic_statistics(np.array([1, 1, 2, 2, 4])))`,
      memory: ['Sort for the median, choose the smallest tied mode, and divide population variance by `N`.'],
    },
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'NumPy', 'Statistics'],
  },
  {
    id: 'best-aptitude-test',
    order: 19,
    title: 'The best aptitude test',
    difficulty: 'Medium',
    summary: 'Choose the test whose ranking has the strongest absolute relationship with GPA.',
    prompt: [
      'Write `best_aptitude_test(gpa, test_scores, method)` for five candidate test-score rows and one GPA vector.',
      'Support Pearson correlation on raw values and Spearman correlation on average-tie ranks. Return the one-based test number, preferring the smaller number on a tie.',
    ],
    signature: `def best_aptitude_test(
    gpa: torch.Tensor,
    test_scores: torch.Tensor,
    method: str = "spearman",
) -> int:
    ...`,
    requirements: [
      '`gpa` has shape `(N,)` and `test_scores` has shape `(5, N)`.',
      'Use the absolute correlation magnitude as predictive strength.',
      'Average the occupied rank positions when values tie under Spearman correlation.',
      'Return a one-based index and reject constant vectors whose correlation is undefined.',
    ],
    examples: [
      {
        label: 'Official-style ranking',
        lines: ['gpa = [7.5, 7.7, 7.9, 8.1, 8.3]', 'test 1 = [10, 30, 20, 40, 50]', 'four other rows are less aligned'],
        result: '1',
      },
    ],
    hint: [
      'Pearson correlation is cosine similarity after centering both vectors.',
      'Spearman is Pearson applied to rank vectors.',
      '`argmax` returns the first maximum, which supplies the smaller-index tie-break.',
    ],
    solutionNotes: [
      'Pearson measures linear agreement after centering:\n`r = sum((x-x_bar)(y-y_bar)) / sqrt(sum((x-x_bar)²) sum((y-y_bar)²))`\nIt is sensitive to outliers and to nonlinear scale changes.',
      'Spearman applies the same formula to ranks:\n`rho = Pearson(rank(x), rank(y))`\nIt stays high for nonlinear but monotonic relationships. Ties need average ranks; two arbitrary ranks would make the answer depend on sort order.',
      'This implementation scores `abs(correlation)`. A strong negative relation is still predictive if the downstream rule can invert it. If the domain requires higher aptitude to imply higher GPA, compare signed positive correlations instead.',
      'A constant test row has zero centered norm, so its correlation is undefined rather than zero. Rejecting it keeps a data-quality problem visible. In noisier settings, compare held-out predictive error instead of selecting on one batch correlation.',
    ],
    solutionCode: `import torch

def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values, stable=True)
    ordered = values[order]
    ranks = torch.empty(values.shape, dtype=torch.float64)
    start = 0
    while start < values.numel():
        end = start + 1
        while end < values.numel() and ordered[end] == ordered[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks

def _correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = x - torch.mean(x), y - torch.mean(y)
    denominator = torch.sqrt(torch.sum(x * x) * torch.sum(y * y))
    if denominator == 0:
        raise ValueError("correlation is undefined for a constant vector")
    return torch.sum(x * y) / denominator

def best_aptitude_test(gpa: torch.Tensor, test_scores: torch.Tensor, method: str = "spearman") -> int:
    gpa = torch.as_tensor(gpa, dtype=torch.float64)
    test_scores = torch.as_tensor(test_scores, dtype=torch.float64)
    if gpa.ndim != 1 or test_scores.shape != (5, gpa.numel()) or gpa.numel() < 2:
        raise ValueError("expected gpa (N,) and test_scores (5,N), with N >= 2")
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be pearson or spearman")
    target = _average_ranks(gpa) if method == "spearman" else gpa
    candidates = [_average_ranks(row) if method == "spearman" else row for row in test_scores]
    strengths = torch.stack([torch.abs(_correlation(row, target)) for row in candidates])
    return int(torch.argmax(strengths).item()) + 1

gpa = torch.tensor([7.5, 7.7, 7.9, 8.1, 8.3])
tests = torch.tensor([[10,30,20,40,50], [11,9,5,19,29], [21,9,15,19,39], [91,9,75,19,89], [81,99,55,59,89]])
print(best_aptitude_test(gpa, tests))`,
    starterCode: `import torch

def best_aptitude_test(
    gpa: torch.Tensor,
    test_scores: torch.Tensor,
    method: str = "spearman",
) -> int:
    raise NotImplementedError("Implement best_aptitude_test")

gpa = torch.tensor([7.5, 7.7, 7.9, 8.1, 8.3])
tests = torch.tensor([[10,30,20,40,50], [11,9,5,19,29], [21,9,15,19,39], [91,9,75,19,89], [81,99,55,59,89]])
print(best_aptitude_test(gpa, tests))`,
    numpyAlternative: {
      code: `import numpy as np

def _ranks(x):
    order = np.argsort(x, kind="stable")
    ranks, start = np.empty(x.size, float), 0
    while start < x.size:
        end = start + 1
        while end < x.size and x[order[end]] == x[order[start]]: end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks

def best_aptitude_test(gpa: np.ndarray, scores: np.ndarray,
                       method: str = "spearman") -> int:
    gpa, scores = np.asarray(gpa, float), np.asarray(scores, float)
    target = _ranks(gpa) if method == "spearman" else gpa
    rows = [_ranks(row) if method == "spearman" else row for row in scores]
    strength = [abs(np.corrcoef(row, target)[0, 1]) for row in rows]
    return int(np.argmax(strength)) + 1`,
      exampleCode: `gpa = np.array([7.5, 7.7, 7.9, 8.1, 8.3])
scores = np.array([[10,30,20,40,50], [11,9,5,19,29], [21,9,15,19,39],
                   [91,9,75,19,89], [81,99,55,59,89]])
print(best_aptitude_test(gpa, scores))`,
      memory: ['Spearman is Pearson on average-tie ranks; return `argmax(abs(r)) + 1`.'],
    },
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'NumPy', 'Statistics', 'Ranking'],
  },
  {
    id: 'laptop-battery-life',
    order: 20,
    title: 'Laptop battery life',
    difficulty: 'Easy',
    summary: 'Encode a data-derived linear relationship that saturates at the battery capacity.',
    prompt: [
      'Write `predict_battery_life(charge_hours)` after inspecting a training log where runtime grows by two hours per charge hour and then saturates at eight hours.',
      'Support a scalar or tensor of non-negative charge durations and preserve its shape.',
    ],
    signature: `def predict_battery_life(charge_hours: torch.Tensor) -> torch.Tensor:
    ...`,
    requirements: [
      'Use the piecewise relationship `min(2 * charge_hours, 8)`.',
      'Return zero for zero charge and reject negative or non-finite inputs.',
      'Preserve scalar or batched input shape.',
      'Do not fit a global line through the saturated observations.',
    ],
    examples: [
      { label: 'Before saturation', lines: ['charge_hours = 1.5'], result: '3.0' },
      { label: 'After saturation', lines: ['charge_hours = 7.0'], result: '8.0' },
    ],
    hint: [
      'Inspect the scatter before choosing a model.',
      '`torch.clamp(2 * charge_hours, max=8.0)` expresses both branches.',
    ],
    solutionNotes: [
      'The log reveals two regimes:\n`battery_hours = 2 * charge_hours, charge_hours < 4`\n`battery_hours = 8, charge_hours >= 4`\nThe breakpoint follows from where the rising line first reaches capacity.',
      'A single least-squares line averages the rising and flat regions, so it underpredicts near the knee and can exceed the physical capacity. The piecewise model encodes the visible mechanism directly.',
      'In a production version, estimate slope, breakpoint, and plateau from training data, then test them on held-out sessions. Residual plots make saturation easier to see than one aggregate regression score.',
    ],
    solutionCode: `import torch

def predict_battery_life(charge_hours: torch.Tensor) -> torch.Tensor:
    charge_hours = torch.as_tensor(charge_hours, dtype=torch.float64)
    if bool(torch.any(~torch.isfinite(charge_hours))) or bool(torch.any(charge_hours < 0)):
        raise ValueError("charge_hours must be finite and non-negative")
    return torch.clamp(2.0 * charge_hours, max=8.0)

print(predict_battery_life(torch.tensor([1.5, 7.0])))`,
    starterCode: `import torch

def predict_battery_life(charge_hours: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("Implement predict_battery_life")

print(predict_battery_life(torch.tensor([1.5, 7.0])))`,
    numpyAlternative: {
      code: `import numpy as np

def predict_battery_life(charge_hours: np.ndarray) -> np.ndarray:
    charge_hours = np.asarray(charge_hours, dtype=float)
    if np.any(~np.isfinite(charge_hours)) or np.any(charge_hours < 0):
        raise ValueError("charge_hours must be finite and non-negative")
    return np.minimum(2.0 * charge_hours, 8.0)`,
      exampleCode: `print(predict_battery_life(np.array([1.5, 7.0])))`,
      memory: ['Inspect the curve first: it is `2x` until four charge hours, then it caps at eight.'],
    },
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'NumPy', 'Regression', 'Piecewise Models'],
  },
  {
    id: 'masked-scaled-dot-product-attention',
    order: 37,
    title: 'Stable masked scaled dot-product attention',
    difficulty: 'Medium',
    summary: 'Implement the attention primitive with a stable softmax and defined all-masked rows.',
    prompt: [
      'Write `scaled_dot_product_attention(q, k, v, mask)` for query length `T` and source length `S`.',
      'Apply a broadcastable validity mask before softmax, avoid token loops, and return zero weights and output for an entirely masked query row.',
    ],
    signature: `def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...`,
    requirements: [
      '`q`, `k`, and `v` have shapes `(B, T, D)`, `(B, S, D)`, and `(B, S, Dv)`.',
      'Compute scores as `q @ k.transpose(-2, -1) / sqrt(D)`.',
      'Subtract the finite row maximum before exponentiating.',
      'Treat `True` as valid and explicitly zero all-masked rows.',
      'Return output `(B, T, Dv)` and weights `(B, T, S)`.',
    ],
    examples: [
      {
        label: 'One masked key',
        lines: ['q = [[[1, 0]]]', 'k = [[[1, 0], [0, 1]]]', 'v = [[[5], [9]]]', 'mask = [[[True, False]]]'],
        result: 'output=[[[5]]], weights=[[[1, 0]]]',
      },
    ],
    hint: [
      'Keep a boolean `valid` tensor broadcast to `(B, T, S)`.',
      'Replace invalid logits before the maximum, then multiply exponentials by the mask.',
      'Use a safe denominator of one only for rows whose valid exponential sum is zero.',
    ],
    solutionNotes: [
      'The score and output shapes are:\n`q: (B,T,D), k.T: (B,D,S) -> scores: (B,T,S)`\n`weights: (B,T,S) @ v: (B,S,Dv) -> output: (B,T,Dv)`',
      'Scaling uses:\n`scores = QK.T / sqrt(D)`\nIf independent query and key components have unit variance, their dot product variance grows with `D`. Scaling keeps logits in a range where softmax gradients do not immediately saturate.',
      'Stable softmax subtracts one row maximum before `exp`. A padding mask hides absent tokens; a causal mask hides future positions. They can be combined with boolean `and` when `True` consistently means valid.',
      'An all-masked row has no probability distribution. Replacing every logit with negative infinity leads to `-inf - -inf` and NaNs, so this implementation uses zero exponentials and a safe denominator, then returns an explicit zero row.',
      'Multi-head attention adds a head axis:\n`q: (B,H,T,Dh), k: (B,H,S,Dh), weights: (B,H,T,S)`\nScore memory is `O(BHTS)`; long sequences often need tiled or fused attention to avoid materializing the full matrix.',
    ],
    solutionCode: `from __future__ import annotations
import torch

def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    q, k, v = (torch.as_tensor(x, dtype=torch.float64) for x in (q, k, v))
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, and v must be rank-three tensors")
    if q.shape[0] != k.shape[0] or k.shape[:2] != v.shape[:2] or q.shape[2] != k.shape[2]:
        raise ValueError("batch, source, and feature dimensions must align")
    scores = q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5)
    valid = torch.ones_like(scores, dtype=torch.bool) if mask is None else torch.broadcast_to(
        torch.as_tensor(mask, dtype=torch.bool), scores.shape)
    safe_scores = torch.where(valid, scores, torch.zeros_like(scores))
    row_max = torch.amax(safe_scores, dim=-1, keepdim=True)
    exponentials = torch.exp(safe_scores - row_max) * valid
    denominator = torch.sum(exponentials, dim=-1, keepdim=True)
    weights = exponentials / torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    return weights @ v, weights

q = torch.tensor([[[1.0, 0.0]]])
k = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
v = torch.tensor([[[5.0], [9.0]]])
print(scaled_dot_product_attention(q, k, v, torch.tensor([[[True, False]]])))`,
    starterCode: `from __future__ import annotations
import torch

def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError("Implement scaled_dot_product_attention")

q = torch.tensor([[[1.0, 0.0]]])
k = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
v = torch.tensor([[[5.0], [9.0]]])
print(scaled_dot_product_attention(q, k, v, torch.tensor([[[True, False]]])))`,
    numpyAlternative: {
      code: `import numpy as np

def scaled_dot_product_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                                 mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    q, k, v = np.asarray(q, float), np.asarray(k, float), np.asarray(v, float)
    scores = q @ k.swapaxes(-1, -2) / np.sqrt(q.shape[-1])
    valid = np.ones(scores.shape, bool) if mask is None else np.broadcast_to(mask, scores.shape)
    safe = np.where(valid, scores, 0.0)
    exp_scores = np.exp(safe - np.max(safe, axis=-1, keepdims=True)) * valid
    total = np.sum(exp_scores, axis=-1, keepdims=True)
    weights = np.divide(exp_scores, total, out=np.zeros_like(exp_scores), where=total > 0)
    return weights @ v, weights`,
      exampleCode: `q = np.array([[[1., 0.]]])
k = np.array([[[1., 0.], [0., 1.]]])
v = np.array([[[5.], [9.]]])
print(scaled_dot_product_attention(q, k, v, np.array([[[True, False]]])))`,
      memory: ['Mask before softmax, subtract the row maximum, and divide only where a row has valid mass.'],
    },
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'NumPy', 'Attention', 'Numerical Stability'],
  },
  {
    id: 'cross-entropy-and-multiclass-metrics',
    order: 6,
    title: 'Stable cross-entropy and multiclass metrics',
    difficulty: 'Medium',
    summary: 'Handle ignored labels, stable logits, a confusion matrix, and macro versus micro scores.',
    prompt: [
      'Implement `cross_entropy_with_logits` and `classification_metrics` for integer class labels, including an `ignore_index`.',
      'Keep cross-entropy stable, define zero-denominator metrics as zero, and make the empty-valid-batch behavior explicit.',
    ],
    signature: `def cross_entropy_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -1,
) -> torch.Tensor:
    ...

def classification_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_index: int = -1,
) -> dict[str, torch.Tensor | float]:
    ...`,
    requirements: [
      'Use log-sum-exp rather than `log(softmax(logits))`.',
      'Build a `(C, C)` confusion matrix with true classes on rows and predicted classes on columns.',
      'Return per-class precision and recall, macro F1, micro F1, and accuracy.',
      'Exclude ignored examples; raise for an empty cross-entropy batch and return zero metrics for an empty metrics batch.',
    ],
    examples: [
      {
        label: 'Ignored example and missing class',
        lines: ['predictions = [0, 2, 1, 1]', 'labels = [0, 1, 1, -1]', 'num_classes = 3'],
        result: 'accuracy=0.66667, macro_f1=0.55556',
      },
    ],
    hint: [
      'Filter `ignore_index` before validating label ranges.',
      'Encode each confusion entry as `true_class * C + predicted_class`, then use `bincount`.',
      'Compute precision from columns and recall from rows.',
    ],
    solutionNotes: [
      'Stable cross-entropy for row `i` is:\n`loss_i = m_i + log(sum(exp(z_i - m_i))) - z_i[label_i]`\nSubtracting `m_i = max(z_i)` prevents overflow without changing the probability ratios.',
      'The confusion layout is:\n`confusion[true_class, predicted_class] += 1`\nRows therefore supply ground-truth counts for recall; columns supply prediction counts for precision.',
      'Macro F1 gives every class equal weight, including rare classes. Micro F1 aggregates counts first; for single-label multiclass classification it equals accuracy. Accuracy can hide failure on rare classes when the majority class dominates.',
      'A zero prediction denominator means no item was predicted as that class; a zero truth denominator means the class is absent. This implementation returns zero for both. It returns zero metrics for no valid labels, but raises for empty loss because a mean over zero examples is undefined.',
      'Logits are unnormalized scores. Probabilities come from softmax, while predicted classes come from `argmax(logits)`. Cross-entropy averages over valid examples or tokens only; ignored positions must not dilute the denominator.',
    ],
    solutionCode: `import torch

def cross_entropy_with_logits(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
    logits = torch.as_tensor(logits, dtype=torch.float64)
    labels = torch.as_tensor(labels, dtype=torch.long)
    valid = labels != ignore_index
    logits, labels = logits[valid], labels[valid]
    if logits.ndim != 2 or labels.numel() == 0 or logits.shape[0] != labels.shape[0]:
        raise ValueError("expected a non-empty valid batch of logits (N,C) and labels (N,)")
    if bool(torch.any((labels < 0) | (labels >= logits.shape[1]))):
        raise ValueError("labels must be valid class ids")
    maximum = torch.amax(logits, dim=1, keepdim=True)
    normalizer = maximum[:, 0] + torch.log(torch.sum(torch.exp(logits - maximum), dim=1))
    return torch.mean(normalizer - logits[torch.arange(labels.numel()), labels])

def classification_metrics(predictions: torch.Tensor, labels: torch.Tensor, num_classes: int,
                           ignore_index: int = -1) -> dict[str, torch.Tensor | float]:
    predictions, labels = torch.as_tensor(predictions, dtype=torch.long), torch.as_tensor(labels, dtype=torch.long)
    valid = labels != ignore_index
    predictions, labels = predictions[valid], labels[valid]
    if predictions.shape != labels.shape or num_classes <= 0:
        raise ValueError("predictions and labels must match and num_classes must be positive")
    if labels.numel() and bool(torch.any((predictions < 0) | (predictions >= num_classes) | (labels < 0) | (labels >= num_classes))):
        raise ValueError("class ids are out of range")
    flat = labels * num_classes + predictions
    confusion = torch.bincount(flat, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    true_positive = torch.tensor([confusion[i, i] for i in range(num_classes)], dtype=torch.float64)
    predicted_count = torch.sum(confusion, dim=0)
    truth_count = torch.sum(confusion, dim=1)
    precision = torch.where(predicted_count > 0, true_positive / torch.clamp(predicted_count, min=1), 0.0)
    recall = torch.where(truth_count > 0, true_positive / torch.clamp(truth_count, min=1), 0.0)
    f1 = torch.where(precision + recall > 0, 2 * precision * recall / torch.clamp(precision + recall, min=1e-12), 0.0)
    correct, total = torch.sum(true_positive), labels.numel()
    score = float((correct / total).item()) if total else 0.0
    return {"confusion_matrix": confusion, "precision": precision, "recall": recall,
            "macro_f1": float(torch.mean(f1).item()), "micro_f1": score, "accuracy": score}

print(classification_metrics(torch.tensor([0,2,1,1]), torch.tensor([0,1,1,-1]), 3))`,
    starterCode: `import torch

def cross_entropy_with_logits(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
    raise NotImplementedError("Implement cross_entropy_with_logits")

def classification_metrics(predictions: torch.Tensor, labels: torch.Tensor, num_classes: int,
                           ignore_index: int = -1) -> dict[str, torch.Tensor | float]:
    raise NotImplementedError("Implement classification_metrics")

print(classification_metrics(torch.tensor([0,2,1,1]), torch.tensor([0,1,1,-1]), 3))`,
    numpyAlternative: {
      code: `import numpy as np

def cross_entropy_with_logits(logits: np.ndarray, labels: np.ndarray, ignore_index: int = -1) -> float:
    logits, labels = np.asarray(logits, float), np.asarray(labels, int)
    valid = labels != ignore_index
    logits, labels = logits[valid], labels[valid]
    maximum = np.max(logits, axis=1, keepdims=True)
    normalizer = maximum[:, 0] + np.log(np.sum(np.exp(logits - maximum), axis=1))
    return np.mean(normalizer - logits[np.arange(labels.size), labels])

def classification_metrics(predictions: np.ndarray, labels: np.ndarray,
                           num_classes: int, ignore_index: int = -1) -> dict:
    predictions, labels = np.asarray(predictions, int), np.asarray(labels, int)
    valid = labels != ignore_index
    predictions, labels = predictions[valid], labels[valid]
    matrix = np.bincount(labels * num_classes + predictions,
                         minlength=num_classes ** 2).reshape(num_classes, num_classes)
    tp = np.diag(matrix).astype(float)
    precision = np.divide(tp, matrix.sum(0), out=np.zeros_like(tp), where=matrix.sum(0) > 0)
    recall = np.divide(tp, matrix.sum(1), out=np.zeros_like(tp), where=matrix.sum(1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=precision + recall > 0)
    accuracy = tp.sum() / labels.size if labels.size else 0.0
    return {"confusion_matrix": matrix, "precision": precision, "recall": recall,
            "macro_f1": f1.mean(), "micro_f1": accuracy, "accuracy": accuracy}`,
      exampleCode: `pred = np.array([0, 2, 1, 1])
labels = np.array([0, 1, 1, -1])
print(classification_metrics(pred, labels, 3))`,
      memory: ['Flatten `(true, pred)` to `true * C + pred`; rows drive recall and columns drive precision.'],
    },
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'NumPy', 'Classification', 'Metrics'],
  },
  {
    id: 'smooth-l1-loss-and-gradient',
    order: 13,
    title: 'Smooth L1 loss and gradient',
    difficulty: 'Medium',
    summary: 'Implement the beta-scaled piecewise loss and its mean-reduction gradient.',
    prompt: [
      'Write `smooth_l1_loss_and_grad(prediction, target, beta)` for the Smooth L1 definition used by PyTorch.',
      'Return the mean loss and the derivative with respect to `prediction`, including the reduction scale.',
    ],
    signature: `def smooth_l1_loss_and_grad(
    prediction: torch.Tensor,
    target: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...`,
    requirements: [
      'Use `r**2 / (2 * beta)` when `|r| < beta` and `|r| - beta / 2` otherwise.',
      'Use derivative `r / beta` inside the quadratic branch and `sign(r)` outside it.',
      'Return a scalar mean loss and a gradient tensor matching `prediction`.',
      'Require matching non-empty shapes and `beta > 0`.',
    ],
    examples: [
      {
        label: 'Mixed branches',
        lines: ['prediction = [0.5, 3.0]', 'target = [0.0, 0.0]', 'beta = 1.0'],
        result: 'loss=1.3125, grad=[0.25, 0.5]',
      },
    ],
    hint: [
      'Compute elementwise loss and derivative branches before selecting with `torch.where`.',
      'Divide the selected derivative by `prediction.numel()` because the loss uses a mean.',
    ],
    solutionNotes: [
      'For residual `r = prediction - target`, Smooth L1 is:\n`r² / (2 beta), |r| < beta`\n`|r| - beta / 2, otherwise`\nThe branches meet with value `beta/2` and slope magnitude one.',
      'Its elementwise derivative is:\n`r / beta, |r| < beta`\n`sign(r), otherwise`\nAt `r = 0`, the quadratic branch gives gradient zero. Pure L1 instead needs a chosen subgradient at zero, commonly zero.',
      'Smooth L1 and Huber share a shape but differ by scale:\n`SmoothL1(r, beta) = Huber(r, delta=beta) / beta`\nThat distinction changes both loss and gradient magnitude when `beta` is not one.',
      'As `beta -> 0`, the quadratic region shrinks and Smooth L1 approaches absolute error. Because the result is a mean, every elementwise derivative is divided by the number of residuals; a sum reduction would omit that factor.',
    ],
    solutionCode: `import torch

def smooth_l1_loss_and_grad(
    prediction: torch.Tensor,
    target: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = torch.as_tensor(prediction, dtype=torch.float64)
    target = torch.as_tensor(target, dtype=torch.float64)
    if prediction.shape != target.shape or prediction.numel() == 0 or beta <= 0:
        raise ValueError("inputs must have matching non-empty shapes and beta must be positive")
    residual = prediction - target
    magnitude = torch.abs(residual)
    loss = torch.where(magnitude < beta, residual ** 2 / (2 * beta), magnitude - beta / 2)
    gradient = torch.where(magnitude < beta, residual / beta, torch.sign(residual))
    return torch.mean(loss), gradient / prediction.numel()

loss, gradient = smooth_l1_loss_and_grad(torch.tensor([0.5, 3.0]), torch.zeros(2))
print(loss, gradient)`,
    starterCode: `import torch

def smooth_l1_loss_and_grad(
    prediction: torch.Tensor,
    target: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError("Implement smooth_l1_loss_and_grad")

loss, gradient = smooth_l1_loss_and_grad(torch.tensor([0.5, 3.0]), torch.zeros(2))
print(loss, gradient)`,
    numpyAlternative: {
      code: `import numpy as np

def smooth_l1_loss_and_grad(prediction: np.ndarray, target: np.ndarray,
                            beta: float = 1.0) -> tuple[float, np.ndarray]:
    prediction, target = np.asarray(prediction, float), np.asarray(target, float)
    residual = prediction - target
    magnitude = np.abs(residual)
    loss = np.where(magnitude < beta, residual ** 2 / (2 * beta), magnitude - beta / 2)
    gradient = np.where(magnitude < beta, residual / beta, np.sign(residual))
    return np.mean(loss), gradient / prediction.size`,
      exampleCode: `loss, grad = smooth_l1_loss_and_grad(np.array([0.5, 3.0]), np.zeros(2))
print(loss, grad)`,
      memory: ['Differentiate each branch, then divide by the element count for mean reduction.'],
    },
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'NumPy', 'Regression', 'Losses', 'Gradients'],
  },
  {
    id: 'sparse-scatter-mean',
    order: 26,
    title: 'Sparse BEV scatter mean',
    difficulty: 'Medium',
    summary: 'Aggregate point features into grid cells without looping over the cells.',
    prompt: [
      'Write `scatter_mean(features, cell_ids, num_cells)` for sparse point or voxel features.',
      'Accumulate sums and counts by cell, return zeros for empty cells, and reject invalid IDs.',
    ],
    signature: `def scatter_mean(
    features: torch.Tensor,
    cell_ids: torch.Tensor,
    num_cells: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...`,
    requirements: [
      '`features` has shape `(N, C)` and `cell_ids` has shape `(N,)` with values in `[0, num_cells)`.',
      'Return `aggregated` with shape `(num_cells, C)` and `counts` with shape `(num_cells,)`.',
      'Do not loop over cells; use indexed accumulation.',
      'Leave empty-cell feature rows at zero and preserve the feature dtype.',
    ],
    examples: [
      {
        label: 'Repeated and empty cells',
        lines: ['features = [[1, 3], [3, 5], [8, 2]]', 'cell_ids = [0, 0, 2]', 'num_cells = 4'],
        result: 'aggregated=[[2,4], [0,0], [8,2], [0,0]], counts=[2,0,1,0]',
      },
    ],
    hint: [
      '`index_add_(0, cell_ids, features)` accumulates every point row into its destination cell.',
      'Accumulate a vector of ones with the same IDs to obtain counts.',
      'Clamp the divisor to one, then explicitly keep empty rows at zero.',
    ],
    solutionNotes: [
      'Indexed accumulation maps sparse rows into a dense grid:\n`features: (N,C) -> sums: (num_cells,C)`\n`cell_ids: (N,) -> counts: (num_cells,)`\nRepeated IDs add into the same destination row.',
      'Mean reduction is:\n`mean[cell] = sum[cell] / count[cell]`\nEmpty cells have count zero, so use a safe divisor and then keep their output rows at zero. The work is `O(NC)` and dense output memory is `O(num_cells*C)`.',
      'A scatter maximum also needs the argmax point per channel. Batched IDs can be flattened before one accumulation:\n`global_id = batch_id * num_cells + cell_id`\nThen reshape the result back into batch and cell axes.',
      'Invalid IDs can be rejected, as here, or filtered under an explicit ignore policy. GPU indexed reductions can be nondeterministic when several threads update one cell; deterministic training may require sorted segments or a deterministic backend path.',
      'Dense output is convenient for convolution but expensive as grid resolution grows. Sparse coordinates plus occupied-cell features avoid allocating empty cells and are often the better representation before a dense BEV stage is required.',
    ],
    solutionCode: `import torch

def scatter_mean(
    features: torch.Tensor,
    cell_ids: torch.Tensor,
    num_cells: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.as_tensor(features)
    cell_ids = torch.as_tensor(cell_ids, dtype=torch.long)
    if features.ndim != 2 or cell_ids.ndim != 1 or features.shape[0] != cell_ids.shape[0]:
        raise ValueError("expected features (N,C) and cell_ids (N,)")
    if num_cells <= 0 or bool(torch.any((cell_ids < 0) | (cell_ids >= num_cells))):
        raise ValueError("cell ids must lie in [0, num_cells)")
    sums = torch.zeros((num_cells, features.shape[1]), dtype=features.dtype)
    counts = torch.zeros(num_cells, dtype=torch.long)
    sums.index_add_(0, cell_ids, features)
    counts.index_add_(0, cell_ids, torch.ones(cell_ids.shape[0], dtype=torch.long))
    divisor = torch.clamp(counts, min=1)[:, None]
    return sums / divisor, counts

features = torch.tensor([[1., 3.], [3., 5.], [8., 2.]])
print(scatter_mean(features, torch.tensor([0, 0, 2]), 4))`,
    starterCode: `import torch

def scatter_mean(
    features: torch.Tensor,
    cell_ids: torch.Tensor,
    num_cells: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError("Implement scatter_mean")

features = torch.tensor([[1., 3.], [3., 5.], [8., 2.]])
print(scatter_mean(features, torch.tensor([0, 0, 2]), 4))`,
    numpyAlternative: {
      code: `import numpy as np

def scatter_mean(features: np.ndarray, cell_ids: np.ndarray,
                 num_cells: int) -> tuple[np.ndarray, np.ndarray]:
    features, cell_ids = np.asarray(features), np.asarray(cell_ids, dtype=int)
    if np.any((cell_ids < 0) | (cell_ids >= num_cells)):
        raise ValueError("cell ids must lie in [0, num_cells)")
    sums = np.zeros((num_cells, features.shape[1]), dtype=features.dtype)
    counts = np.zeros(num_cells, dtype=int)
    np.add.at(sums, cell_ids, features)
    np.add.at(counts, cell_ids, 1)
    return np.divide(sums, counts[:, None], out=np.zeros_like(sums),
                     where=counts[:, None] > 0), counts`,
      exampleCode: `features = np.array([[1., 3.], [3., 5.], [8., 2.]])
print(scatter_mean(features, np.array([0, 0, 2]), 4))`,
      memory: ['Use `np.add.at` for repeated IDs; ordinary advanced-index `+=` can lose repeated updates.'],
    },
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'NumPy', 'Autonomous Driving', 'Scatter', 'BEV'],
  },
  {
    id: 'rotate-image-quarter-turns',
    order: 9,
    title: 'Rotate an image by quarter turns',
    difficulty: 'Easy',
    summary: 'Rotate the last two image axes clockwise without interpolation or pixel loops.',
    prompt: [
      'Write `rotate_image(image, quarter_turns)` for a two-dimensional image or a tensor with arbitrary leading batch and channel axes.',
      'Positive turns rotate clockwise. Preserve dtype and rotate only the final height and width axes.',
    ],
    signature: `def rotate_image(image: torch.Tensor, quarter_turns: int = 1) -> torch.Tensor:
    ...`,
    requirements: [
      'Accept rank-two or higher input shaped `(..., H, W)`.',
      'Normalize negative or large turn counts modulo four.',
      'Use transpose and flip, not loops over pixels.',
      'A 90-degree turn swaps `H` and `W`; a 180-degree turn does not.',
    ],
    examples: [
      {
        label: 'One clockwise turn',
        lines: ['image = [[1, 2, 3], [4, 5, 6]]'],
        result: '[[4, 1], [5, 2], [6, 3]]',
      },
    ],
    hint: [
      'One clockwise turn is `transpose(H, W)` followed by a flip along the new width axis.',
      '`quarter_turns % 4` handles negative and repeated rotations.',
    ],
    solutionNotes: [
      'One clockwise turn changes the layout as:\n`(..., H, W) -> transpose -> (..., W, H) -> flip last axis`\nThe operation reindexes pixels; it does not interpolate values.',
      'A non-square image swaps its last two sizes on odd turns. Batch and channel axes are leading axes, so the same code handles `(H,W)`, `(C,H,W)`, and `(B,C,H,W)`.',
      'Transpose can return a non-contiguous view, while flip commonly materializes reversed data. Call `contiguous()` only when a downstream kernel requires it; an unconditional copy adds cost without changing the result.',
      'Arbitrary-angle rotation is a different problem. It needs an inverse coordinate map, an output-size policy, interpolation such as bilinear sampling, and a boundary-fill rule. Quarter turns need none of those choices.',
    ],
    solutionCode: `import torch

def rotate_image(image: torch.Tensor, quarter_turns: int = 1) -> torch.Tensor:
    image = torch.as_tensor(image)
    if image.ndim < 2:
        raise ValueError("image must have at least height and width axes")
    turns = quarter_turns % 4
    for _ in range(turns):
        image = torch.flip(image.transpose(-2, -1), dims=(-1,))
    return image

print(rotate_image(torch.tensor([[1, 2, 3], [4, 5, 6]])))`,
    starterCode: `import torch

def rotate_image(image: torch.Tensor, quarter_turns: int = 1) -> torch.Tensor:
    raise NotImplementedError("Implement rotate_image")

print(rotate_image(torch.tensor([[1, 2, 3], [4, 5, 6]])))`,
    numpyAlternative: {
      code: `import numpy as np

def rotate_image(image: np.ndarray, quarter_turns: int = 1) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim < 2:
        raise ValueError("image must have at least height and width axes")
    return np.rot90(image, k=-(quarter_turns % 4), axes=(-2, -1))`,
      exampleCode: `print(rotate_image(np.array([[1, 2, 3], [4, 5, 6]])))`,
      memory: ['Clockwise is negative `np.rot90` direction; rotate only axes `(-2, -1)`.'],
    },
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'NumPy', 'Images', 'Tensor Shapes'],
  },
  {
    id: 'reflect-points-across-line',
    order: 10,
    title: 'Reflect points across a line',
    difficulty: 'Medium',
    summary: 'Reflect a batch of 2D points across an implicit line with one vectorized projection.',
    prompt: [
      'Write `reflect_points(points, line)` for points shaped `(N, 2)` and a line `a*x + b*y + c = 0`.',
      'Return the mirror points without looping over rows and reject a line with no valid normal vector.',
    ],
    signature: `def reflect_points(points: torch.Tensor, line: torch.Tensor) -> torch.Tensor:
    ...`,
    requirements: [
      '`points` has shape `(N, 2)` and `line` has shape `(3,)` containing `[a, b, c]`.',
      'Reflect every point with the same vectorized formula.',
      'Do not require `[a, b]` to be normalized.',
      'Raise `ValueError` when both `a` and `b` are zero.',
    ],
    examples: [
      { label: 'Reflect across y = x', lines: ['points = [[2, 1], [-1, 3]]', 'line = [1, -1, 0]'], result: '[[1, 2], [3, -1]]' },
    ],
    hint: [
      'The signed numerator is `a*x + b*y + c`; divide it by `a**2 + b**2`.',
      'Move twice the normal projection to reach the mirror point.',
      'Broadcast one `(N, 1)` distance factor against the normal `(2,)`.',
    ],
    solutionNotes: [
      'For line normal `n = [a,b]`, the signed projection factor is:\n`d = (points @ n + c) / (n @ n)`\nThe denominator makes the formula independent of how the line coefficients are scaled.',
      'Reflection moves through the line by twice the normal component:\n`reflected = points - 2 * d[:,None] * n[None,:]`\nHere `(N,1) * (1,2)` broadcasts to one displacement vector per point.',
      'Horizontal and vertical lines need no special cases. For `y = k`, use `[a,b,c] = [0,1,-k]`; for `x = k`, use `[1,0,-k]`. A line through the origin simply has `c = 0`.',
      'Reflecting twice should recover the original points within floating-point tolerance, and points on the line should remain fixed. Those two invariants catch sign and factor-of-two errors better than one example.',
      'The 3D extension reflects points across a plane `n dot x + d = 0` with the same formula and a three-component normal. A line reflection in 3D is different: it needs projection onto a direction axis.',
    ],
    solutionCode: `import torch

def reflect_points(points: torch.Tensor, line: torch.Tensor) -> torch.Tensor:
    points = torch.as_tensor(points, dtype=torch.float64)
    line = torch.as_tensor(line, dtype=torch.float64)
    if points.ndim != 2 or points.shape[1] != 2 or line.shape != (3,):
        raise ValueError("expected points (N,2) and line (3,)")
    normal, offset = line[:2], line[2]
    denominator = torch.sum(normal * normal)
    if denominator == 0:
        raise ValueError("line normal cannot be zero")
    projection = (points @ normal + offset) / denominator
    return points - 2.0 * projection[:, None] * normal[None, :]

print(reflect_points(torch.tensor([[2., 1.], [-1., 3.]]), torch.tensor([1., -1., 0.])))`,
    starterCode: `import torch

def reflect_points(points: torch.Tensor, line: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("Implement reflect_points")

print(reflect_points(torch.tensor([[2., 1.], [-1., 3.]]), torch.tensor([1., -1., 0.])))`,
    numpyAlternative: {
      code: `import numpy as np

def reflect_points(points: np.ndarray, line: np.ndarray) -> np.ndarray:
    points, line = np.asarray(points, float), np.asarray(line, float)
    normal, offset = line[:2], line[2]
    denominator = normal @ normal
    if denominator == 0:
        raise ValueError("line normal cannot be zero")
    projection = (points @ normal + offset) / denominator
    return points - 2 * projection[:, None] * normal[None, :]`,
      exampleCode: `points = np.array([[2., 1.], [-1., 3.]])
print(reflect_points(points, np.array([1., -1., 0.])))`,
      memory: ['Subtract twice the signed normal projection; `(N,1)` broadcasts against `(1,2)`.'],
    },
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'NumPy', 'Geometry', 'Broadcasting'],
  },
] satisfies readonly CodePracticeProblem[];
