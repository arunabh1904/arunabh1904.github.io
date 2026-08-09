export interface PytorchTensorExample {
  title: string;
  initialCode: string;
  referenceCode: string;
  description: string;
  notes?: string;
}

export const pytorchTensorExamples: Record<string, PytorchTensorExample> = {
  construction: {
    title: 'Construct, copy, and place a tensor',
    description: 'Edit the source value or dtype, then run the snippet again.',
    initialCode: `import numpy as np
import torch

source = np.array([[1, 2], [3, 4]], dtype=np.float32)
copied = torch.tensor(source, dtype=torch.float32, device="cpu")
shared = torch.as_tensor(source)

source[0, 0] = 99

print("copied:", copied.tolist())
print("shared:", shared.tolist())
print("dtype:", copied.dtype)
print("device:", copied.device)
`,
    referenceCode: `import numpy as np
import torch

source = np.array([[1, 2], [3, 4]], dtype=np.float32)

# torch.tensor always copies its input data.
copied = torch.tensor(source, dtype=torch.float32, device="cpu")

# These APIs can share CPU storage when the input permits it.
shared = torch.as_tensor(source)
shared_from_numpy = torch.from_numpy(source)

source[0, 0] = 99

print("copied:", copied.tolist())
print("as_tensor:", shared.tolist())
print("from_numpy:", shared_from_numpy.tolist())
print("dtype:", copied.dtype)
print("device:", copied.device)
`,
    notes:
      'The browser runner uses NumPy underneath, so its copy/share behavior mirrors the example but it does not allocate CUDA tensors.',
  },

  creationOps: {
    title: 'Create zeros, ones, and random tensors',
    description: 'The seed makes the random examples repeatable in this small exercise.',
    initialCode: `import torch

torch.manual_seed(7)

factories = {
    "zeros": torch.zeros((2, 3)),
    "ones": torch.ones((2, 3)),
    "full": torch.full((2, 3), 7),
    "rand": torch.rand((2, 3)),
    "randn": torch.randn((2, 3)),
    "arange": torch.arange(6).reshape(2, 3),
    "eye": torch.eye(3),
}

for name, value in factories.items():
    print(name, "shape:", value.shape)
    print(value)
`,
    referenceCode: `import torch

torch.manual_seed(7)

zeros = torch.zeros((2, 3), dtype=torch.float32)
ones = torch.ones((2, 3), dtype=torch.float32)
full = torch.full((2, 3), fill_value=7, dtype=torch.int64)
uniform = torch.rand((2, 3), dtype=torch.float32)       # [0, 1)
normal = torch.randn((2, 3), dtype=torch.float32)       # N(0, 1)
sequence = torch.arange(6, dtype=torch.int64).reshape(2, 3)
identity = torch.eye(3, dtype=torch.float32)
empty = torch.empty((2, 3))  # allocated, but not initialized

for name, value in {
    "zeros": zeros,
    "ones": ones,
    "full": full,
    "uniform": uniform,
    "normal": normal,
    "sequence": sequence,
    "identity": identity,
}.items():
    print(name, "shape:", value.shape)
    print(value)
`,
    notes: 'Try changing the factory shape. `empty` is useful when every element will be overwritten, but never read it before initialization.',
  },

  inspectStorage: {
    title: 'Inspect shape, scalar values, and storage views',
    description: 'Change the slice and observe the shape, stride, and storage relationship.',
    initialCode: `import torch

base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
view = base[:, 1:3]

print("shape:", base.shape)
print("ndim:", base.ndim)
print("numel:", base.numel())
print("element size:", base.element_size(), "bytes")
print("base stride:", base.stride())
print("view stride:", view.stride())
print("share storage:", base.untyped_storage().data_ptr() == view.untyped_storage().data_ptr())
print("one Python scalar:", base[1, 2].item())
print("many Python values:", view.tolist())
`,
    referenceCode: `import torch

base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
view = base[:, 1:3]

print("shape:", base.shape)
print("ndim:", base.ndim)
print("numel:", base.numel())
print("element size:", base.element_size(), "bytes")
print("base stride:", base.stride())
print("view stride:", view.stride())
print(
    "share storage:",
    base.untyped_storage().data_ptr() == view.untyped_storage().data_ptr(),
)

# item() is scalar conversion, not raw-storage access.
print("one Python scalar:", base[1, 2].item())
print("many Python values:", view.tolist())
`,
    notes:
      'A tensor is metadata plus a view over storage. Use `item()` for exactly one value, `tolist()` for a small tensor, and `untyped_storage().data_ptr()` only when you need to reason about aliasing.',
  },

  indexing: {
    title: 'Index and slice by dimension',
    description: 'Try negative indices, a different step, or a different boolean mask.',
    initialCode: `import torch

x = torch.arange(20).reshape(4, 5)
indices = torch.tensor([0, 3])

print("one element:", x[1, 2].item())
print("one row:\\n", x[1])
print("rows 0:2, columns 1:4:\\n", x[0:2, 1:4])
print("every other column:\\n", x[:, ::2])
print("advanced row selection:\\n", x[indices])

# Assignment through a slice changes x in-place.
x[:2, 0] = -1
print("after slice assignment:\\n", x)
`,
    referenceCode: `import torch

x = torch.arange(20).reshape(4, 5)
indices = torch.tensor([0, 3], dtype=torch.int64)

print("one element:", x[1, 2].item())
print("one row:\\n", x[1])
print("rows 0:2, columns 1:4:\\n", x[0:2, 1:4])
print("every other column:\\n", x[:, ::2])
print("advanced row selection:\\n", x[indices])

# Basic indexing/slicing returns a view. Assignment is in-place.
x[:2, 0] = -1
print("after slice assignment:\\n", x)
`,
    notes:
      'Basic slices such as `x[:, 1:4]` are views; advanced indexing such as `x[indices]` produces a copy. Assignment through either form mutates the destination tensor.',
  },

  joining: {
    title: 'Join and reshape tensors',
    description: 'Change `dim` and predict the resulting shape before running the code.',
    initialCode: `import torch

a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

cat_rows = torch.cat((a, b), dim=0)
cat_columns = torch.cat((a, b), dim=1)
stacked = torch.stack((a, b), dim=0)
reshaped = a.reshape(1, 4)

print("cat dim=0:", cat_rows.shape, "\\n", cat_rows)
print("cat dim=1:", cat_columns.shape, "\\n", cat_columns)
print("stack dim=0:", stacked.shape, "\\n", stacked)
print("reshape:", reshaped.shape, "\\n", reshaped)
`,
    referenceCode: `import torch

a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

# cat joins along an existing dimension.
cat_rows = torch.cat((a, b), dim=0)       # shape (4, 2)
cat_columns = torch.cat((a, b), dim=1)    # shape (2, 4)

# stack inserts a new dimension; inputs must have the same shape.
stacked = torch.stack((a, b), dim=0)      # shape (2, 2, 2)

# view shares data when the size/stride constraint is satisfied.
flat_view = a.view(4)
reshaped = a.reshape(1, 4)                # view or copy; do not rely on which

print("cat rows:", cat_rows.shape)
print("cat columns:", cat_columns.shape)
print("stacked:", stacked.shape)
print("flat view:", flat_view.shape)
print("reshaped:", reshaped.shape)
`,
    notes:
      '`cat` preserves rank and joins an existing dimension; `stack` increases rank by inserting a dimension. `reshape` is flexible, while `view` requires compatible strides.',
  },

  mutation: {
    title: 'Mutate explicitly and protect autograd',
    description: 'Run the example, then replace an in-place operation with an out-of-place one and compare the result.',
    initialCode: `import torch

x = torch.zeros((2, 3), dtype=torch.float32)

with torch.no_grad():
    x.add_(1.0)
    x[0, 1] = 7.0
    x[:, 2].fill_(4.0)

y = x + 1.0
print("x after mutation:\\n", x)
print("y from out-of-place +:\\n", y)
`,
    referenceCode: `import torch

x = torch.zeros((2, 3), dtype=torch.float32)

with torch.no_grad():
    x.add_(1.0)       # underscore marks an in-place operation
    x[0, 1] = 7.0     # indexing assignment is also in-place
    x[:, 2].fill_(4.0)

y = x + 1.0           # out-of-place: x is unchanged by this line
print("x after mutation:\\n", x)
print("y from out-of-place +:\\n", y)
`,
    notes:
      'In real training code, prefer out-of-place operations unless the memory saving is deliberate and autograd permits the mutation. Never use `.data` as a shortcut around autograd.',
  },
};
