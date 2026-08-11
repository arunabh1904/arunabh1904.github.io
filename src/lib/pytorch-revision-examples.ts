export interface PytorchRevisionExample {
  title: string;
  initialCode: string;
}

export const pytorchRevisionExamples: Record<string, PytorchRevisionExample> = {
  tensorIntuition: {
    title: 'Trace one tensor from storage to model input',
    initialCode: `import torch

base = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
tokens = base[:, :, ::2]
bias = torch.tensor([10.0, -10.0])
shifted = tokens + bias
features = shifted.reshape(6, 2).clone()

print("base     ", base.shape, base.stride())
print("tokens   ", tokens.shape, tokens.stride())
print("shifted  ", shifted.shape)
print("features ", features.shape, features.is_contiguous())

tokens[0, 0, 0] = -99
print("view mutation reached base:", base[0, 0, 0].item())
print("clone stayed independent:  ", features[0, 0].item())
`,
  },

  gradientIntuition: {
    title: 'Trace a loss backward through a linear model',
    initialCode: `import torch

x = torch.tensor([[-1.0], [0.0], [1.0], [2.0]])
target = 3 * x + 2
weight = torch.tensor([[0.0]])
bias = torch.tensor([0.0])
learning_rate = 0.1

prediction = torch.matmul(x, weight) + bias
residual = prediction - target
loss = torch.mean(residual ** 2)

grad_prediction = 2 * residual / x.shape[0]
grad_weight = torch.matmul(torch.transpose(x, 0, 1), grad_prediction)
grad_bias = torch.sum(grad_prediction, dim=0)

with torch.no_grad():
    weight -= learning_rate * grad_weight
    bias -= learning_rate * grad_bias

print("loss:", loss.item())
print("grad weight / bias:", grad_weight.item(), grad_bias.item())
print("updated weight / bias:", weight.item(), bias.item())
`,
  },

  systemsIntuition: {
    title: 'Budget memory before choosing an execution strategy',
    initialCode: `batch = 4
tokens = 2048
width = 4096
layers = 32
heads = 32
bytes_per_value = 2
world_size = 8
parameter_count = 7_000_000_000

gib = 1024 ** 3
parameter_gib = parameter_count * bytes_per_value / gib
activation_gib = batch * tokens * width * layers * bytes_per_value / gib
attention_gib = batch * heads * tokens * tokens * bytes_per_value / gib
sharded_parameter_gib = parameter_gib / world_size

print(f"parameters per replica: {parameter_gib:.2f} GiB")
print(f"parameters if evenly sharded: {sharded_parameter_gib:.2f} GiB")
print(f"one activation-sized term: {activation_gib:.2f} GiB")
print(f"one dense attention-score term: {attention_gib:.2f} GiB")

print("\\nDouble tokens and rerun: the activation term doubles,")
print("while the dense attention-score term grows by four.")
`,
  },

  construction: {
    title: 'Construction and NumPy interop',
    initialCode: `import numpy as np
import torch

source = np.array([[1, 2], [3, 4]], dtype=np.float32)

copied = torch.tensor(source, dtype=torch.float32, device="cpu")
shared = torch.as_tensor(source)
from_numpy = torch.from_numpy(source)

source[0, 0] = 99

print("torch.tensor copied:", copied.tolist())
print("as_tensor shares:", shared.tolist())
print("from_numpy shares:", from_numpy.tolist())
print("dtype / device:", copied.dtype, copied.device)
`,
  },

  factories: {
    title: 'Factories, dtype, device, and randomness',
    initialCode: `import numpy as np
import torch

torch.manual_seed(7)

items = {
    "zeros": torch.zeros(2, 3, dtype=torch.float32),
    "ones_like": torch.ones_like(torch.zeros(2, 3)),
    "full": torch.full((2, 3), 7, dtype=torch.int64),
    "rand": torch.rand(2, 3),
    "randn": torch.randn(2, 3),
    "randint": torch.randint(0, 10, (2, 3)),
    "arange": torch.arange(6).reshape(2, 3),
    "linspace": torch.linspace(0, 1, 5),
    "eye": torch.eye(3),
}

for name, value in items.items():
    print(f"{name:10s}", value.shape, value.dtype)

print("NumPy default float:", np.zeros((1,)).dtype)
print("Torch default float:", torch.zeros(1).dtype)
`,
  },

  views: {
    title: 'Storage, strides, views, and mutation',
    initialCode: `import torch

base = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
view = base[:, :, ::2]
transposed = torch.permute(base, (0, 2, 1))

print("base:", base.shape, base.stride(), base.is_contiguous())
print("view:", view.shape, view.stride(), view.is_contiguous())
print("permute:", transposed.shape, transposed.stride(), transposed.is_contiguous())
print("same storage:", base.untyped_storage().data_ptr() == view.untyped_storage().data_ptr())

view[0, 0, 0] = -1
print("mutation reached base:", base[0, 0, 0].item())
print("scalar / list:", base[0, 0, 1].item(), view[0].tolist())
`,
  },

  indexing: {
    title: 'Indexing, masks, gather, and scatter',
    initialCode: `import torch

x = torch.arange(20).reshape(4, 5)
rows = torch.tensor([0, 3], dtype=torch.int64)
gather_index = torch.tensor([[0, 2], [1, 4], [0, 3], [2, 2]])

basic = x[1:3, ::2]
advanced = x[rows]
gathered = torch.gather(x, dim=1, index=gather_index)
mask = x % 3 == 0

print("basic view:\\n", basic)
print("advanced copy:\\n", advanced)
print("gathered:\\n", gathered)
print("masked values:", x[mask].tolist())

x[:2, 0] = -1
print("slice assignment:\\n", x)
`,
  },

  shapes: {
    title: 'Shape transforms and broadcasting',
    initialCode: `import numpy as np
import torch

x = torch.arange(24).reshape(2, 3, 4)
bias = torch.arange(4)

print("broadcast result:", (x + bias).shape)
print("unsqueeze:", torch.unsqueeze(x, 1).shape)
print("squeeze:", torch.squeeze(torch.unsqueeze(x, 1), 1).shape)
print("flatten middle:", torch.flatten(x, 1, 2).shape)
print("transpose:", torch.transpose(x, 1, 2).shape)
print("permute:", torch.permute(x, (2, 0, 1)).shape)

a = torch.ones(2, 3)
b = torch.zeros(2, 3)
print("cat:", torch.cat((a, b), dim=0).shape)
print("stack:", torch.stack((a, b), dim=0).shape)
print("NumPy agrees:", np.stack((a, b), axis=0).shape)
`,
  },

  linalg: {
    title: 'Matmul, einsum, and linear algebra',
    initialCode: `import numpy as np
import torch

batch, tokens, width, out = 2, 3, 4, 5
x = torch.arange(batch * tokens * width, dtype=torch.float32).reshape(batch, tokens, width)
w = torch.arange(width * out, dtype=torch.float32).reshape(width, out)

matmul = torch.matmul(x, w)
einsum = torch.einsum("btd,do->bto", x, w)

print("matmul:", matmul.shape)
print("einsum:", einsum.shape)
print("same values:", torch.allclose(matmul, einsum))
print("dot:", torch.dot(torch.arange(4), torch.arange(4)).item())
print("outer:\\n", torch.outer(torch.arange(3), torch.arange(2)))
print("NumPy matmul agrees:", np.allclose(matmul, np.matmul(x, w)))
`,
  },

  reductions: {
    title: 'Reductions and numerical stability',
    initialCode: `import torch

logits = torch.tensor([[1000.0, 1001.0, 999.0], [-1000.0, -999.0, -1001.0]])

probabilities = torch.softmax(logits, dim=-1)
log_probabilities = torch.log_softmax(logits, dim=-1)
normalizer = torch.logsumexp(logits, dim=-1)
top = torch.topk(logits, k=2, dim=-1)

print("softmax row sums:", torch.sum(probabilities, dim=-1))
print("logsumexp:", normalizer)
print("top values:\\n", top.values)
print("top indices:\\n", top.indices)
print("L2 norm:", torch.norm(logits, p=2, dim=-1))
`,
  },

  autograd: {
    title: 'Autograd: backward, accumulation, and gradient modes',
    initialCode: `import torch

scalar = torch.tensor(3.0, requires_grad=True)
scalar_output = scalar ** 2
scalar_output.backward()
print("d(scalar ** 2) / dscalar:", scalar.grad)

vector = torch.tensor([1.0, 2.0], requires_grad=True)
vector_output = vector ** 2
vector_output.backward(torch.ones_like(vector_output))
print("vector-Jacobian product:", vector.grad)

vector.grad.zero_()
second_output = (vector ** 2).sum()
second_output.backward()
print("after clearing and backward:", vector.grad)

with torch.no_grad():
    vector -= 0.1 * vector.grad
print("updated without recording the update:", vector)
`,
  },

  modules: {
    title: 'Module registration and buffers',
    initialCode: `import torch
from torch import nn

torch.manual_seed(3)

class Projector(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.proj = nn.Linear(width, width)
        self.scale = nn.Parameter(torch.ones(width))
        self.register_buffer("running_total", torch.zeros(width))
        self.unregistered = [nn.Linear(width, width)]

    def forward(self, x):
        return self.proj(x) * self.scale

model = Projector(4)
x = torch.ones(2, 4)

print("output shape:", model(x).shape)
print("parameters:", [name for name, _ in model.named_parameters()])
print("buffers:", [name for name, _ in model.named_buffers()])
print("children:", [name for name, _ in model.named_children()])
print("state dict:", list(model.state_dict()))
`,
  },

  containers: {
    title: 'Module containers, mode, and state',
    initialCode: `import torch
from torch import nn

torch.manual_seed(5)

class MLP(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        for layer in self.layers:
            x = nn.functional.relu(layer(x))
        return self.dropout(x)

model = MLP(width=4, depth=3)
inputs = torch.ones(2, 4)

model.train()
train_output = model(inputs)
model.eval()
eval_output = model(inputs)

checkpoint = model.state_dict()
result = model.load_state_dict(checkpoint, strict=True)

print("registered layers:", [name for name, _ in model.named_modules()])
print("state keys:", list(checkpoint))
print("mode:", model.training, model.dropout.training)
print("train/eval shapes:", train_output.shape, eval_output.shape)
print("load result:", result.missing_keys, result.unexpected_keys)
`,
  },

  data: {
    title: 'Dataset, DataLoader, and collation',
    initialCode: `import torch
from torch.utils.data import DataLoader, Dataset

class Pairs(Dataset):
    def __init__(self):
        self.features = torch.arange(30, dtype=torch.float32).reshape(10, 3)
        self.targets = torch.arange(10, dtype=torch.int64)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return {"features": self.features[index], "target": self.targets[index]}

torch.manual_seed(11)
loader = DataLoader(Pairs(), batch_size=4, shuffle=True, drop_last=False)

for batch_index, batch in enumerate(loader):
    print(batch_index, batch["features"].shape, batch["target"].tolist())
`,
  },

  training: {
    title: 'Training-loop invariants',
    initialCode: `import torch

torch.manual_seed(13)
x = torch.linspace(-1, 1, 16).reshape(16, 1)
target = 3 * x + 2

weight = torch.randn(1, 1)
bias = torch.zeros(1)
learning_rate = 0.2

for step in range(12):
    prediction = torch.matmul(x, weight) + bias
    residual = prediction - target
    loss = torch.mean(residual ** 2)

    grad_weight = 2 * torch.matmul(torch.transpose(x, 0, 1), residual) / x.shape[0]
    grad_bias = 2 * torch.mean(residual, dim=0)

    with torch.no_grad():
        weight.sub_(learning_rate * grad_weight)
        bias.sub_(learning_rate * grad_bias)

print("loss:", float(loss))
print("weight / bias:", weight.item(), bias.item())
`,
  },

  optimization: {
    title: 'Accumulation, AMP, clipping, and schedulers',
    initialCode: `import torch

# Loss scaling keeps tiny float16 gradients representable.
gradient = torch.tensor([2e-8, -7e-8, 4e-5], dtype=torch.float16)
scale = 1024.0
scaled = gradient * scale
unscaled = scaled / scale

print("original float16:", gradient)
print("scaled:", scaled)
print("unscaled:", unscaled)

# Accumulating four microbatches requires dividing each loss by four.
microbatch_losses = torch.tensor([1.2, 0.8, 1.0, 1.4])
effective_loss = torch.sum(microbatch_losses / 4)
print("effective mean loss:", effective_loss.item())
`,
  },

  functionalTransforms: {
    title: 'Vectorization and per-sample gradients',
    initialCode: `import torch

def loss(weight, example, target):
    prediction = torch.sum(weight * example)
    return (prediction - target) ** 2

weight = torch.tensor([0.3, -0.2, 0.8], dtype=torch.float64)
examples = torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, -1.0]])
targets = torch.tensor([1.0, -2.0])

epsilon = 1e-6
per_sample = []
for example, target in zip(examples, targets):
    gradient = torch.zeros_like(weight)
    for index in range(weight.shape[0]):
        step = torch.zeros_like(weight)
        step[index] = epsilon
        gradient[index] = (
            loss(weight + step, example, target)
            - loss(weight - step, example, target)
        ) / (2 * epsilon)
    per_sample.append(gradient)

print("per-sample gradient shape:", torch.stack(per_sample).shape)
print(torch.stack(per_sample))
`,
  },
};
