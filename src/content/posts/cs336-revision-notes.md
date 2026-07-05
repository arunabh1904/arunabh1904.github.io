---
title: Lecture 2
date: '2025-05-15T04:00:00.000Z'
section: revision-notes
postSlug: cs336-revision-notes
legacyPath: /revision notes/2025/05/15/cs336-revision-notes.html
tags:
  - Other
summary: CS336 Lecture 2 — Tensor Fundamentals & Computation Efficiency
---
# CS336 Lecture 2 — Tensor Fundamentals & Computation Efficiency

*Revision notes updated May 24 2025*

---

## Quick Overview

This lecture covers dtype tradeoffs, tensor memory layout, Einops reshaping, computational-efficiency metrics, and initialization strategies. The throughline is practical: understand what the hardware is actually moving and computing.

---

## 1 · Environment set-up & timing

```bash
!pip install -q uv
!uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# \u23f1\ufe0f add wall-clock timing
import time, subprocess, sys
start = time.perf_counter()
# \u2192 assume the two shell commands above just executed
print(f"Install runtime: {time.perf_counter()-start:.2f} s")
```

```text
Using Python 3.11.12 environment at: /usr
Audited 3 packages in 100 ms
Install runtime: 2.37 s
```

We use `uv` because it is a Rust-based drop-in replacement for `pip` that resolves, builds, and installs wheels in parallel — typically **4–6× faster** than pip on large scientific stacks.

---

## 2 · CUDA check & dtype benchmark

```python
import torch, time
print("CUDA available" if torch.cuda.is_available() else "CUDA not available. Defaulting to CPU.")

device = "cuda" if torch.cuda.is_available() else "cpu"

DTYPES = {
    "float64": torch.float64,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    # "float8_e4m3fn": torch.float8_e4m3fn,   # Hopper-class GPUs only
}

N = 2_000_000
for name, dt in DTYPES.items():
    x = torch.randn(N, device=device, dtype=dt)
    if x.is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter(); y = x.square().sqrt()
    if x.is_cuda:
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1e3
    rel = (torch.norm(y - x) / torch.norm(x)).item()
    print(f"{name:<11} | {x.element_size()} B | {ms:6.2f} ms | rel-err {rel:.2e}")
```

```text
float64     | 8 B |  56.08 ms | rel-err 1.41e+00
float32     | 4 B |  18.83 ms | rel-err 1.41e+00
bfloat16    | 2 B |  11.55 ms | rel-err 1.41e+00
float16     | 2 B |  36.21 ms | rel-err 1.41e+00
```

Training is usually **bandwidth-bound**. Halving element size improves the compute-to-memory-traffic ratio and can move kernels onto tensor cores. `float64` is the high-precision variant typically used for CPU-heavy numerical work. `float32` remains the default for many deep-learning training paths. `float16` and `bfloat16` are both half-precision formats, but `bfloat16` keeps the 8-bit exponent of FP32, so it usually avoids loss scaling while `float16` often needs it. `FP8 (E4M3/E5M2)` is still mostly an inference-focused format here because backpropagation stresses its tiny dynamic range. Blackwell GPUs also introduce `FP4` variants.

---

## 3 · Autograd inspector

```python
import torch, inspect, time

t     = torch.randn(2, 3, 4, device=device, requires_grad=True)
act   = torch.relu(t)                                 # non-linear
proj  = torch.randn(4, 3, device=device=device)       # weight matrix
z     = torch.einsum("bcd,dc->bc", t, proj)          # contraction
loss  = z.mean()

loss.backward()

for n in [t, act, z, loss]:
    print(f"{tuple(n.shape)} | grad_fn={type(n.grad_fn).__name__ if n.grad_fn else None}")
```

```text
(2, 3, 4) | grad_fn=None
(2, 3, 4) | grad_fn=ReluBackward0
(2, 3)    | grad_fn=ViewBackward0
()        | grad_fn=MeanBackward0
```

The `grad_fn` field lets you inspect the dynamic computation graph **without** external tools, which is useful when gradients look wrong.

---

## 4 · Memory layout, views, and hidden copies

```python
# Build a 4\u00d74 tensor 0\u202f..\u202f15, then examine a transposed view.

a      = torch.arange(16).reshape(4, 4)
view   = a.t()          # view: same storage, different strides
clone  = view.clone()   # force deep copy

print(f"contiguous? view={view.is_contiguous()}  clone={clone.is_contiguous()}")
share_ptr = a.storage().data_ptr() == view.storage().data_ptr()
print(f"share underlying storage? {share_ptr}")
```

```text
contiguous? view=False  clone=False
share underlying storage? True
```

**Strides rule everything.** A tensor is `(data_ptr, sizes, strides)`. Transpose costs 0 B until a kernel needs contiguous memory; then PyTorch silently allocates a fresh buffer. Keep one consistent layout, such as `[B, Seq, Heads, Dim]`, through the pipeline.

---

## 5 · Clean Tensor Manipulations with Einops

Einops gives tensor reshaping, permutation, and reduction a compact syntax. The practical benefit is readability: it is much harder to hide a dimension-ordering bug inside a named pattern than inside manual indexing.

```python
from einops import rearrange, reduce, repeat

x = torch.randn(2, 224, 224, 3)

x_nchw = rearrange(x, "b h w c -> b c h w")
seq = rearrange(x_nchw, "b c h w -> b (h w) c")
x_reconstructed = rearrange(seq, "b (h w) c -> b c h w", h=224)
gap = reduce(x_nchw, "b c h w -> b c", "mean")
emb_map = repeat(torch.randn(2, 64), "b d -> b d h w", h=7, w=7)
```

When strides allow it, Einops changes only metadata. That gives you safer reshaping without unnecessary memory traffic.

---

## 6 · Tracking computational efficiency

FLOPs describe the arithmetic work. Model FLOP Utilization (MFU) compares the FLOPs a model actually achieves to the hardware's theoretical maximum. MFU is usually below 1 because memory latency, kernel launch overhead, and imperfect hardware utilization all get in the way.

$$
MFU = \frac{\text{achieved FLOPs}}{\text{theoretical FLOPs}}
$$

```python
from math import prod

def gemm_flops(m:int, n:int, k:int):
    return 2*m*n*k

def conv2d_flops(c_in:int, c_out:int, k:int, h:int, w:int):
    return 2*c_in*c_out*(k**2)*h*w

def attn_flops(seq:int, dim:int):
    return 4*seq*seq*dim + seq*seq

```

| Kernel         | Rule-of-thumb FLOPs               |
| -------------- | --------------------------------- |
| GEMM / Linear  | `2·m·n·k`                         |
| Conv2d         | `2·C_in·C_out·K²·H_out·W_out`     |
| Self-Attention | `4·S²·D`                          |

Bias, ReLU, and LayerNorm are often less than 1% of an LLM's FLOPs, but they can still dominate latency when launch overhead or memory stalls bite.

---

## 7 · Glorot & He initialisation in practice

```python
import torch, math

def fan(t):
    return t.size(1), t.size(0)

def glorot(t):
    fi, fo = fan(t); u = math.sqrt(6/(fi + fo))
    return torch.nn.init.uniform_(t, -u, u)

def kaiming(t):
    fi, _ = fan(t); std = math.sqrt(2/fi)
    return torch.nn.init.normal_(t, 0, std)

w1 = torch.empty(256, 512); glorot(w1)
w2 = torch.empty(256, 512); kaiming(w2)

print("Glorot std", w1.std().item())
print("Kaiming std", w2.std().item())
```

Sample output

```
Glorot std 0.0782
Kaiming std 0.1104
```

Glorot/Xavier Initialisation (Glorot & Bengio, 2010) assumes symmetric activations like `tanh` and balances variance between the forward and backward passes.

He Initialisation (He et al., 2015) is tailored for ReLU-style activations, which zero out about half the inputs and therefore need a larger variance.

Typical standard deviations for large matrices differ: around 0.07–0.08 for Glorot versus 0.10–0.11 for He.

Transformers often scale residual connections with `x + 0.1 * f(x)` to stabilise depth. Some apply μParam (μP) to decouple width and depth while preserving training dynamics at scale.

Initialization matters most for very deep networks or extreme dtypes such as FP8, where the optimizer cannot easily rescue bad early signal flow.

---

## Source
Percy Liang, **CS336 — Large Language Models**, Stanford University, Lecture 2: *Tensor Fundamentals & Computation Efficiency* (Winter 2025).

## Colab notebook
A Google Colab notebook with the code is available [here](https://colab.research.google.com/drive/1heDkWNo4DDdJn9pXYDFEyUYUHLL3M-qc?usp=sharing).
