---
layout: content
title: "Lecture 2"
date: 2025-05-15 00:00:00 -0400
categories: ["Revision Notes"]
---

# CS336 Lecture 2 — Tensor Fundamentals & Computation Efficiency

*Revision notes updated May 24 2025*

---

## Quick Overview

This lecture discusses different data types and guides you on when each should be used, provides a clear mental model of computational structures, briefly covers tensor reshaping with Einops, and introduces computational efficiency metrics and initialization strategies.

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

Training is usually **bandwidth-bound**; halving the element size doubles the compute-to-memory-traffic ratio and can switch kernels onto tensor-cores. `float64` is the original high precision variant typically used for CPU Ops. `float32` is the default setting for training deep learning models. `float16` and `bfloat16` are both half precision variants but, bfloat16 keeps the 8-bit exponent of FP32 so you almost never need loss scaling, while float16 often does. `FP8 (E4M3/E5M2)` is bleeding-edge and **strictly inference** for now – back-prop squares gradients and wrecks its tiny dynamic range. There are also `FP4` variants enabled by blackwell GPUs. 

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

The `grad_fn` field lets you walk the dynamic computation graph **without** external tools — perfect for debugging odd gradients.

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

**Strides rule everything.** A tensor is `(data_ptr, sizes, strides)`. Transpose costs 0 B until you hit a kernel that needs contiguous memory — then PyTorch silently allocates a fresh buffer. Keep one consistent layout (e.g. `[B, Seq, Heads, Dim]`) through your pipeline.

---

## 5 · Clean Tensor Manipulations with Einops

Einops is a library that provides a concise and intuitive syntax for tensor manipulations such as reshaping, permuting, and reducing tensors. It enhances code readability and helps avoid errors associated with manual indexing and reshaping operations.

```python
from einops import rearrange, reduce, repeat

x = torch.randn(2, 224, 224, 3)

x_nchw = rearrange(x, "b h w c -> b c h w")
seq = rearrange(x_nchw, "b c h w -> b (h w) c")
x_reconstructed = rearrange(seq, "b (h w) c -> b c h w", h=224)
gap = reduce(x_nchw, "b c h w -> b c", "mean")
emb_map = repeat(torch.randn(2, 64), "b d -> b d h w", h=7, w=7)
```

Benefit: Einops manipulates only metadata when strides allow, enabling intuitive, safe reshaping without unintended copies, thus avoiding unnecessary bandwidth usage.

---

## 6 · Tracking computational efficiency

FLOPs (Floating Point Operations per Second) measure computational efficiency. Model FLOP Utilization (MFU) is the ratio of actual FLOPs achieved by the hardware to its theoretical maximum, typically less than 1 due to overheads like memory latency, kernel launch overheads, and suboptimal hardware utilization.

<div class="equation-box">
\[
MFU = \frac{\text{achieved FLOPs}}{\text{theoretical FLOPs}}
\]
</div>

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

Bias, ReLU, and LayerNorm are <1 % of an LLM’s FLOPs yet can dominate latency if launch overhead or memory stalls bite.

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

Glorot/Xavier Initialisation (Glorot & Bengio, 2010) assumes symmetric activations like `tanh` and balances the variance between the forward and backward passes.

He Initialisation (He et al., 2015) is tailored for ReLU and its derivatives, which zero out half the inputs and thus double the variance.

Typical standard deviations for large matrices differ: around 0.07–0.08 for Glorot versus 0.10–0.11 for He.

Transformers often scale residual connections with `x + 0.1 * f(x)` to stabilise depth. Some apply μParam (μP) to decouple width and depth while preserving training dynamics at scale.

When does it matter? For very deep networks (>200 layers) or extreme dtypes such as FP8, where poor initialisation cannot be fixed by the optimiser.

---

## Source
Percy Liang, **CS336 — Large Language Models**, Stanford University, Lecture 2: *Tensor Fundamentals & Computation Efficiency* (Winter 2025).

## Colab notebook
A Google Colab notebook with the code is available [here](https://colab.research.google.com/drive/1heDkWNo4DDdJn9pXYDFEyUYUHLL3M-qc?usp=sharing).
