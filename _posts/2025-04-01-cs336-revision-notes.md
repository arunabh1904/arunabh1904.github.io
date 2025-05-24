---
layout: post
title: "Percy Liang's CS336 - Revision Notes"
date: 2025-04-01 00:00:00 -0400
categories: ["Revision Notes"]
---

# Percy Liang's CS336 - Revision Notes

These are my personal revision notes for [CS336](https://stanford-cs336.github.io/), 
a course on large language models taught by Percy Liang at Stanford.

# CS336 Revision Notes – Lecture 2

## Tensor Fundamentals & Computational Efficiency

---

### Sources

* Stanford **CS336 (Winter 2025)** Lecture‑2 slides/video — Percy Liang
* [Micikevicius et al., 2018 — *Mixed Precision Training*](https://arxiv.org/abs/1710.03740)
* [Glorot & Bengio, 2010 — *Understanding the difficulty of training deep feed‑forward neural nets*](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
* [He et al., 2015 — *Delving deep into rectifiers*](https://arxiv.org/abs/1502.01852)
* [Rogozhnikov, 2022 — *Einops*](https://einops.rocks/)

---

> **How to read:** Code first (always runnable), then an *engineering commentary* that connects practice ↔ theory. Copy‑paste to a Jupyter cell and play.
> **Legend:** ⏱️ = run‑time concern • 🧐 = conceptual note • 📚 = pointer to paper

---

## Lecture Outline

1. Floating‑Point DTypes
2. Tensor Anatomy & Autograd
3. Views vs Copies
4. Einops for Readable Reshaping
5. FLOPs Accounting
6. Model FLOP Utilisation (MFU)
7. Xavier / He Initialisation

---

## 1 · Floating‑Point DTypes

```python
"""dtype_showcase.py — Precision×Speed explorer"""
import torch, time, math
DTYPES = {
    "float64": torch.float64,   # research‑grade CPU ops
    "float32": torch.float32,   # baseline for training
    "bfloat16": torch.bfloat16, # 8‑bit exponent, AMP‑friendly
    "float16": torch.float16,   # legacy half, beware overflow
    "float8_e4m3fn": torch.float8_e4m3fn, # Hopper‑class inference
}
N = 2_000_000
for name, dt in DTYPES.items():
    x = torch.randn(N, device="cuda" if torch.cuda.is_available() else "cpu", dtype=dt)
    torch.cuda.synchronize() if x.is_cuda else None
    t0 = time.perf_counter(); y = x.square().sqrt();
    torch.cuda.synchronize() if x.is_cuda else None
    dt_ms = (time.perf_counter()-t0)*1e3
    print(f"{name:<11} | {x.element_size()}\u00a0B | {dt_ms:6.2f}\u00a0ms | rel‑err {torch.norm(y-x)/torch.norm(x):.2e}")
```

**🧐 Why care?**
Training is bandwidth‑bound; smaller dtypes raise the *arithmetic‑to‑memory‑traffic ratio* and unlock tensor‑core instructions. `bfloat16` preserves range → no loss scaling pain, whereas `float16` **must** use dynamic scaling (see Micikevicius et al., 2018 📚). FP8 (E4M3/E5M2) is strictly *inference* today because back‑prop squares the gradient spectrum.

**⏱️ Tip:** Benchmark end‑to‑end wall‑time with `torch.cuda.nvtx.range_push/pop` markers — micro‑kernels lie.

---

## 2 · Tensor Anatomy & Autograd

```python
"""autograd_graph.py — Inspect dynamic graph"""
import torch, inspect

t = torch.randn(2,3,4, device="cuda", requires_grad=True)
act = torch.relu(t)
proj = torch.randn(4,3, device="cuda")
z = torch.einsum("bcd,dc->bc", t, proj)  # (2,3)
loss = z.mean(); loss.backward()

for n in [t, act, z, loss]:
    print(f"{n.shape} | grad_fn={type(n.grad_fn).__name__ if n.grad_fn else None}")
```

**🧐 Dynamic graphs**
PyTorch records ops lazily: each tensor carries a `grad_fn` pointing to its creator. Back‑prop runs a reverse topological walk. **In‑place ops** (`tensor.add_`) mutate storage early → autograd inserts **version counters**; mismatch triggers `RuntimeError: one of the variables needed for gradient computation has been modified`.

**📚 Deep dive:** See *PyTorch Autograd Engine* design note (pytorch.org/docs/stable/notes/autograd.html).

---

## 3 · Views vs Copies

```python
"""strides_demo.py"""
import torch; a = torch.arange(16).reshape(4,4)
view = a.t()        # just stride swap
clone = view.clone()# deep copy
print("contiguous?", view.is_contiguous(), clone.is_contiguous())
print("shared\u00a0ptr?", a.storage().data_ptr()==view.storage().data_ptr())
```

**🧐 Strides rule everything**
A tensor is `(data_ptr, sizes, strides)`. `is_contiguous` =  row‑major stride pattern. Transpose costs *zero* until you call a kernel that requires contiguous memory — then PyTorch allocates a fresh buffer implicitly (hidden tax). Prefer keeping the layout you train with (e.g., `[B, Seq, Heads, HeadDim]`) all the way to the matmul.

**⏱️ Memory alias traps:** Overlapping views + writes ⇒ undefined behaviour. Use `torch.Tensor.as_strided` only for hacks; safety comes from `torch._C._debug_only_check_eq_storage_offset`.

---

## 4 · Einops for Readable Reshaping

```python
"""einops_patches.py — ViT example"""
from einops import rearrange, reduce; import torch
img = torch.randn(1,3,224,224)
patches = rearrange(img, "b c (h ph) (w pw) -> b (h w) (ph pw c)", ph=16, pw=16)
cls_token = torch.zeros(1,1,patches.size(-1), device=patches.device)
seq = torch.cat([cls_token, patches], dim=1) # (1,197,768)
print(seq.shape)

img_recon = rearrange(seq[:,1:], "b (h w) (ph pw c) -> b c (h ph) (w pw)",
                      h=14,w=14,ph=16,pw=16,c=3)
assert torch.allclose(img, img_recon)
```

**🧐 Pattern language** ([Rogozhnikov 2022] 📚) → self‑documenting shape transforms: no more `x.view(b, -1, h*w)` guesswork. Rearrange is free if the pattern is stride‑compatible; reduce lowers to efficient kernels (`mean`, `max`, custom lambda). Works across PyTorch/JAX/TF.

---

## 5 · FLOPs Accounting

```python
"""flops_mix.py"""
from math import prod

def attn_flops(seq, dim):
    # QK^T (2)*S*H*D   +  softmax (S^2)  +  AV  (2)*S*H*D
    return 4*seq*seq*dim + seq*seq          # ignore bias, LN

seq, dim = 2048, 128
print(f"Self‑attention ≈ {attn_flops(seq,dim)/1e9:.2f}\u202fGFLOPs")
```

**🧐 What counts?**
Rule of thumb: GEMM ≈ 2 mnk, Conv ≈ 2 C_in C_out K² HW, Attention ≈ 4 S² D. Bias add, ReLU, LayerNorm each <1 % FLOPs for LLMs, yet can dominate **latency** if memory‑bound. FLOPs=scalar multiply‑adds; hardware vendors love quoting TFLOPs/s peak — your kernel mix rarely exceeds 60 % of that.

---

## 6 · Model FLOP Utilisation (MFU)

```python
"""mfu.py"""
import torch, time
from itertools import repeat

def mfu(layer, reps=10):
    if not torch.cuda.is_available():
        return None
    layer.cuda().eval(); inp = torch.randn(*layer.input_shape, device="cuda")
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in repeat(None, reps): layer(inp)
    torch.cuda.synchronize(); dt = (time.perf_counter()-t0)/reps
    return layer.flops / dt / peak_flops(torch.device("cuda"))
```

Build a `Layer` wrapper with `.input_shape` and `.flops` to profile each block (attention, MLP, LayerNorm) and spot under‑utilised outliers. **Roofline analysis**: if MFU ≪ 1 *and* dram_bw_util ≈ 1 → memory‑bound; otherwise launch‑bound.

Tools: `nsys profile`, `nvbench  –metrics achieved_occupancy`.

---

## 7 · Xavier / He Initialisation

```python
"""init_variance.py"""
import torch, math

def fan(t): return t.size(1), t.size(0)  # (fan_in, fan_out)

def glorot(t):
    fi, fo = fan(t); u = math.sqrt(6/(fi+fo));
    return torch.nn.init.uniform_(t, -u, u)

def kaiming(t):
    fi, _ = fan(t); std = math.sqrt(2/fi); return torch.nn.init.normal_(t, 0, std)

w = torch.empty(256,512); glorot(w); print("Glorot std", w.std().item())
```

**🧐 Variance hygiene**
Glorot assumes `tanh` activations → keep var constant fwd & bwd. He adjusts variance ×2 for ReLU’s half‑sparsity. Transformer blocks often use **scaled residuals** (e.g., `0.1 * x + f(x)`) + *μP* parameterisation to decouple width/depth (see *OpenAI μP cookbook* 2023).

**⏱️ When it matters:** matters chiefly for *very deep* networks (>200 layers) or exotic dtypes (FP8). After a few optimizer steps the spectrum is optimiser‑ruled.

--------

