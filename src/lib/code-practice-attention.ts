import type { CodePracticeProblem } from './code-practice';

const PYTORCH_AND_NUMPY_PACKAGES = ['torch', 'numpy'] as const;

type AttentionProblemEnrichment = Pick<
  CodePracticeProblem,
  'reasoning' | 'interview'
> &
  Partial<Pick<CodePracticeProblem, 'title' | 'summary'>>;

export const ATTENTION_PROBLEM_ENRICHMENTS: Readonly<
  Record<string, AttentionProblemEnrichment>
> = {
  'stable-softmax-cross-entropy': {
    reasoning: [
      {
        axis: 'Tensor reasoning',
        detail:
          'For logits `(N, C)`, the row maximum has shape `(N, 1)`. Keeping the class axis makes the subtraction broadcast across each row without mixing examples.',
      },
      {
        axis: 'Inference efficiency',
        detail:
          'The reference exposes the reductions. Production kernels normally fuse the max, exponentiation, sum, and normalization to reduce memory traffic.',
      },
      {
        axis: 'Memory / computation tradeoff',
        detail:
          'Materializing shifted logits and exponentials is clear but costs two `(N, C)` buffers. A fused kernel keeps fewer intermediates live.',
      },
    ],
    interview: {
      durationMinutes: 30,
      evaluationCriteria: [
        'States why subtracting one row maximum leaves softmax probabilities unchanged.',
        'Tracks `(N, C)`, `(N, 1)`, and `(N,)` through every reduction and gather.',
        'Tests a large-logit case that would overflow a naive implementation.',
      ],
      followUps: [
        'How would you handle an all-masked row in attention softmax?',
        'Why might a fused softmax kernel use less memory than this reference?',
      ],
    },
  },
  'causal-attention-mask': {
    reasoning: [
      {
        axis: 'Tensor reasoning',
        detail:
          'Query positions form the row axis and key positions form the column axis. Comparing `(T, 1) >= (1, T)` creates the `(T, T)` visibility matrix.',
      },
      {
        axis: 'Inference efficiency',
        detail:
          'During single-token append-only decoding, a kernel can avoid materializing a full mask because the query is already at the newest valid position.',
      },
      {
        axis: 'Memory / computation tradeoff',
        detail:
          'A dense `(T, T)` mask is easy to inspect but grows quadratically. Production attention kernels usually encode causality in indexing rather than storing the matrix.',
      },
      {
        axis: 'Cache update correctness',
        detail:
          'With a cache, compare absolute query positions against absolute key positions. A fresh local lower triangle is wrong when the query chunk starts after position zero.',
      },
    ],
    interview: {
      durationMinutes: 30,
      evaluationCriteria: [
        'Names the query-row and key-column axes before broadcasting them.',
        'Combines causality with per-example padding without Python loops.',
        'Explains how cached decoding changes the position indices.',
      ],
      followUps: [
        'How would the mask change for a query chunk that begins at `start_pos > 0`?',
        'When can a decoding kernel omit the explicit mask?',
      ],
    },
  },
  'rope-rotary-positional-embedding': {
    reasoning: [
      {
        axis: 'Tensor reasoning',
        detail:
          'Even and odd channels each have shape `(B, T, H, D / 2)`. Sine and cosine tables use `(1, T, 1, D / 2)` so they broadcast over batches and heads.',
      },
      {
        axis: 'Inference efficiency',
        detail:
          'Decode only needs the angles for the new absolute positions. Rebuilding a table for the whole prefix repeats work that the cache does not need.',
      },
      {
        axis: 'Memory / computation tradeoff',
        detail:
          'Precomputed sine and cosine tables trade a small linear memory cost for fewer transcendental operations. Computing them on demand avoids a long-lived table.',
      },
      {
        axis: 'Cache update correctness',
        detail:
          'Rotate each key exactly once at its absolute position before caching it. Re-rotating cached keys or restarting positions at zero silently corrupts attention scores.',
      },
    ],
    interview: {
      durationMinutes: 35,
      evaluationCriteria: [
        'Explains the adjacent-pair rotation and the even head-dimension requirement.',
        'Tracks the broadcast shape of the angle table.',
        'Connects the position offset to incremental KV-cache updates.',
      ],
      followUps: [
        'How would you add `start_pos` for cached decoding?',
        'Why are queries and keys rotated but values are not?',
      ],
    },
  },
  'scaled-dot-product-self-attention': {
    title: 'Multi-head self-attention (MHA)',
    summary:
      'Implement MHA end to end: project Q/K/V, split heads, apply stable masked attention, merge heads, and project the result.',
    reasoning: [
      {
        axis: 'Tensor reasoning',
        detail:
          'Splitting `(B, T, D_model)` gives Q/K/V shaped `(B, H, T, D_head)`. `Q @ Kᵀ` then produces scores `(B, H, T, T)` before values restore the last axis to `D_head`.',
      },
      {
        axis: 'Inference efficiency',
        detail:
          'Prefill computes all token pairs. During decode, caching past K/V avoids re-projecting the prefix, so one new query attends over `T` cached keys instead of recomputing the whole prefix.',
      },
      {
        axis: 'Memory / computation tradeoff',
        detail:
          'The score matrix is quadratic during prefill, while the decode cache grows linearly as `2 × B × H × T × D_head` elements for keys and values.',
      },
      {
        axis: 'Cache update correctness',
        detail:
          'A cached implementation must append K/V along the sequence axis, preserve batch/head/head-dimension layout, and apply the same positional transform exactly once.',
      },
    ],
    interview: {
      durationMinutes: 45,
      evaluationCriteria: [
        'Writes every intermediate tensor shape before the matrix multiplications.',
        'Applies masking before a stable softmax and handles blocked entries exactly.',
        'Explains what changes between prefill and single-token decode.',
      ],
      followUps: [
        'Where would you insert RoPE and a KV cache?',
        'Which tensors dominate memory during prefill and during decode?',
      ],
    },
  },
};

export const ATTENTION_CODE_PRACTICE_PROBLEMS = [
  {
    id: 'incremental-kv-cache',
    order: 29,
    title: 'Build an append-correct KV cache',
    difficulty: 'Hard',
    summary:
      'Implement a stateful KV cache that validates tensor layout, appends along the sequence axis, and rejects stale or skipped positions.',
    prompt: [
      'You are implementing autoregressive decoding. Build a `KVCache` dataclass whose `update(key, value, start_pos)` method stores new key/value chunks shaped `(B, H_kv, T_new, D_head)`.',
      'Treat `start_pos` as an explicit correctness contract. An update is valid only when it starts at the current cache length; stale writes and gaps must fail instead of silently changing token positions.',
    ],
    signature: `@dataclass(slots=True)
class KVCache:
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None
    length: int = 0

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        start_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...`,
    requirements: [
      '`key` and `value` have the same positive shape `(B, H_kv, T_new, D_head)`.',
      '`start_pos` must equal the current cache length; the first update must start at zero.',
      'Later updates must preserve batch size, KV-head count, and head dimension.',
      'Append only along sequence axis `2`, update `length`, and return the complete cached tensors.',
      'Raise `ValueError` for malformed tensors, stale writes, or skipped positions.',
    ],
    examples: [
      {
        label: 'Two decode updates',
        lines: [
          'cache.update(k0, v0, start_pos=0)  # k0.shape == (1, 2, 2, 4)',
          'keys, values = cache.update(k1, v1, start_pos=2)',
        ],
        result: 'keys.shape == values.shape == (1, 2, 3, 4) and cache.length == 3',
      },
      {
        label: 'Rejected stale write',
        lines: ['cache.length == 3', 'cache.update(k1, v1, start_pos=2)'],
        result: 'ValueError: start_pos must equal the current cache length',
      },
    ],
    hint: [
      'The sequence axis is `2` in `(B, H_kv, T, D_head)`; the other three axes must stay stable across updates.',
      'Check `start_pos == self.length` before mutating either tensor.',
      'Clone the first chunk so outside mutation cannot rewrite cached history.',
      'This browser reference uses `torch.cat` for clarity. Be ready to discuss why a production cache would preallocate storage.',
    ],
    reasoning: [
      {
        axis: 'Tensor reasoning',
        detail:
          'Only axis `2` grows. Batch size, KV-head count, and head dimension identify the cache layout and must match every later update.',
      },
      {
        axis: 'Inference efficiency',
        detail:
          'The cache removes repeated K/V projection for the prefix. Each decode step projects only the new token chunk and reads the stored history.',
      },
      {
        axis: 'Memory / computation tradeoff',
        detail:
          'The cache uses linear memory, `2 × B × H_kv × T × D_head` elements. This clear `cat` implementation reallocates; production systems preallocate or page storage to avoid quadratic copying.',
      },
      {
        axis: 'Cache update correctness',
        detail:
          'Checking `start_pos == length` makes updates append-only. Perform every validation before assignment so a failed update cannot leave key and value state out of sync.',
      },
    ],
    interview: {
      durationMinutes: 40,
      evaluationCriteria: [
        'States the `(B, H_kv, T, D_head)` layout and identifies the append axis.',
        'Validates the full update before mutating cache state.',
        'Explains concatenation versus preallocation and quantifies cache growth.',
      ],
      followUps: [
        'How would you redesign this with preallocated or paged storage?',
        'Where should RoPE be applied relative to the cache write?',
        'How would beam search change the cache API?',
      ],
    },
    solutionNotes: [
      'The cache has one critical invariant:\n`start_pos == current_cache_length`\nA smaller position overwrites history; a larger one leaves a gap. Reject both.',
      'Keys and values use the same layout and grow only on the token axis:\n`(B, H_kv, T_past, D_head) + (B, H_kv, T_new, D_head)`\n`→ (B, H_kv, T_past + T_new, D_head)`\nBatch, KV-head count, and head width must remain fixed.',
      'All validation happens before mutation. That ordering matters because an exception after writing keys but before writing values would leave an unusable cache.',
      'Concatenation keeps the interview implementation readable. A serving system would normally preallocate, use blocks, or use paged attention so repeated appends do not copy the full prefix.',
    ],
    solutionDiagram: `new K,V: (B, Hkv, Tnew, Dh)
                 append on axis 2
cache K,V: (B, Hkv, Tpast + Tnew, Dh)

legal update: start_pos == cache.length
new length:   start_pos + Tnew`,
    starterCode: `from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class KVCache:
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None
    length: int = 0

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        start_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: validate the whole append, mutate both tensors, then advance length.
        raise NotImplementedError("Implement update")

def smoke_test() -> None:
    cache = KVCache()
    first = torch.ones(1, 2, 2, 4)
    second = torch.zeros(1, 2, 1, 4)
    cache.update(first, first, start_pos=0)
    keys, values = cache.update(second, second, start_pos=2)
    assert keys.shape == values.shape == (1, 2, 3, 4) and cache.length == 3
    try:
        cache.update(second, second, start_pos=2)
        raise AssertionError("stale writes must fail")
    except ValueError:
        pass
    print("KV cache smoke test passed:", tuple(keys.shape))

smoke_test()`,
    solutionCode: `from dataclasses import dataclass
import torch

@dataclass(slots=True)
class KVCache:
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None
    length: int = 0

    def update(self, key, value, start_pos):
        key = torch.as_tensor(key, dtype=torch.float64)
        value = torch.as_tensor(value, dtype=torch.float64)
        if key.ndim != 4 or any(size <= 0 for size in key.shape): raise ValueError("key must have positive shape (B, H_kv, T, D_head)")
        if value.shape != key.shape: raise ValueError("value must match key shape")
        if start_pos != self.length: raise ValueError("start_pos must equal the current cache length")
        if self.key is not None:
            cached_layout = (self.key.shape[0], self.key.shape[1], self.key.shape[3])
            update_layout = (key.shape[0], key.shape[1], key.shape[3])
            if cached_layout != update_layout: raise ValueError("cache layout changed")
            next_key = torch.cat((self.key, key), dim=2)
            next_value = torch.cat((self.value, value), dim=2)
        else:
            next_key, next_value = key.clone(), value.clone()
        self.key, self.value = next_key, next_value
        self.length = start_pos + key.shape[2]
        return self.key, self.value

def smoke_test():
    cache = KVCache()
    first = torch.ones(1, 2, 2, 4)
    second = torch.zeros(1, 2, 1, 4)
    cache.update(first, first, start_pos=0)
    keys, values = cache.update(second, second, start_pos=2)
    assert keys.shape == values.shape == (1, 2, 3, 4) and cache.length == 3
    try:
        cache.update(second, second, start_pos=2)
        raise AssertionError("stale writes must fail")
    except ValueError:
        pass
    print("KV cache smoke test passed:", tuple(keys.shape))

smoke_test()`,
    walkthroughCode: `from dataclasses import dataclass
import torch

@dataclass(slots=True)
class KVCache:
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None
    length: int = 0

    def update(self, key, value, start_pos):
        # Normalize both chunks before checking any state transition.
        key = torch.as_tensor(key, dtype=torch.float64)
        value = torch.as_tensor(value, dtype=torch.float64)
        if key.ndim != 4 or any(size <= 0 for size in key.shape):
            raise ValueError("key must have positive shape (B, H_kv, T, D_head)")
        if value.shape != key.shape:
            raise ValueError("value must match key shape")
        # Equality rejects both a stale overwrite and a skipped position.
        if start_pos != self.length:
            raise ValueError("start_pos must equal the current cache length")

        if self.key is not None:
            # Sequence length may grow; batch, KV heads, and head width may not.
            cached_layout = (self.key.shape[0], self.key.shape[1], self.key.shape[3])
            update_layout = (key.shape[0], key.shape[1], key.shape[3])
            if cached_layout != update_layout:
                raise ValueError("cache layout changed")
            next_key = torch.cat((self.key, key), dim=2)
            next_value = torch.cat((self.value, value), dim=2)
        else:
            # Own the first chunk instead of aliasing the caller's mutable tensor.
            next_key, next_value = key.clone(), value.clone()

        # Mutate only after every check and both next tensors have succeeded.
        self.key, self.value = next_key, next_value
        self.length = start_pos + key.shape[2]
        return self.key, self.value`,
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'Transformers', 'Inference', 'KV Cache'],
  },
  {
    id: 'grouped-query-and-multi-query-attention',
    order: 30,
    title: 'Grouped-query and multi-query attention (GQA/MQA)',
    difficulty: 'Hard',
    summary:
      'Implement grouped-query attention where many query heads share fewer key/value heads, with MQA as the one-KV-head limit.',
    prompt: [
      'Implement `GroupedQueryAttention` as an `nn.Module` with `d_model`, `num_query_heads`, and `num_kv_heads`. Use the same input sequence for queries, keys, and values.',
      'Queries use `Hq` heads, while keys and values use only `Hkv` heads. Repeat each KV head across its query-head group, run standard scaled dot-product attention, and return an output with the same shape as the input.',
    ],
    signature: `class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
    ):
        ...

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        ...`,
    requirements: [
      '`x` has shape `(B, L, d_model)`; `d_model` must be divisible by `num_query_heads`.',
      '`num_query_heads % num_kv_heads == 0`; each KV head is shared by an adjacent group of query heads.',
      'Project Q to `d_model` channels, but project K and V to `num_kv_heads * head_dim` channels.',
      'Split `[B, L, H * D_head]` into `[B, H, L, D_head]` and repeat K/V along the head axis with `repeat_interleave`.',
      'Use scaled dot-product attention with a stable softmax over the final key-position axis.',
      'Return `(B, L, d_model)` and include a test with `d_model=32`, `Hq=8`, and `Hkv=2`.',
    ],
    examples: [
      {
        label: 'GQA shape flow',
        lines: [
          'x.shape = (2, 5, 32)',
          'Hq = 8, Hkv = 2, head_dim = 4',
          'Q.shape = (2, 8, 5, 4)',
          'K.shape = V.shape = (2, 2, 5, 4) before repetition',
        ],
        result: 'after repetition, K and V are (2, 8, 5, 4); output.shape == (2, 5, 32)',
      },
      {
        label: 'Head-count hierarchy',
        lines: [
          'Hq = Hkv  -> MHA',
          '1 < Hkv < Hq -> GQA',
          'Hkv = 1  -> MQA',
        ],
        result: 'the attention path is the same; only the number of stored KV heads changes',
      },
    ],
    hint: [
      'Compute `head_dim = d_model // num_query_heads`, then set `kv_dim = num_kv_heads * head_dim` for the K/V projections.',
      'The repeat factor is `num_query_heads // num_kv_heads`. Repeat on dimension `1`, the head dimension after splitting.',
      'Once K/V expose `Hq` heads, the rest is ordinary attention: `Q @ K.transpose(-2, -1)`, scale, softmax, and `weights @ V`.',
      'Permute `[B, Hq, L, D_head]` to `[B, L, Hq, D_head]` before flattening the heads back to `d_model`.',
      'The reference materializes repeated K/V tensors so the mapping is visible. Optimized kernels can share them by indexing instead.',
    ],
    reasoning: [
      {
        axis: 'Tensor reasoning',
        detail:
          'With `H_q × D_head = d_model`, Q has shape `(B, H_q, L, D_head)` while compact K/V have `(B, H_kv, L, D_head)`. Repetition changes only the head axis.',
      },
      {
        axis: 'Inference efficiency',
        detail:
          'GQA and MQA reduce KV projection work, cache reads, and memory bandwidth. The score tensor still has `H_q` query heads, so query-side expressivity remains.',
      },
      {
        axis: 'Memory / computation tradeoff',
        detail:
          'The KV cache uses `H_kv / H_q` of the MHA head storage. Explicitly repeating K/V can give that memory back in this reference, so optimized kernels use grouped indexing without materializing copies.',
      },
      {
        axis: 'Cache update correctness',
        detail:
          'Store only the compact `(B, H_kv, L, D_head)` keys and values. Repeat them when computing attention, not when writing the cache.',
      },
    ],
    interview: {
      durationMinutes: 45,
      evaluationCriteria: [
        'Derives the KV repeat factor from `H_q / H_kv` and tracks every shape.',
        'Uses stable softmax without hiding the core operations.',
        'Quantifies the KV-cache reduction and identifies explicit repetition as a reference-only cost.',
      ],
      followUps: [
        'How would a fused kernel avoid materializing repeated K/V heads?',
        'What fraction of MHA cache memory does `Hq = 32, Hkv = 8` use?',
        'Would GQA reduce the `(T_q, T_k)` score computation by the same factor?',
      ],
    },
    solutionNotes: [
      'MHA, GQA, and MQA use the same attention equation. They differ in how many K/V heads are stored:\n`Hq = Hkv -> MHA`\n`1 < Hkv < Hq -> GQA`\n`Hkv = 1 -> MQA`',
      'The shape split is:\n`Q: (B, Hq, L, Dh)`\n`K,V: (B, Hkv, L, Dh)`\nwhere `Dh = d_model / Hq`.',
      'Each compact KV head serves a group of query heads. The repeat factor is:\n`repeats = Hq // Hkv`\n`k = k.repeat_interleave(repeats, dim=1)`',
      'After repetition, standard attention applies:\n`scores = q @ k.transpose(-2, -1) / sqrt(Dh)`\n`weights = stable_softmax(scores, dim=-1)`\n`context = weights @ v`',
      'For `Hq=8` and `Hkv=2`, each KV head serves four query heads. The compact KV cache stores about `2 / 8 = 1/4` of the MHA head storage, while the score tensor still uses eight query heads.',
      'Memory cue: project Q with `Hq`; project K/V with `Hkv`; split; repeat K/V on the head axis; run ordinary attention; merge. Store the compact K/V view in a decoder cache.',
    ],
    solutionDiagram: `x: (B, L, d_model)
       ├─ Q projection -> (B, Hq,  L, Dh)
       └─ K,V projections -> (B, Hkv, L, Dh)

K,V ─ repeat each head Hq/Hkv times on axis 1 ─┐
Q ─────────────────────────────────────────────┤
scores: (B, Hq, L, L) -> context -> merge -> (B, L, d_model)

Hkv = Hq: MHA   |   1 < Hkv < Hq: GQA   |   Hkv = 1: MQA`,
    starterCode: `import math

import torch
from torch import nn

def stable_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # TODO: subtract the maximum, exponentiate, and normalize along dim.
    raise NotImplementedError("Implement stable_softmax")

class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
    ) -> None:
        # TODO: validate head counts and create full-width Q/O and compact K/V projections.
        raise NotImplementedError("Implement __init__")

    def split_heads(
        self,
        x: torch.Tensor,
        num_heads: int,
    ) -> torch.Tensor:
        # TODO: reshape [B, L, H * Dh] into [B, H, L, Dh].
        raise NotImplementedError("Implement split_heads")

    def repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: share each KV head across its query-head group.
        raise NotImplementedError("Implement repeat_kv")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: project, split, repeat K/V, attend, merge, and project out.
        raise NotImplementedError("Implement forward")

def test_gqa() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 5, 32)
    layer = GroupedQueryAttention(32, 8, 2)
    output = layer(x)
    print(output.shape)
    assert output.shape == (2, 5, 32)

test_gqa()`,
    solutionCode: `import math

import torch
from torch import nn

def stable_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, num_query_heads: int, num_kv_heads: int) -> None:
        super().__init__()
        if d_model <= 0 or num_query_heads <= 0 or num_kv_heads <= 0:
            raise ValueError("head counts and d_model must be positive")
        if d_model % num_query_heads or num_query_heads % num_kv_heads:
            raise ValueError("num_query_heads must divide d_model and num_kv_heads")
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_query_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        kv_dim = num_kv_heads * self.head_dim
        self.k_proj = nn.Linear(d_model, kv_dim, bias=False)
        self.v_proj = nn.Linear(d_model, kv_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        batch, length, _ = x.shape
        x = x.reshape(batch, length, num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        repeats = self.num_query_heads // self.num_kv_heads
        return x.repeat_interleave(repeats, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.split_heads(self.q_proj(x), self.num_query_heads)
        k = self.split_heads(self.k_proj(x), self.num_kv_heads)
        v = self.split_heads(self.v_proj(x), self.num_kv_heads)
        k, v = self.repeat_kv(k), self.repeat_kv(v)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        weights = stable_softmax(scores, dim=-1)
        context = (weights @ v).permute(0, 2, 1, 3)
        context = context.reshape(x.shape[0], x.shape[1], self.d_model)
        return self.out_proj(context)

def test_gqa() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 5, 32)
    layer = GroupedQueryAttention(32, 8, 2)
    output = layer(x)
    print(output.shape)
    assert output.shape == (2, 5, 32)

test_gqa()`,
    walkthroughCode: `import math

import torch

def repeat_kv(x: torch.Tensor, repeats: int) -> torch.Tensor:
    return x.repeat_interleave(repeats, dim=1)

def stable_softmax(x: torch.Tensor) -> torch.Tensor:
    x = x - x.max(dim=-1, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=-1, keepdim=True)

def grouped_query_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    repeats = query.shape[1] // key.shape[1]
    key = repeat_kv(key, repeats)
    value = repeat_kv(value, repeats)
    scores = query @ key.transpose(-2, -1) / math.sqrt(query.shape[-1])
    return stable_softmax(scores) @ value`,
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'Transformers', 'GQA', 'MQA', 'Inference'],
  },
] as const satisfies readonly CodePracticeProblem[];
