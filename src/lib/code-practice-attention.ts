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


cache = KVCache()
k0 = torch.ones(1, 2, 2, 4, dtype=torch.float64)
v0 = torch.zeros(1, 2, 2, 4, dtype=torch.float64)
k1 = torch.ones(1, 2, 1, 4, dtype=torch.float64)
v1 = torch.zeros(1, 2, 1, 4, dtype=torch.float64)
cache.update(k0, v0, start_pos=0)
keys, values = cache.update(k1, v1, start_pos=2)
print(tuple(keys.shape), tuple(values.shape), cache.length)`,
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
        return self.key, self.value`,
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
    title: 'Implement GQA and MQA',
    difficulty: 'Hard',
    summary:
      'Implement one grouped-query attention kernel whose KV-head count covers MHA, GQA, and MQA.',
    prompt: [
      'Implement `grouped_query_attention(query, key, value, mask=None)` for pre-split tensors. Queries have shape `(B, H_q, T_q, D_head)` while keys and values have shape `(B, H_kv, T_k, D_head)`.',
      'Require `H_q` to be divisible by `H_kv`. Repeat each KV head across its query-head group, compute stable masked attention, and return `(B, H_q, T_q, D_head)`. Explain how `H_kv = H_q`, `1 < H_kv < H_q`, and `H_kv = 1` correspond to MHA, GQA, and MQA.',
    ],
    signature: `def grouped_query_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    ...`,
    requirements: [
      '`query` has shape `(B, H_q, T_q, D_head)`; `key` and `value` share `(B, H_kv, T_k, D_head)`.',
      '`H_q % H_kv == 0`; each KV head serves `H_q / H_kv` adjacent query heads.',
      '`mask`, if supplied, is broadcastable to `(B, H_q, T_q, T_k)` with `1` for visible positions.',
      'Use scaled dot-product attention and a numerically stable masked softmax.',
      'Return `(B, H_q, T_q, D_head)` and reject incompatible shapes.',
    ],
    examples: [
      {
        label: 'Grouped-query attention',
        lines: ['query.shape = (2, 8, 1, 64)', 'key.shape = value.shape = (2, 2, 128, 64)'],
        result: 'output.shape == (2, 8, 1, 64); each KV head serves four query heads',
      },
      {
        label: 'Variant boundary',
        lines: ['H_q = 8'],
        result: 'H_kv = 8 is MHA; H_kv = 2 is GQA; H_kv = 1 is MQA',
      },
    ],
    hint: [
      'Insert a group axis: `(B, H_kv, 1, T_k, D_head)`, broadcast it to the group count, then merge KV-head and group axes.',
      'After repetition, Q/K/V all expose `H_q` heads, so the score shape is `(B, H_q, T_q, T_k)`.',
      'Set blocked logits to negative infinity before the stable softmax.',
      'The explicit repetition is pedagogical. A production GQA kernel should map query heads to KV heads without materializing copies.',
    ],
    reasoning: [
      {
        axis: 'Tensor reasoning',
        detail:
          'Reshape the KV-head mapping as `H_kv × groups = H_q`. Repeating K/V changes the exposed head axis, not batch, sequence, or head width.',
      },
      {
        axis: 'Inference efficiency',
        detail:
          'GQA and MQA reduce KV projection work, cache reads, and memory bandwidth. The attention score tensor still has `H_q` query heads.',
      },
      {
        axis: 'Memory / computation tradeoff',
        detail:
          'Cache size falls by `H_kv / H_q` versus MHA. Explicitly repeating K/V can give that memory back, so optimized kernels use grouped indexing without materializing copies.',
      },
      {
        axis: 'Cache update correctness',
        detail:
          'Store only the original `H_kv` heads. Repeating before the cache write wastes memory and can make the cache layout inconsistent with future updates.',
      },
    ],
    interview: {
      durationMinutes: 45,
      evaluationCriteria: [
        'Derives the KV repeat factor from `H_q / H_kv` and tracks every shape.',
        'Uses stable masked softmax without hiding the core operations.',
        'Quantifies the KV-cache reduction and identifies explicit repetition as a reference-only cost.',
      ],
      followUps: [
        'How would a fused kernel avoid materializing repeated K/V heads?',
        'What fraction of MHA cache memory does `H_q = 32, H_kv = 8` use?',
        'Would GQA reduce the `(T_q, T_k)` score computation by the same factor?',
      ],
    },
    solutionNotes: [
      'MHA, GQA, and MQA differ only in the number of stored key/value heads:\n`repeat = H_q / H_kv`\nEach KV head serves that many query heads.',
      'The reference broadcasts then reshapes K/V so the matrix multiplications are easy to inspect. Optimized kernels keep the compact KV representation and perform the mapping inside the attention kernel.',
      'Store the compact cache:\n`KV cache: (B, H_kv, T_k, D_head)`\nDo not store the repeated query-head view; the compact head axis is where GQA and MQA save decode memory.',
    ],
    solutionDiagram: `Q: (B, Hq,  Tq, Dh)
K: (B, Hkv, Tk, Dh) ─ repeat groups=Hq/Hkv ┐
V: (B, Hkv, Tk, Dh) ─ repeat groups=Hq/Hkv ┤
                                                  ↓
scores: (B, Hq, Tq, Tk) -> output: (B, Hq, Tq, Dh)

Hkv = Hq: MHA   |   1 < Hkv < Hq: GQA   |   Hkv = 1: MQA`,
    starterCode: `from __future__ import annotations

import torch


def grouped_query_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    # TODO: validate heads, expose each KV head to its query group, then attend stably.
    raise NotImplementedError("Implement grouped_query_attention")


query = torch.ones(1, 4, 1, 2, dtype=torch.float64)
key = torch.ones(1, 2, 3, 2, dtype=torch.float64)
value = torch.tensor(
    [[[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
      [[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]]]],
    dtype=torch.float64,
)
print(grouped_query_attention(query, key, value).shape)`,
    solutionCode: `import torch

def _repeat_kv(tensor, repeats):
    batch, heads, length, dim = tensor.shape
    expanded = torch.broadcast_to(
        tensor[:, :, None, :, :], (batch, heads, repeats, length, dim)
    )
    return torch.reshape(expanded, (batch, heads * repeats, length, dim))

def _stable_masked_softmax(scores):
    valid = torch.isfinite(scores)
    scores = torch.where(valid, scores, torch.zeros_like(scores))
    scores = scores - torch.amax(scores, dim=-1, keepdim=True)
    weights = torch.exp(scores) * torch.as_tensor(valid, dtype=scores.dtype)
    return weights / torch.clamp(torch.sum(weights, dim=-1, keepdim=True), min=1e-8)

def grouped_query_attention(query, key, value, mask=None):
    if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape: raise ValueError("expected Q and matching K/V tensors with rank four")
    batch, query_heads, query_len, head_dim = query.shape
    kv_batch, kv_heads, key_len, kv_dim = key.shape
    if batch != kv_batch or head_dim != kv_dim or query_heads % kv_heads != 0: raise ValueError("incompatible batch, head, or feature dimensions")
    repeats = query_heads // kv_heads
    key, value = _repeat_kv(key, repeats), _repeat_kv(value, repeats)
    scores = query @ key.transpose(-1, -2) / (head_dim ** 0.5)
    if mask is not None:
        mask = torch.broadcast_to(torch.as_tensor(mask), scores.shape)
        scores = torch.where(mask != 0, scores, torch.full_like(scores, float("-inf")))
    return _stable_masked_softmax(scores) @ value`,
    walkthroughCode: `import torch

def _repeat_kv(tensor, repeats):
    batch, heads, length, dim = tensor.shape
    # Insert a group axis, broadcast each stored head, then merge head and group.
    expanded = torch.broadcast_to(
        tensor[:, :, None, :, :], (batch, heads, repeats, length, dim)
    )
    return torch.reshape(expanded, (batch, heads * repeats, length, dim))

def _stable_masked_softmax(scores):
    valid = torch.isfinite(scores)
    scores = torch.where(valid, scores, torch.zeros_like(scores))
    scores = scores - torch.amax(scores, dim=-1, keepdim=True)
    weights = torch.exp(scores) * torch.as_tensor(valid, dtype=scores.dtype)
    return weights / torch.clamp(torch.sum(weights, dim=-1, keepdim=True), min=1e-8)

def grouped_query_attention(query, key, value, mask=None):
    if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
        raise ValueError("expected Q and matching K/V tensors with rank four")
    batch, query_heads, query_len, head_dim = query.shape
    kv_batch, kv_heads, key_len, kv_dim = key.shape
    if batch != kv_batch or head_dim != kv_dim or query_heads % kv_heads != 0:
        raise ValueError("incompatible batch, head, or feature dimensions")

    # MHA repeats once, GQA repeats by its group size, and MQA repeats one KV head Hq times.
    repeats = query_heads // kv_heads
    key, value = _repeat_kv(key, repeats), _repeat_kv(value, repeats)
    scores = query @ key.transpose(-1, -2) / (head_dim ** 0.5)
    if mask is not None:
        mask = torch.broadcast_to(torch.as_tensor(mask), scores.shape)
        scores = torch.where(mask != 0, scores, torch.full_like(scores, float("-inf")))
    return _stable_masked_softmax(scores) @ value`,
    packages: PYTORCH_AND_NUMPY_PACKAGES,
    tags: ['PyTorch', 'Transformers', 'GQA', 'MQA', 'Inference'],
  },
] as const satisfies readonly CodePracticeProblem[];
