/**
 * A deliberately small PyTorch-shaped runtime for the browser exercises.
 *
 * Pyodide does not ship the compiled PyTorch runtime. The snippets still use
 * real PyTorch names and tensor idioms, but this compatibility layer maps the
 * subset used by the practice problems to NumPy arrays. It is not a drop-in
 * replacement for PyTorch: there is no autograd, CUDA, nn.Module, or compiler.
 */
export const TORCH_COMPAT_PACKAGE = 'torch';

export const TORCH_COMPAT_SOURCE = String.raw`
import sys as _sys
import types as _types
import numpy as _np

_TORCH_COMPAT_INT_DTYPES = {
    _np.dtype(_np.int8),
    _np.dtype(_np.int16),
    _np.dtype(_np.int32),
    _np.dtype(_np.int64),
    _np.dtype(_np.uint8),
}

def _asarray(value, dtype=None):
    return _np.asarray(value, dtype=dtype)

def _shape(size):
    return (int(size),) if isinstance(size, (int, _np.integer)) else tuple(int(v) for v in size)

def _axis(dim):
    return dim

def _safe_denominator(denominator):
    return _np.where(denominator > 0, denominator, _np.ones_like(denominator))

def _stable_softmax(value, dim=-1):
    value = _asarray(value, dtype=_np.float64)
    finite = _np.isfinite(value)
    safe_value = _np.where(finite, value, 0.0)
    maximum = _np.max(safe_value, axis=dim, keepdims=True)
    exponentiated = _np.exp(safe_value - maximum) * finite
    denominator = _np.sum(exponentiated, axis=dim, keepdims=True)
    return _np.where(
        denominator > 0,
        exponentiated / _safe_denominator(denominator),
        _np.zeros_like(exponentiated),
    )

def _log_softmax(value, dim=-1):
    value = _asarray(value, dtype=_np.float64)
    finite = _np.isfinite(value)
    safe_value = _np.where(finite, value, 0.0)
    maximum = _np.max(safe_value, axis=dim, keepdims=True)
    shifted = safe_value - maximum
    denominator = _np.sum(_np.exp(shifted) * finite, axis=dim, keepdims=True)
    return shifted - _np.log(_safe_denominator(denominator))

def _sum(value, dim=None, keepdim=False):
    return _np.sum(value, axis=_axis(dim), keepdims=keepdim)

def _mean(value, dim=None, keepdim=False):
    return _np.mean(value, axis=_axis(dim), keepdims=keepdim)

def _amax(value, dim=None, keepdim=False):
    return _np.max(value, axis=_axis(dim), keepdims=keepdim)

def _amin(value, dim=None, keepdim=False):
    return _np.min(value, axis=_axis(dim), keepdims=keepdim)

def _norm(value, p=2, dim=None, keepdim=False):
    if p != 2:
        return _np.linalg.norm(value, ord=p, axis=_axis(dim), keepdims=keepdim)
    return _np.sqrt(_np.sum(_np.asarray(value) ** 2, axis=_axis(dim), keepdims=keepdim))

def _arange(start, end=None, step=1, dtype=None, **_kwargs):
    if end is None:
        start, end = 0, start
    return _np.arange(start, end, step, dtype=dtype)

def _flatten(value, start_dim=0, end_dim=-1):
    value = _np.asarray(value)
    if end_dim < 0:
        end_dim += value.ndim
    prefix = value.shape[:start_dim]
    suffix = value.shape[end_dim + 1:]
    middle = int(_np.prod(value.shape[start_dim:end_dim + 1]))
    return value.reshape(prefix + (middle,) + suffix)

def _gather(value, dim, index):
    value = _np.asarray(value)
    index = _np.asarray(index, dtype=_np.int64)
    return _np.take_along_axis(value, index, axis=dim)

def _topk(value, k, dim=-1, largest=True, sorted=True):
    value = _np.asarray(value)
    order = _np.argsort(-value if largest else value, axis=dim, kind='stable')
    indices = _np.take(order, _np.arange(k), axis=dim)
    values = _np.take_along_axis(value, indices, axis=dim)
    return _types.SimpleNamespace(values=values, indices=indices)

def _cross_entropy(logits, targets, reduction='mean'):
    log_probs = _log_softmax(logits, dim=-1)
    targets = _np.asarray(targets, dtype=_np.int64)
    losses = -_np.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
    if reduction == 'none':
        return losses
    if reduction == 'sum':
        return _np.sum(losses)
    return _np.mean(losses)

def _logsumexp(value, dim=-1, keepdim=False):
    value = _asarray(value, dtype=_np.float64)
    maximum = _np.max(value, axis=dim, keepdims=True)
    result = maximum + _np.log(_np.sum(_np.exp(value - maximum), axis=dim, keepdims=True))
    return result if keepdim else _np.squeeze(result, axis=dim)

class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

torch = _types.ModuleType('torch')
torch.__path__ = []
torch.Tensor = _np.ndarray
torch.float16 = _np.float16
torch.float32 = _np.float32
torch.float64 = _np.float64
torch.int8 = _np.int8
torch.int16 = _np.int16
torch.int32 = _np.int32
torch.int64 = _np.int64
torch.long = _np.int64
torch.uint8 = _np.uint8
torch.bool = _np.bool_
torch.as_tensor = _asarray
torch.tensor = lambda value, dtype=None, **_kwargs: _np.array(value, dtype=dtype)
torch.from_numpy = lambda value: _np.asarray(value)
torch.clone = lambda value: _np.array(value, copy=True)
torch.zeros = lambda size, dtype=_np.float32, **_kwargs: _np.zeros(_shape(size), dtype=dtype)
torch.ones = lambda size, dtype=_np.float32, **_kwargs: _np.ones(_shape(size), dtype=dtype)
torch.empty = lambda size, dtype=_np.float32, **_kwargs: _np.empty(_shape(size), dtype=dtype)
torch.full = lambda size, fill_value, dtype=None, **_kwargs: _np.full(_shape(size), fill_value, dtype=dtype)
torch.zeros_like = lambda value, **_kwargs: _np.zeros_like(value)
torch.ones_like = lambda value, **_kwargs: _np.ones_like(value)
torch.empty_like = lambda value, **_kwargs: _np.empty_like(value)
torch.full_like = lambda value, fill_value, **_kwargs: _np.full_like(value, fill_value)
torch.arange = _arange
torch.tril = lambda value, diagonal=0: _np.tril(value, diagonal=diagonal)
torch.eye = lambda n, m=None, dtype=_np.float32, **_kwargs: _np.eye(n, m if m is not None else n, dtype=dtype)
torch.linspace = lambda start, end, steps, **_kwargs: _np.linspace(start, end, steps)
torch.exp = _np.exp
torch.log = _np.log
torch.sin = _np.sin
torch.cos = _np.cos
torch.sqrt = _np.sqrt
torch.pow = _np.power
torch.abs = _np.abs
torch.clamp = lambda value, min=None, max=None: _np.minimum(_np.maximum(value, -_np.inf if min is None else min), _np.inf if max is None else max)
torch.maximum = _np.maximum
torch.minimum = _np.minimum
torch.sum = _sum
torch.mean = _mean
torch.amax = _amax
torch.amin = _amin
torch.max = _amax
torch.min = _amin
torch.norm = _norm
torch.argmax = lambda value, dim=None, keepdim=False: _np.argmax(value, axis=_axis(dim))
torch.argmin = lambda value, dim=None, keepdim=False: _np.argmin(value, axis=_axis(dim))
torch.matmul = _np.matmul
torch.mm = _np.matmul
torch.bmm = _np.matmul
torch.transpose = lambda value, dim0, dim1: _np.swapaxes(value, dim0, dim1)
torch.permute = lambda value, dims: _np.transpose(value, axes=tuple(dims))
torch.reshape = _np.reshape
torch.flatten = _flatten
torch.unsqueeze = lambda value, dim: _np.expand_dims(value, axis=dim)
torch.squeeze = lambda value, dim=None: _np.squeeze(value, axis=dim)
torch.cat = lambda values, dim=0: _np.concatenate(values, axis=dim)
torch.stack = lambda values, dim=0: _np.stack(values, axis=dim)
torch.where = _np.where
torch.broadcast_to = _np.broadcast_to
torch.isfinite = _np.isfinite
torch.is_floating_point = lambda value: _np.issubdtype(_np.asarray(value).dtype, _np.floating)
torch.all = lambda value, dim=None, keepdim=False: _np.all(value, axis=_axis(dim), keepdims=keepdim)
torch.any = lambda value, dim=None, keepdim=False: _np.any(value, axis=_axis(dim), keepdims=keepdim)
torch.unique = lambda value, sorted=True, **_kwargs: _np.unique(value)
torch.argsort = lambda value, dim=-1, descending=False, stable=False: _np.argsort(-value if descending else value, axis=dim, kind='stable' if stable else None)
torch.topk = _topk
torch.gather = _gather
torch.index_select = lambda value, dim, index: _np.take(value, _np.asarray(index, dtype=_np.int64), axis=dim)
torch.softmax = _stable_softmax
torch.log_softmax = _log_softmax
torch.logsumexp = _logsumexp
torch.manual_seed = lambda seed: _np.random.seed(seed)
torch.no_grad = _NullContext
torch.inference_mode = _NullContext

_nn = _types.ModuleType('torch.nn')
_nn.__path__ = []
_functional = _types.ModuleType('torch.nn.functional')
_functional.softmax = _stable_softmax
_functional.log_softmax = _log_softmax
_functional.cross_entropy = _cross_entropy
_nn.functional = _functional
torch.nn = _nn

_linalg = _types.ModuleType('torch.linalg')
_linalg.norm = _norm
torch.linalg = _linalg

_sys.modules['torch'] = torch
_sys.modules['torch.nn'] = _nn
_sys.modules['torch.nn.functional'] = _functional
_sys.modules['torch.linalg'] = _linalg
`;
