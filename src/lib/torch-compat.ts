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

class _CompatStorage:
    def __init__(self, tensor):
        self.tensor = tensor

    def data_ptr(self):
        return _storage_data_ptr(self.tensor)

class _CompatTensor(_np.ndarray):
    def __new__(cls, value, dtype=None, copy=True):
        if copy:
            array = _np.array(value, dtype=dtype, copy=True)
        else:
            array = _np.asarray(value, dtype=dtype)
        return array.view(cls)

    def __array_finalize__(self, _source):
        pass

    @property
    def device(self):
        return 'cpu'

    @property
    def requires_grad(self):
        return False

    @property
    def is_cuda(self):
        return False

    def numel(self):
        return int(self.size)

    def element_size(self):
        return int(self.dtype.itemsize)

    def stride(self):
        return tuple(int(byte_stride // self.itemsize) for byte_stride in self.strides)

    def is_contiguous(self):
        return bool(self.flags['C_CONTIGUOUS'])

    def contiguous(self):
        return self if self.is_contiguous() else _tensor(self, copy=True)

    def clone(self):
        return _tensor(self, copy=True)

    def detach(self):
        return self

    def requires_grad_(self, _requires_grad=True):
        return self

    def to(self, dtype=None, device=None, **_kwargs):
        if dtype is None or _np.dtype(dtype) == self.dtype:
            return self
        return _tensor(self, dtype=dtype, copy=True)

    def view(self, *shape):
        return self.reshape(*shape)

    def add_(self, value):
        self[...] = _np.asarray(self) + value
        return self

    def sub_(self, value):
        self[...] = _np.asarray(self) - value
        return self

    def mul_(self, value):
        self[...] = _np.asarray(self) * value
        return self

    def div_(self, value):
        self[...] = _np.asarray(self) / value
        return self

    def zero_(self):
        self[...] = 0
        return self

    def fill_(self, value):
        self[...] = value
        return self

    def copy_(self, source):
        self[...] = _np.asarray(source, dtype=self.dtype)
        return self

    def data_ptr(self):
        return int(self.__array_interface__['data'][0])

    def storage_offset(self):
        return int((self.data_ptr() - _storage_data_ptr(self)) // self.itemsize)

    def untyped_storage(self):
        return _CompatStorage(self)

    def storage(self):
        return self.untyped_storage()

def _storage_root(value):
    root = value
    while isinstance(getattr(root, 'base', None), _np.ndarray):
        root = root.base
    return _np.asarray(root)

def _storage_data_ptr(value):
    return int(_storage_root(value).__array_interface__['data'][0])

def _tensor(value, dtype=None, copy=True):
    if isinstance(value, _CompatTensor) and dtype is None and not copy:
        return value
    return _CompatTensor(value, dtype=dtype, copy=copy)

def _wrap(value):
    if isinstance(value, _np.ndarray) and not isinstance(value, _CompatTensor):
        return value.view(_CompatTensor)
    return value

def _asarray(value, dtype=None):
    return _np.asarray(value, dtype=dtype)

def _shape(size):
    return (int(size),) if isinstance(size, (int, _np.integer)) else tuple(int(v) for v in size)

def _shape_args(sizes):
    return _shape(sizes[0]) if len(sizes) == 1 else tuple(int(v) for v in sizes)

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
    return _wrap(_np.where(
        denominator > 0,
        exponentiated / _safe_denominator(denominator),
        _np.zeros_like(exponentiated),
    ))

def _log_softmax(value, dim=-1):
    value = _asarray(value, dtype=_np.float64)
    finite = _np.isfinite(value)
    safe_value = _np.where(finite, value, 0.0)
    maximum = _np.max(safe_value, axis=dim, keepdims=True)
    shifted = safe_value - maximum
    denominator = _np.sum(_np.exp(shifted) * finite, axis=dim, keepdims=True)
    return _wrap(shifted - _np.log(_safe_denominator(denominator)))

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
    return _wrap(_np.arange(start, end, step, dtype=dtype))

def _flatten(value, start_dim=0, end_dim=-1):
    value = _np.asarray(value)
    if end_dim < 0:
        end_dim += value.ndim
    prefix = value.shape[:start_dim]
    suffix = value.shape[end_dim + 1:]
    middle = int(_np.prod(value.shape[start_dim:end_dim + 1]))
    return _wrap(value.reshape(prefix + (middle,) + suffix))

def _gather(value, dim, index):
    value = _np.asarray(value)
    index = _np.asarray(index, dtype=_np.int64)
    return _wrap(_np.take_along_axis(value, index, axis=dim))

def _topk(value, k, dim=-1, largest=True, sorted=True):
    value = _np.asarray(value)
    order = _np.argsort(-value if largest else value, axis=dim, kind='stable')
    indices = _np.take(order, _np.arange(k), axis=dim)
    values = _np.take_along_axis(value, indices, axis=dim)
    return _types.SimpleNamespace(values=_wrap(values), indices=_wrap(indices))

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
torch.Tensor = _CompatTensor
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
torch.as_tensor = lambda value, dtype=None, **_kwargs: _tensor(value, dtype=dtype, copy=False)
torch.tensor = lambda value, dtype=None, **_kwargs: _tensor(value, dtype=dtype, copy=True)
torch.from_numpy = lambda value: _tensor(value, copy=False)
torch.clone = lambda value: _tensor(value, copy=True)
torch.zeros = lambda *size, dtype=_np.float32, **_kwargs: _tensor(_np.zeros(_shape_args(size), dtype=dtype), copy=False)
torch.ones = lambda *size, dtype=_np.float32, **_kwargs: _tensor(_np.ones(_shape_args(size), dtype=dtype), copy=False)
torch.empty = lambda *size, dtype=_np.float32, **_kwargs: _tensor(_np.empty(_shape_args(size), dtype=dtype), copy=False)
torch.full = lambda size, fill_value, dtype=None, **_kwargs: _tensor(_np.full(_shape(size), fill_value, dtype=dtype), copy=False)
torch.rand = lambda *size, dtype=_np.float32, **_kwargs: _tensor(_np.random.rand(*_shape_args(size)).astype(dtype), copy=False)
torch.randn = lambda *size, dtype=_np.float32, **_kwargs: _tensor(_np.random.randn(*_shape_args(size)).astype(dtype), copy=False)
torch.zeros_like = lambda value, **_kwargs: _tensor(_np.zeros_like(value), copy=False)
torch.ones_like = lambda value, **_kwargs: _tensor(_np.ones_like(value), copy=False)
torch.empty_like = lambda value, **_kwargs: _tensor(_np.empty_like(value), copy=False)
torch.full_like = lambda value, fill_value, **_kwargs: _tensor(_np.full_like(value, fill_value), copy=False)
torch.arange = _arange
torch.tril = lambda value, diagonal=0: _wrap(_np.tril(value, diagonal=diagonal))
torch.eye = lambda n, m=None, dtype=_np.float32, **_kwargs: _tensor(_np.eye(n, m if m is not None else n, dtype=dtype), copy=False)
torch.linspace = lambda start, end, steps, **_kwargs: _tensor(_np.linspace(start, end, steps), copy=False)
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
torch.matmul = lambda left, right: _wrap(_np.matmul(left, right))
torch.mm = torch.matmul
torch.bmm = torch.matmul
torch.transpose = lambda value, dim0, dim1: _wrap(_np.swapaxes(value, dim0, dim1))
torch.permute = lambda value, dims: _wrap(_np.transpose(value, axes=tuple(dims)))
torch.reshape = lambda value, shape: _wrap(_np.reshape(value, _shape(shape)))
torch.flatten = _flatten
torch.unsqueeze = lambda value, dim: _wrap(_np.expand_dims(value, axis=dim))
torch.squeeze = lambda value, dim=None: _wrap(_np.squeeze(value, axis=dim))
torch.cat = lambda values, dim=0: _wrap(_np.concatenate(values, axis=dim))
torch.concat = torch.cat
torch.stack = lambda values, dim=0: _wrap(_np.stack(values, axis=dim))
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
torch.index_select = lambda value, dim, index: _wrap(_np.take(value, _np.asarray(index, dtype=_np.int64), axis=dim))
torch.numel = lambda value: int(_np.asarray(value).size)
torch.equal = lambda left, right: bool(_np.array_equal(left, right))
torch.softmax = _stable_softmax
torch.log_softmax = _log_softmax
torch.logsumexp = _logsumexp
torch.manual_seed = lambda seed: _np.random.seed(seed)
torch.no_grad = _NullContext
torch.inference_mode = _NullContext

class _Device(str):
    def __new__(cls, value):
        return str.__new__(cls, value)

    def __repr__(self):
        return "device(type='" + str(self) + "')"

torch.device = _Device

_cuda = _types.ModuleType('torch.cuda')
_cuda.is_available = lambda: False
torch.cuda = _cuda

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
_sys.modules['torch.cuda'] = _cuda
`;
