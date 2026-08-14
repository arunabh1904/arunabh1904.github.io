/**
 * A deliberately small PyTorch-shaped runtime for the browser exercises.
 *
 * Pyodide does not ship the compiled PyTorch runtime. The snippets still use
 * real PyTorch names and tensor idioms, but this compatibility layer maps the
 * subset used by the practice problems to NumPy arrays. It is not a drop-in
 * replacement for PyTorch: module and data-loader support is pedagogical, and
 * there is no autograd engine, accelerator runtime, compiler, or distributed backend.
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

    def dim(self):
        return int(self.ndim)

    def ndimension(self):
        return int(self.ndim)

    def cpu(self):
        return self

    def numpy(self):
        return _np.asarray(self)

    def float(self):
        return self.to(dtype=_np.float32)

    def long(self):
        return self.to(dtype=_np.int64)

    def to(self, dtype=None, device=None, **_kwargs):
        if dtype is None or _np.dtype(dtype) == self.dtype:
            return self
        return _tensor(self, dtype=dtype, copy=True)

    def view(self, *shape):
        return self.reshape(*shape)

    def permute(self, *dims):
        # Match torch.Tensor.permute(*dims), including its variadic dimension API.
        return _wrap(_np.transpose(self, axes=tuple(int(dim) for dim in dims)))

    def transpose(self, dim0, dim1):
        # Tensor.transpose swaps exactly two axes; NumPy's transpose expects all axes.
        return _wrap(_np.swapaxes(self, int(dim0), int(dim1)))

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

def _sort(value, dim=-1, descending=False, stable=False):
    value = _np.asarray(value)
    order = _np.argsort(
        -value if descending else value,
        axis=dim,
        kind='stable' if stable else None,
    )
    values = _np.take_along_axis(value, order, axis=dim)
    return _types.SimpleNamespace(values=_wrap(values), indices=_wrap(order))

def _randint(low, high=None, size=None, dtype=_np.int64, **_kwargs):
    if high is None:
        low, high = 0, low
    return _tensor(_np.random.randint(low, high, size=_shape(size), dtype=dtype), copy=False)

def _randperm(n, dtype=_np.int64, **_kwargs):
    return _tensor(_np.random.permutation(int(n)).astype(dtype), copy=False)

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
torch.randint = _randint
torch.randperm = _randperm
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
torch.atan2 = _np.arctan2
torch.sqrt = _np.sqrt
torch.pow = _np.power
torch.abs = _np.abs
torch.clamp = lambda value, min=None, max=None: _np.minimum(_np.maximum(value, -_np.inf if min is None else min), _np.inf if max is None else max)
torch.maximum = _np.maximum
torch.minimum = _np.minimum
torch.sum = _sum
torch.mean = _mean
torch.cumsum = lambda value, dim: _wrap(_np.cumsum(value, axis=dim))
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
torch.einsum = lambda equation, *operands: _wrap(_np.einsum(equation, *operands))
torch.outer = lambda left, right: _wrap(_np.outer(left, right))
torch.dot = lambda left, right: _wrap(_np.dot(left, right))
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
torch.sort = _sort
torch.gather = _gather
torch.index_select = lambda value, dim, index: _wrap(_np.take(value, _np.asarray(index, dtype=_np.int64), axis=dim))
torch.numel = lambda value: int(_np.asarray(value).size)
torch.equal = lambda left, right: bool(_np.array_equal(left, right))
torch.allclose = lambda left, right, rtol=1e-5, atol=1e-8, **_kwargs: bool(_np.allclose(left, right, rtol=rtol, atol=atol))
torch.isclose = lambda left, right, rtol=1e-5, atol=1e-8, **_kwargs: _wrap(_np.isclose(left, right, rtol=rtol, atol=atol))
torch.nan_to_num = lambda value, nan=0.0, posinf=None, neginf=None: _wrap(_np.nan_to_num(value, nan=nan, posinf=posinf, neginf=neginf))
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

class _Parameter(_CompatTensor):
    def __new__(cls, value, requires_grad=True):
        obj = _np.array(value, copy=True).view(cls)
        obj._compat_requires_grad = bool(requires_grad)
        return obj

    def __array_finalize__(self, source):
        self._compat_requires_grad = getattr(source, '_compat_requires_grad', True)

    @property
    def requires_grad(self):
        return self._compat_requires_grad

    def requires_grad_(self, requires_grad=True):
        self._compat_requires_grad = bool(requires_grad)
        return self

class _Module:
    def __init__(self):
        object.__setattr__(self, '_parameters', {})
        object.__setattr__(self, '_buffers', {})
        object.__setattr__(self, '_non_persistent_buffers', set())
        object.__setattr__(self, '_modules', {})
        object.__setattr__(self, 'training', True)

    def __setattr__(self, name, value):
        if name.startswith('_') or '_modules' not in self.__dict__:
            object.__setattr__(self, name, value)
            return
        if isinstance(value, _Parameter):
            self._parameters[name] = value
            self._modules.pop(name, None)
        elif isinstance(value, _Module):
            self._modules[name] = value
            self._parameters.pop(name, None)
        object.__setattr__(self, name, value)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *_args, **_kwargs):
        raise NotImplementedError

    def register_parameter(self, name, parameter):
        setattr(self, name, parameter)

    def register_buffer(self, name, tensor, persistent=True):
        tensor = None if tensor is None else _tensor(tensor, copy=False)
        self._buffers[name] = tensor
        if not persistent:
            self._non_persistent_buffers.add(name)
        object.__setattr__(self, name, tensor)

    def add_module(self, name, module):
        setattr(self, name, module)

    def named_parameters(self, prefix='', recurse=True):
        for name, parameter in self._parameters.items():
            if parameter is not None:
                yield (prefix + ('.' if prefix else '') + name, parameter)
        if recurse:
            for module_name, module in self._modules.items():
                child_prefix = prefix + ('.' if prefix else '') + module_name
                yield from module.named_parameters(child_prefix, recurse=True)

    def parameters(self, recurse=True):
        for _, parameter in self.named_parameters(recurse=recurse):
            yield parameter

    def named_buffers(self, prefix='', recurse=True):
        for name, buffer in self._buffers.items():
            if buffer is not None:
                yield (prefix + ('.' if prefix else '') + name, buffer)
        if recurse:
            for module_name, module in self._modules.items():
                child_prefix = prefix + ('.' if prefix else '') + module_name
                yield from module.named_buffers(child_prefix, recurse=True)

    def buffers(self, recurse=True):
        for _, buffer in self.named_buffers(recurse=recurse):
            yield buffer

    def named_children(self):
        yield from self._modules.items()

    def children(self):
        yield from self._modules.values()

    def named_modules(self, memo=None, prefix=''):
        memo = set() if memo is None else memo
        if id(self) in memo:
            return
        memo.add(id(self))
        yield prefix, self
        for name, module in self._modules.items():
            child_prefix = prefix + ('.' if prefix else '') + name
            yield from module.named_modules(memo, child_prefix)

    def modules(self):
        for _, module in self.named_modules():
            yield module

    def get_submodule(self, target):
        module = self
        if target:
            for atom in target.split('.'):
                module = getattr(module, atom)
                if not isinstance(module, _Module):
                    raise AttributeError(target)
        return module

    def train(self, mode=True):
        self.training = bool(mode)
        for module in self.children():
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def requires_grad_(self, requires_grad=True):
        for parameter in self.parameters():
            parameter.requires_grad_(requires_grad)
        return self

    def state_dict(self):
        state = {}
        for name, parameter in self.named_parameters():
            state[name] = parameter.clone()
        for name, buffer in self.named_buffers():
            owner, _, local_name = name.rpartition('.')
            module = self.get_submodule(owner) if owner else self
            if local_name not in module._non_persistent_buffers:
                state[name] = buffer.clone()
        return state

    def load_state_dict(self, state_dict, strict=True):
        live = dict(self.named_parameters()) | dict(self.named_buffers())
        missing = [name for name in live if name not in state_dict]
        unexpected = [name for name in state_dict if name not in live]
        for name, value in state_dict.items():
            if name in live:
                live[name].copy_(value)
        if strict and (missing or unexpected):
            raise RuntimeError('missing=' + repr(missing) + ', unexpected=' + repr(unexpected))
        return _types.SimpleNamespace(missing_keys=missing, unexpected_keys=unexpected)

    def to(self, *_args, **_kwargs):
        return self

    def apply(self, fn):
        for child in self.children():
            child.apply(fn)
        fn(self)
        return self

class _ModuleList(_Module):
    def __init__(self, modules=None):
        super().__init__()
        for module in modules or []:
            self.append(module)

    def append(self, module):
        self.add_module(str(len(self._modules)), module)
        return self

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def __getitem__(self, index):
        return list(self._modules.values())[index]

class _ModuleDict(_Module):
    def __init__(self, modules=None):
        super().__init__()
        for name, module in (modules or {}).items():
            self.add_module(name, module)

    def __getitem__(self, key):
        return self._modules[key]

    def __iter__(self):
        return iter(self._modules)

    def items(self):
        return self._modules.items()

class _ParameterList(_Module):
    def __init__(self, values=None):
        super().__init__()
        for value in values or []:
            self.append(value)

    def append(self, value):
        parameter = value if isinstance(value, _Parameter) else _Parameter(value)
        self.register_parameter(str(len(self._parameters)), parameter)
        return self

    def __len__(self):
        return len(self._parameters)

    def __iter__(self):
        return iter(self._parameters.values())

    def __getitem__(self, index):
        return list(self._parameters.values())[index]

class _ParameterDict(_Module):
    def __init__(self, values=None):
        super().__init__()
        for name, value in (values or {}).items():
            parameter = value if isinstance(value, _Parameter) else _Parameter(value)
            self.register_parameter(name, parameter)

    def __getitem__(self, key):
        return self._parameters[key]

    def items(self):
        return self._parameters.items()

class _Sequential(_Module):
    def __init__(self, *modules):
        super().__init__()
        for module in modules:
            self.add_module(str(len(self._modules)), module)

    def forward(self, value):
        for module in self._modules.values():
            value = module(value)
        return value

    def __iter__(self):
        return iter(self._modules.values())

class _Linear(_Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        scale = 1.0 / max(1, int(in_features)) ** 0.5
        self.weight = _Parameter(_np.random.uniform(-scale, scale, (out_features, in_features)))
        self.bias = _Parameter(_np.random.uniform(-scale, scale, out_features)) if bias else None

    def forward(self, value):
        output = _np.matmul(value, self.weight.T)
        if self.bias is not None:
            output = output + self.bias
        return _wrap(output)

class _ReLU(_Module):
    def forward(self, value):
        return _wrap(_np.maximum(value, 0))

class _Dropout(_Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = float(p)

    def forward(self, value):
        if not self.training or self.p == 0:
            return value
        keep = _np.random.random(_np.asarray(value).shape) >= self.p
        return _wrap(_np.asarray(value) * keep / (1.0 - self.p))

class _Identity(_Module):
    def forward(self, value):
        return value

_nn.Module = _Module
_nn.Parameter = _Parameter
_nn.ModuleList = _ModuleList
_nn.ModuleDict = _ModuleDict
_nn.ParameterList = _ParameterList
_nn.ParameterDict = _ParameterDict
_nn.Sequential = _Sequential
_nn.Linear = _Linear
_nn.ReLU = _ReLU
_nn.Dropout = _Dropout
_nn.Identity = _Identity
_functional = _types.ModuleType('torch.nn.functional')
_functional.softmax = _stable_softmax
_functional.log_softmax = _log_softmax
_functional.cross_entropy = _cross_entropy
_functional.relu = lambda value, inplace=False: _wrap(_np.maximum(value, 0))
_functional.linear = lambda value, weight, bias=None: _wrap(_np.matmul(value, weight.T) + (0 if bias is None else bias))
_nn.functional = _functional
torch.nn = _nn

class _Dataset:
    pass

class _IterableDataset(_Dataset):
    pass

class _TensorDataset(_Dataset):
    def __init__(self, *tensors):
        if tensors and any(len(tensor) != len(tensors[0]) for tensor in tensors[1:]):
            raise ValueError('all tensors must have the same first dimension')
        self.tensors = tensors

    def __len__(self):
        return 0 if not self.tensors else len(self.tensors[0])

    def __getitem__(self, index):
        return tuple(_wrap(tensor[index]) for tensor in self.tensors)

def _default_collate(batch):
    first = batch[0]
    if isinstance(first, tuple):
        return tuple(_wrap(_np.stack(values, axis=0)) for values in zip(*batch))
    if isinstance(first, dict):
        return {key: _wrap(_np.stack([item[key] for item in batch], axis=0)) for key in first}
    return _wrap(_np.stack(batch, axis=0))

class _DataLoader:
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        collate_fn=None,
        drop_last=False,
        **_kwargs,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.collate_fn = collate_fn or _default_collate
        self.drop_last = drop_last

    def __iter__(self):
        if self.batch_sampler is not None:
            batches = self.batch_sampler
        else:
            indices = list(self.sampler) if self.sampler is not None else list(range(len(self.dataset)))
            if self.shuffle:
                _np.random.shuffle(indices)
            if self.batch_size is None:
                for index in indices:
                    yield self.dataset[index]
                return
            batches = [indices[start:start + self.batch_size] for start in range(0, len(indices), self.batch_size)]
        for indices in batches:
            if self.drop_last and len(indices) < self.batch_size:
                continue
            yield self.collate_fn([self.dataset[index] for index in indices])

    def __len__(self):
        if self.batch_sampler is not None:
            return len(self.batch_sampler)
        if self.batch_size is None:
            return len(self.dataset)
        size = len(self.dataset) // self.batch_size
        return size if self.drop_last or len(self.dataset) % self.batch_size == 0 else size + 1

_utils = _types.ModuleType('torch.utils')
_utils.__path__ = []
_data = _types.ModuleType('torch.utils.data')
_data.Dataset = _Dataset
_data.IterableDataset = _IterableDataset
_data.TensorDataset = _TensorDataset
_data.DataLoader = _DataLoader
_data.default_collate = _default_collate
_utils.data = _data
torch.utils = _utils

_linalg = _types.ModuleType('torch.linalg')
_linalg.norm = _norm
torch.linalg = _linalg

_sys.modules['torch'] = torch
_sys.modules['torch.nn'] = _nn
_sys.modules['torch.nn.functional'] = _functional
_sys.modules['torch.linalg'] = _linalg
_sys.modules['torch.cuda'] = _cuda
_sys.modules['torch.utils'] = _utils
_sys.modules['torch.utils.data'] = _data
`;
