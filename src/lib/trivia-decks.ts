import {
  pytorchAtomicSpecs,
  type AtomicTriviaSpec,
} from './trivia-atomic-specs';
import { practicalPythonTriviaDeck } from './python-practical-trivia';
import { triviaExplanations } from './trivia-explanations';

export interface TriviaCard {
  id: string;
  topic: string;
  question: string;
  answer: string;
  acceptedAnswers?: string[];
  explanation?: string;
  code?: string;
  detail?: string;
}

export interface TriviaDeckData {
  id: string;
  title: string;
  cards: TriviaCard[];
}

interface TriviaSourceCard {
  id: string;
  topic: string;
  question: string;
  answer: string;
  code?: string;
  detail?: string;
}

interface TriviaSourceDeck {
  id: string;
  title: string;
  cards: TriviaSourceCard[];
}

const pytorchTriviaSourceDeck: TriviaSourceDeck = {
  id: 'pytorch-interview-trivia-v1',
  title: 'PyTorch interview trivia',
  cards: [
    {
      id: 'torch-tensor-metadata',
      topic: 'Tensors',
      question: 'Which properties define how a dense strided tensor maps indices to memory?',
      answer: 'Storage, storage offset, shape, and strides define the mapping. Dtype determines element interpretation, while device and layout determine where and how the storage is managed.',
    },
    {
      id: 'torch-view-reshape',
      topic: 'Tensors',
      question: 'How do `view` and `reshape` differ?',
      answer: '`view` requires strides compatible with the requested shape and fails otherwise. `reshape` returns a view when possible but may silently allocate a copy, so callers should not assume aliasing.',
    },
    {
      id: 'torch-transpose-contiguous',
      topic: 'Tensors',
      question: 'Does `transpose` copy tensor data?',
      answer: 'Usually no. It returns a view with reordered shape and strides. Some later kernels require a particular layout and may need `contiguous()`, which copies only when the requested memory format is absent.',
    },
    {
      id: 'torch-clone-detach',
      topic: 'Tensors',
      question: 'What is the difference between `clone()` and `detach()`?',
      answer: '`clone()` copies values into new storage but preserves the autograd connection. `detach()` cuts the incoming graph edge but usually shares storage. Use `x.detach().clone()` for an independent snapshot.',
    },
    {
      id: 'torch-broadcasting',
      topic: 'Tensors',
      question: 'What is PyTorch’s broadcasting rule?',
      answer: 'Compare dimensions from the right. Aligned dimensions must be equal or one must be `1`; missing leading dimensions act like `1`. Expansion can use zero strides instead of copying the smaller input.',
    },
    {
      id: 'torch-cat-stack',
      topic: 'Tensors',
      question: 'How do `torch.cat` and `torch.stack` change shape?',
      answer: '`cat` joins tensors along an existing axis and preserves rank. `stack` requires equal input shapes and inserts a new axis, increasing rank by one.',
    },
    {
      id: 'torch-indexing-copy',
      topic: 'Tensors',
      question: 'Which tensor indexing operations usually return views versus copies?',
      answer: 'Basic indexing with integers and slices usually returns a view. Advanced indexing with integer tensors, lists, or boolean masks returns selected values in new storage; assignment through that syntax still writes to the destination.',
    },
    {
      id: 'torch-inplace-promotion',
      topic: 'Tensors',
      question: 'Why can a mixed-dtype expression work out of place but fail in place?',
      answer: 'An out-of-place operation can allocate the promoted result dtype. An in-place operation must write back into the destination dtype, so PyTorch rejects conversions that cannot satisfy that contract safely.',
    },
    {
      id: 'torch-to-noop',
      topic: 'Tensors',
      question: 'Does `x.to(device=x.device, dtype=x.dtype)` always copy?',
      answer: 'No. If no conversion is needed, `to` can return `x` itself. Pass `copy=True` only when a distinct tensor is required.',
    },
    {
      id: 'torch-empty',
      topic: 'Tensors',
      question: 'What values does `torch.empty` contain?',
      answer: 'Uninitialized values from the allocated memory. The code must write every element before reading it; `empty` is a performance-oriented allocation primitive, not a zero-filled tensor.',
    },
    {
      id: 'torch-requires-grad-recording',
      topic: 'Autograd',
      question: 'When does autograd record an operation in normal grad mode?',
      answer: 'When at least one input requires gradients and the operation is differentiable. The output receives graph history through `grad_fn`; leaf tensors supplied by the user normally have no `grad_fn`.',
    },
    {
      id: 'torch-leaf-grad',
      topic: 'Autograd',
      question: 'Which tensors receive a populated `.grad` after backward?',
      answer: 'Leaf tensors with `requires_grad=True` accumulate gradients by default. Intermediates need `retain_grad()` if their `.grad` field must be inspected. This default keeps retained gradient state focused on optimization inputs.',
    },
    {
      id: 'torch-grad-accumulation',
      topic: 'Autograd',
      question: 'Does `backward()` overwrite existing parameter gradients?',
      answer: 'No. It accumulates into `.grad`. Clear gradients at the intended accumulation boundary, commonly with `optimizer.zero_grad(set_to_none=True)`. Accumulation enables larger effective batches but makes stale gradients dangerous.',
    },
    {
      id: 'torch-none-zero-grad',
      topic: 'Autograd',
      question: 'Why can a gradient of `None` be more informative than a zero tensor?',
      answer: '`None` means no gradient has been accumulated for that parameter. A zero tensor means a differentiable path reached it and produced zero. Optimizers may also treat missing and zero gradients differently.',
    },
    {
      id: 'torch-nonscalar-backward',
      topic: 'Autograd',
      question: 'Why does a non-scalar tensor need an argument to `backward()`?',
      answer: 'The argument supplies the upstream vector for a vector-Jacobian product. A scalar loss has the unambiguous default upstream gradient of one.',
    },
    {
      id: 'torch-no-grad-inference',
      topic: 'Autograd',
      question: 'How does `torch.inference_mode()` differ from `torch.no_grad()`?',
      answer: 'Both prevent backward-graph recording. Inference mode also removes more autograd bookkeeping and can be faster, but tensors created there cannot later participate normally in a graph recorded by autograd.',
    },
    {
      id: 'torch-eval-no-grad',
      topic: 'Autograd',
      question: 'Does `model.eval()` disable gradient tracking?',
      answer: 'No. `eval()` changes the `training` flag used by modules such as Dropout and BatchNorm. Gradient mode is orthogonal; evaluation usually combines `model.eval()` with `torch.inference_mode()` or `torch.no_grad()`.',
    },
    {
      id: 'torch-inplace-autograd',
      topic: 'Autograd',
      question: 'Why can an in-place operation trigger an autograd version-counter error?',
      answer: 'Backward may need a tensor value saved during forward. If an in-place write changes that storage before backward, the saved value is no longer valid, so autograd detects the version mismatch.',
    },
    {
      id: 'torch-retain-create-graph',
      topic: 'Autograd',
      question: 'How do `retain_graph=True` and `create_graph=True` differ?',
      answer: '`retain_graph` keeps the current forward graph available for another backward traversal. `create_graph` records the derivative computation itself so higher-order derivatives can be taken.',
    },
    {
      id: 'torch-grad-vs-backward',
      topic: 'Autograd',
      question: '`torch.autograd.grad` versus `Tensor.backward`: when does each fit?',
      answer: '`backward` accumulates gradients into graph leaves, matching optimizer training loops. `autograd.grad` returns gradients for named inputs directly and is useful for functional gradients, penalties, and higher-order work.',
    },
    {
      id: 'torch-parameter-registration',
      topic: 'Modules & state',
      question: 'What makes `nn.Parameter` different from a tensor with `requires_grad=True`?',
      answer: 'Assigning an `nn.Parameter` as a module attribute registers it, so it appears in `parameters()` and `state_dict()`. A plain tensor is not registered as a parameter merely because it requires gradients.',
    },
    {
      id: 'torch-module-list',
      topic: 'Modules & state',
      question: 'Why use `nn.ModuleList` instead of a Python list of submodules?',
      answer: '`ModuleList` registers its children. A plain list can hold working modules, but the parent will not discover their parameters, device moves, training-mode changes, or state unless you register them another way.',
    },
    {
      id: 'torch-parameter-buffer',
      topic: 'Modules & state',
      question: 'Parameter versus buffer: what is the contract?',
      answer: 'Parameters are trainable model state exposed to optimizers. Buffers are non-parameter state that follows device moves and normally appears in `state_dict`, such as BatchNorm running statistics.',
    },
    {
      id: 'torch-state-dict',
      topic: 'Modules & state',
      question: 'What does a module `state_dict` contain?',
      answer: 'A mapping of registered parameters and persistent buffers. It does not serialize arbitrary Python attributes or the model’s class definition, so code must reconstruct the module before loading weights.',
    },
    {
      id: 'torch-load-state-strict',
      topic: 'Modules & state',
      question: 'What does `load_state_dict(..., strict=True)` check?',
      answer: 'It requires checkpoint keys to match the module’s expected parameter and persistent-buffer keys. With `strict=False`, inspect the returned missing and unexpected keys instead of silently assuming a complete load.',
    },
    {
      id: 'torch-forward-call',
      topic: 'Modules & state',
      question: 'Why call `model(x)` instead of `model.forward(x)`?',
      answer: '`nn.Module.__call__` wraps `forward` with framework behavior such as hooks and compiled-call machinery. Directly calling `forward` bypasses that wrapper. Treat `model(...)` as the module’s public invocation boundary.',
    },
    {
      id: 'torch-cross-entropy-logits',
      topic: 'Training',
      question: 'Should `CrossEntropyLoss` receive softmax probabilities?',
      answer: 'No. It expects unnormalized logits and internally performs the stable log-softmax plus negative log-likelihood calculation. Applying softmax first loses numerical stability and changes the intended interface.',
    },
    {
      id: 'torch-cross-entropy-target',
      topic: 'Training',
      question: 'What target dtype and shape does class-index cross entropy expect?',
      answer: 'For logits shaped `(N, C, ...)`, class-index targets normally use integer `torch.long` values shaped `(N, ...)`, with each value in `[0, C)`. Probability targets use the same shape as logits and a floating dtype.',
    },
    {
      id: 'torch-bce-logits',
      topic: 'Training',
      question: 'Why prefer `BCEWithLogitsLoss` over `sigmoid` followed by binary cross entropy?',
      answer: 'The fused loss uses a numerically stable formulation that avoids materializing extreme probabilities. It expects logits and floating targets, commonly with the same shape.',
    },
    {
      id: 'torch-training-order',
      topic: 'Training',
      question: 'What is the standard optimizer-step order?',
      answer: 'Clear gradients, run forward, compute loss, call backward, then step the optimizer. Gradient clipping, logging, schedulers, and accumulation add boundaries that must be placed deliberately.',
      code: 'optimizer.zero_grad(set_to_none=True)\nloss = criterion(model(inputs), targets)\nloss.backward()\noptimizer.step()',
    },
    {
      id: 'torch-amp',
      topic: 'Training',
      question: 'What jobs do autocast and gradient scaling perform?',
      answer: 'Autocast chooses lower precision for eligible operations while keeping sensitive operations in safer dtypes. Gradient scaling helps FP16 gradients avoid underflow; BF16 often does not need scaling because it has FP32-like exponent range.',
    },
    {
      id: 'torch-clip-unscale',
      topic: 'Training',
      question: 'When using a gradient scaler, when should gradients be clipped?',
      answer: 'Unscale the optimizer’s gradients first, then clip, then call the scaler’s step and update. Clipping scaled gradients applies the wrong threshold.',
    },
    {
      id: 'torch-gradient-accumulation-loss',
      topic: 'Training',
      question: 'Why is loss often divided by the number of accumulation steps?',
      answer: 'Backward sums gradient contributions. Dividing each microbatch loss makes the accumulated gradient match the mean over the effective batch; omitting the division scales the gradient and changes the effective learning rate.',
    },
    {
      id: 'torch-checkpoint',
      topic: 'Training',
      question: 'What belongs in a resumable training checkpoint?',
      answer: 'At minimum: model state, optimizer state, current step or epoch, and scheduler or scaler state when used. Exact continuation may also require random-number-generator, sampler, and data-pipeline state.',
    },
    {
      id: 'torch-dataset-kinds',
      topic: 'Data loading',
      question: 'Map-style versus iterable-style dataset: what changes?',
      answer: 'A map-style dataset supports keyed indexing and usually length. An `IterableDataset` yields a stream and fits logs, remote sources, or generated data, but each worker must shard the stream to avoid duplicates.',
    },
    {
      id: 'torch-dataloader-workers',
      topic: 'Data loading',
      question: 'What does `DataLoader(num_workers > 0)` do?',
      answer: 'It loads and processes samples in subprocesses, which can overlap CPU input work with model execution. More workers are not always faster because startup, serialization, memory, and storage contention can dominate.',
    },
    {
      id: 'torch-pin-memory',
      topic: 'Data loading',
      question: 'What does `pin_memory=True` help with?',
      answer: 'Workers place CPU tensors in page-locked host memory, enabling faster and potentially asynchronous host-to-CUDA copies. Pair the later device transfer with `non_blocking=True` when the rest of the pipeline can overlap it.',
    },
    {
      id: 'torch-persistent-workers',
      topic: 'Data loading',
      question: 'What trade-off does `persistent_workers=True` make?',
      answer: 'It keeps worker processes and their dataset instances alive between epochs, reducing repeated startup cost while retaining their memory and other resources for longer.',
    },
    {
      id: 'torch-worker-cuda-tensors',
      topic: 'Data loading',
      question: 'Should multiprocess DataLoader workers normally return CUDA tensors?',
      answer: 'No. CUDA plus multiprocessing has subtle lifetime and initialization constraints. PyTorch recommends returning CPU tensors, optionally pinned, and moving batches to the accelerator in the training process.',
    },
    {
      id: 'torch-reproducibility',
      topic: 'Data loading',
      question: 'Does setting `torch.manual_seed` guarantee identical results everywhere?',
      answer: 'No. Reproducibility can depend on device, release, nondeterministic kernels, library seeds, worker seeds, and algorithm selection. Deterministic settings can reduce variance but may cost performance or lack an implementation.',
    },
    {
      id: 'torch-ddp-data-parallel',
      topic: 'Distributed',
      question: 'Why is `DistributedDataParallel` preferred over `nn.DataParallel`?',
      answer: 'DDP uses one process per device and synchronizes gradients efficiently without a single Python thread scattering work and gathering outputs. It also scales across machines.',
    },
    {
      id: 'torch-ddp-inputs',
      topic: 'Distributed',
      question: 'Does DDP automatically split the input batch across ranks?',
      answer: 'No. DDP replicates the module and synchronizes gradients. The program must partition input data, commonly with `DistributedSampler` or explicit iterable-dataset sharding.',
    },
    {
      id: 'torch-distributed-sampler-epoch',
      topic: 'Distributed',
      question: 'Why call `DistributedSampler.set_epoch(epoch)`?',
      answer: 'It changes the deterministic shuffle seed each epoch while keeping ranks coordinated. Without it, the sampler can produce the same ordering every epoch.',
    },
    {
      id: 'torch-ddp-unused',
      topic: 'Distributed',
      question: 'Why can conditional model branches cause a DDP reduction error?',
      answer: 'A rank may leave some registered parameters unused, so expected gradient reductions do not occur. `find_unused_parameters=True` can detect this at overhead, but stable graph design is usually better.',
    },
    {
      id: 'torch-fsdp',
      topic: 'Distributed',
      question: 'What memory problem does Fully Sharded Data Parallel address?',
      answer: 'FSDP shards parameters, gradients, and optimizer state across ranks, materializing fuller parameter views only when computation needs them. DDP normally keeps a complete model replica on every rank.',
    },
    {
      id: 'torch-compile',
      topic: 'Performance',
      question: 'What does `torch.compile` try to optimize?',
      answer: 'It captures regions of Python-level PyTorch execution, transforms the forward and backward graphs, and generates optimized kernels. Unsupported Python behavior can create graph breaks, and changing shapes or guards can trigger recompilation.',
    },
    {
      id: 'torch-cuda-async-timing',
      topic: 'Performance',
      question: 'Why can ordinary wall-clock timing underreport CUDA operation time?',
      answer: 'CUDA launches are asynchronous with respect to the host, so the timer may stop before device work completes. Synchronize around the measured region or use CUDA events.',
    },
    {
      id: 'torch-allocated-reserved',
      topic: 'Performance',
      question: 'What is the difference between CUDA memory allocated and reserved?',
      answer: 'Allocated memory backs live tensors. Reserved memory is held by PyTorch’s caching allocator and includes allocated blocks plus reusable free blocks; it can remain high after tensors are deleted.',
    },
    {
      id: 'torch-empty-cache',
      topic: 'Performance',
      question: 'Does `torch.cuda.empty_cache()` free memory held by live tensors?',
      answer: 'No. It releases unused cached blocks for other applications to see, but live tensor storage remains allocated. Calling it every iteration can add synchronization and allocation overhead.',
    },
    {
      id: 'torch-item-sync',
      topic: 'Performance',
      question: 'Why can frequent `.item()` calls hurt accelerator throughput?',
      answer: 'Converting a device tensor to a Python scalar requires the host to observe the result, which can synchronize pending accelerator work and break overlap.',
    },
    {
      id: 'torch-optimizer-memory',
      topic: 'Performance',
      question: 'Why can optimizer state use more memory than model weights?',
      answer: 'Optimizers may keep one or more full-sized tensors per parameter, such as Adam’s first and second moments, often in higher precision. Gradients, activations, and temporary workspaces add further peaks.',
    },
    {
      id: 'torch-activation-checkpoint',
      topic: 'Performance',
      question: 'What trade-off does activation checkpointing make?',
      answer: 'Activation checkpointing saves memory by discarding selected forward intermediates and recomputing them during backward. The current API recommends passing `use_reentrant=False` explicitly.',
    },
    {
      id: 'torch-ddp-no-sync',
      topic: 'Distributed',
      question: 'How should DDP gradient accumulation avoid redundant synchronization?',
      answer: '`DDP.no_sync()` defers gradient synchronization inside its context. The forward pass must also occur inside the context, and the first forward-backward outside it performs the synchronization.',
    },
    {
      id: 'torch-load-weights-only',
      topic: 'Modules & state',
      question: 'How should tensor checkpoints from outside the trust boundary be loaded?',
      answer: 'Use `torch.load(..., weights_only=True)` and never treat pickle-based loading as safe for untrusted data. The restricted loader accepts tensors, primitive types, dictionaries, and explicitly allowlisted types.',
    },
    {
      id: 'torch-load-map-location',
      topic: 'Modules & state',
      question: 'Why load a GPU checkpoint with `map_location="cpu"`?',
      answer: 'Mapping storages to CPU avoids restoring them directly onto their saved CUDA devices and can prevent a GPU-memory surge. Move the reconstructed model to its target device after loading state.',
    },
    {
      id: 'torch-anomaly-detection',
      topic: 'Autograd',
      question: 'When should autograd anomaly detection be enabled?',
      answer: 'Use anomaly detection to debug the forward operation that produced a failing backward or NaN. It adds substantial bookkeeping and synchronization overhead, so it should not remain enabled in normal training.',
    },
    {
      id: 'torch-zero-grad-none',
      topic: 'Training',
      question: 'Why use `zero_grad(set_to_none=True)`?',
      answer: 'Setting gradients to `None` can reduce memory writes and lets the optimizer distinguish parameters that received no gradient from parameters whose gradient is zero.',
    },
    {
      id: 'torch-softmax-dim',
      topic: 'Tensors',
      question: 'Why must softmax specify the class dimension?',
      answer: 'Softmax normalizes along the selected dimension. Choosing the wrong axis produces probabilities that sum to one over the wrong objects while preserving a plausible tensor shape.',
    },
  ],
};

function buildAtomicDeck(
  sourceDeck: TriviaSourceDeck,
  specs: AtomicTriviaSpec[],
): TriviaDeckData {
  const sourceCards = new Map(sourceDeck.cards.map((card) => [card.id, card]));
  const coveredSourceIds = new Set(specs.map((spec) => spec.sourceId));
  const uncoveredSourceIds = sourceDeck.cards
    .map((card) => card.id)
    .filter((id) => !coveredSourceIds.has(id));

  if (uncoveredSourceIds.length > 0) {
    throw new Error(`Atomic trivia specs missing source cards: ${uncoveredSourceIds.join(', ')}`);
  }

  const cards = specs.flatMap((spec): TriviaCard[] => {
    const source = sourceCards.get(spec.sourceId);
    if (!source) throw new Error(`Unknown trivia source card: ${spec.sourceId}`);
    const explanation = triviaExplanations[spec.id];
    if (!explanation) throw new Error(`Trivia card missing a focused explanation: ${spec.id}`);

    const conceptCard: TriviaCard = {
      id: spec.id,
      topic: source.topic,
      question: spec.question,
      answer: spec.answer,
      acceptedAnswers: spec.acceptedAnswers,
      explanation,
      detail: source.detail ?? source.answer,
    };

    if (!spec.code) return [conceptCard];
    if (!spec.codeQuestion || !spec.codeAnswer) {
      throw new Error(`Code trivia spec missing a question or answer: ${spec.id}`);
    }
    const codeId = `${spec.id}-code`;
    const codeExplanation = triviaExplanations[codeId];
    if (!codeExplanation) {
      throw new Error(`Code trivia card missing a focused explanation: ${codeId}`);
    }

    return [
      conceptCard,
      {
        id: codeId,
        topic: 'Code scenarios',
        question: spec.codeQuestion,
        answer: spec.codeAnswer,
        acceptedAnswers: spec.codeAcceptedAnswers,
        explanation: codeExplanation,
        code: spec.code,
        detail: source.detail ?? source.answer,
      },
    ];
  });

  const ids = cards.map((card) => card.id);
  if (new Set(ids).size !== ids.length) {
    throw new Error(`Atomic trivia card IDs must be unique in ${sourceDeck.title}`);
  }

  return {
    id: sourceDeck.id.replace(/-v\d+$/, '-v2'),
    title: sourceDeck.title,
    cards,
  };
}

export const pythonTriviaDeck = practicalPythonTriviaDeck;
export const pytorchTriviaDeck = buildAtomicDeck(pytorchTriviaSourceDeck, pytorchAtomicSpecs);
