/**
 * The second teaching layer for each Python trivia card.
 *
 * The card-local explanation states the immediate rule. These details name the
 * mental model, boundary, or failure mode that makes the rule reusable.
 */
export const pythonTriviaDetails: Record<string, string> = {
  // References and values
  'python-aliasing-append': 'Python names point to objects. Assignment changes which object a name points to; it does not duplicate that object. Copy only when the two names must be allowed to evolve independently.',
  'python-mutation-vs-rebinding': 'A function receives another reference to the caller’s object. Mutating that object crosses the call boundary, while rebinding the local parameter changes only the function’s local name.',
  'python-identity-vs-equality': '`is` asks whether two names point to the same object. `==` asks whether their values compare equal and may invoke custom equality logic. Identity is the reliable test for `None` and unique sentinels.',
  'python-missing-sentinel': 'A sentinel adds a third state: missing, explicit `None`, or a concrete value. Compare a private sentinel by identity so no user value can accidentally equal it.',
  'python-shared-mutable-default': 'A function definition is executable code, so its default expressions run once when Python executes the `def`. Treat defaults as long-lived objects; allocate per-call mutable state inside the function.',
  'python-repeated-inner-list': 'Sequence multiplication repeats element references, not the objects behind those references. Use a comprehension such as `[[] for _ in range(3)]` when each position needs independent mutable state.',
  'python-shallow-list-copy': 'A shallow copy breaks aliasing at one container boundary only. Draw the object graph: `a` and `b` are different lists, but both outgoing edges still reach the same dictionary.',
  'python-tuple-shallow-immutability': 'Tuple immutability protects the tuple’s slots from rebinding. It says nothing about the mutability of objects reached through those slots, so nested dictionaries and lists can still change.',
  'python-falsy-valid-value': 'Truth testing merges several distinct values—`0`, `0.0`, `False`, empty containers, and `None`—into one branch. Test the exact missing-state contract instead of using truthiness as a substitute.',
  'python-nan-equality': 'IEEE NaN represents an undefined or unrepresentable numeric result, and comparisons involving it are intentionally unordered. Equality therefore cannot serve as a missing-value test.',
  'python-float-comparison': 'A useful tolerance has two parts: relative tolerance scales with the magnitude, while absolute tolerance handles values near zero. Choose both from the error budget of the computation.',

  // Collections
  'python-list-vs-tuple': 'Choose a collection by contract, not syntax alone. A list advertises changeable sequence state; a tuple advertises a fixed grouping and is hashable only when every contained value is hashable.',
  'python-dict-key-contract': 'A dictionary first selects a hash bucket and then uses equality to find the key. If a key’s hash-relevant state changes while stored, future lookups may search the wrong bucket.',
  'python-set-deduplication': 'Sets use the same hash-and-equality machinery as dictionaries. They make membership and uniqueness cheap, but they do not promise a presentation order; sort at output boundaries that require determinism.',
  'python-split-leakage-intersection': 'A nonempty intersection is a concrete witness that the same identity appears in both splits. Check the intersection before training so evaluation cannot benefit from duplicated examples.',
  'python-dict-iteration-order': 'Insertion order is a history of writes, not an ordering relation on keys. Two dictionaries with the same pairs can iterate differently if they were constructed in different orders.',
  'python-sort-vs-sorted': 'The mutating method returns `None` so code does not confuse the operation with a new value. Use `sorted` when the input must survive unchanged or when the source is any iterable rather than a list.',
  'python-zip-strict': 'Ordinary `zip` treats early exhaustion as normal control flow. At paired-data boundaries, strict mode converts a silent truncation bug into an immediate structural error.',
  'python-counter-labels': 'A `Counter` is a frequency mapping specialized for repeated observations. It keeps counting logic declarative and exposes operations such as `most_common` without a manual update loop.',
  'python-defaultdict-grouping': 'The factory defines how missing keys become values. Remember that reading a missing key also inserts it, so use ordinary `dict.get` when observation must not mutate the mapping.',
  'python-bounded-loss-history': 'A bounded deque makes retention part of the data structure rather than cleanup logic. Appending a new item evicts the oldest automatically, keeping memory bounded over an unbounded run.',

  // Functions and iteration
  'python-iterable-vs-iterator': 'An iterable is a source that can produce traversal state. An iterator is that stateful traversal itself: `iter(iterator) is iterator`, and each `next` consumes one position.',
  'python-generator-exhaustion': 'A generator stores one suspended execution frame, not a replayable recipe. Recreate the generator for another pass or materialize its values when repeated traversal is required.',
  'python-eager-vs-lazy-comprehension': 'A list comprehension pays all compute and memory up front. A generator expression defers work and can stream, but errors also move to consumption time and the values remain one-shot.',
  'python-yield-generator-function': 'Calling a generator function creates a suspended frame without running its body. Each `next` resumes execution until the next `yield`, preserving local variables between resumptions.',
  'python-first-match': 'The generator expression and `next` form a short-circuit search: production stops as soon as one item satisfies the predicate. The explicit default defines the no-match branch without an exception.',
  'python-keyword-only-hyperparameters': 'Keyword-only parameters turn call-site position into a named contract. They make reviews easier and let APIs add or reorder optional controls without changing the meaning of existing calls.',
  'python-closure-late-binding': 'A closure captures a variable cell, not a frozen snapshot of its current value. A default argument works here because its expression is evaluated when each lambda is created.',
  'python-partial-callable': 'Partial application stores selected arguments and returns another callable. It is useful for configuration, but the resulting callable still shares any mutable objects captured in those bound arguments.',
  'python-wraps-decorator': 'A decorator replaces one callable with another. `wraps` copies the public metadata and records `__wrapped__`, allowing introspection, documentation, and tools to recover the original function.',
  'python-lru-cache-contract': 'Caching trades memory and staleness risk for avoided work. The key comes from function arguments, so results are safe to reuse only when those arguments fully determine the result and invalidation is understood.',

  // Classes and interfaces
  'python-instance-shadows-class': 'Attribute lookup checks the instance before falling back to the class. Assigning through an instance creates a nearer binding; it does not edit the class attribute seen by other instances.',
  'python-mutable-class-attribute': 'A class attribute is one object shared through class lookup. Use it for intentional shared state; initialize per-instance mutable state in `__init__` when instances must be isolated.',
  'python-bound-method-self': 'Functions implement the descriptor protocol. Reading one through an instance produces a bound method that carries the instance, which is why the later call supplies only the remaining arguments.',
  'python-classmethod-constructor': 'A class method receives the runtime class rather than an instance. Constructing with `cls(...)` preserves polymorphism: a subclass calling the inherited constructor receives a subclass instance.',
  'python-staticmethod-helper': 'A static method changes namespacing, not behavior: it is an ordinary function retrieved from the class without automatic `self` or `cls`. Keep it on the class only when that ownership aids discovery.',
  'python-property-boundary': 'Attribute syntax suggests cheap, deterministic access. A property can preserve that interface while computing a value, but hidden network, disk, or heavyweight work violates the cost model callers infer.',
  'python-callable-transform': 'A callable object separates configuration from invocation. `__init__` stores reusable state, while `__call__` applies that state through the same interface as a function.',
  'python-composition-pipeline': 'Composition keeps behavior in small objects connected by an explicit data flow. Each stage can be replaced or tested independently, avoiding inheritance trees where one change affects unrelated subclasses.',
  'python-protocol-interface': 'A protocol names the behavior a consumer needs instead of the ancestry a provider must have. Static compatibility follows members and signatures, so third-party types can participate without adapter inheritance.',
  'python-super-mro': '`super` is a cooperative method-resolution tool. It starts after the current class in the runtime MRO, which lets every class in a multiple-inheritance chain participate exactly once when each forwards correctly.',

  // Typing
  'python-annotations-runtime': 'Annotations are metadata consumed by type checkers, editors, frameworks, or explicit runtime validators. The Python call mechanism itself does not reject a value merely because it disagrees with an annotation.',
  'python-any-vs-object': '`Any` asks the checker to stop checking operations that flow through the value. `object` preserves uncertainty safely: every value fits, but code must narrow the type before using specialized operations.',
  'python-nullable-not-optional': 'Nullability is a property of the accepted value set. Optionality is a property of the call signature. A required parameter can accept `None`, and an optional parameter can still reject `None`.',
  'python-sequence-parameter': 'Annotate the smallest behavior the function needs. `Sequence` provides length and indexed read access while leaving tuples, lists, and other sequence implementations available to callers.',
  'python-iterable-vs-iterator-type': 'Accepting an iterator tells callers the function may consume shared traversal state. Accepting an iterable is broader and may permit the function to request a fresh iterator.',
  'python-mapping-parameter': 'A read-only `Mapping` annotation prevents the signature from promising mutation and accepts more providers than `dict`. Use a mutable mapping type only when writes are part of the contract.',
  'python-callable-signature': '`Callable[[Sample], Sample]` describes the input-output relation but not every call detail. For richer methods, keyword names, or attributes, define a callback `Protocol` instead.',
  'python-typeddict-runtime': 'A `TypedDict` exists primarily in the static type system. Data crossing a JSON or user-input boundary still needs explicit runtime validation before code can trust its keys and values.',
  'python-literal-optimizer': 'A literal type turns a small closed set of values into a checked interface. It catches misspellings before runtime and lets a checker narrow branches based on the selected value.',
  'python-annotated-field': '`Annotated` keeps the base static type while attaching metadata for a cooperating runtime library. The metadata has no effect unless some tool, such as Pydantic, chooses to interpret it.',
  'python-typevar-relationship': 'A type variable carries a relationship across positions in a signature. Each call chooses a concrete `T`, allowing the checker to infer that the returned value has the same element type as the input sequence.',

  // Dataclasses
  'python-dataclass-generated-methods': 'A dataclass derives boilerplate from annotated fields. It is still a normal Python class: annotations are not validated, methods can be added, and generated behavior can be configured or overridden.',
  'python-dataclass-mutable-default': 'The factory is stored in the field definition and called during each initialization. This moves mutable allocation to instance construction and prevents object graphs from being shared accidentally.',
  'python-dataclass-post-init': '`__post_init__` runs after the generated initializer has assigned every field. That makes it the natural place for derived fields and invariants that depend on more than one constructor argument.',
  'python-dataclass-classvar': 'Marking a name as `ClassVar` tells dataclass machinery that the name belongs to the class contract, not instance data. It is excluded from generated initialization, comparison, and representation.',
  'python-dataclass-frozen-shallow': 'Frozen instances prevent normal attribute assignment, which is useful for stable configuration references. Deep immutability still requires immutable field values or defensive copying.',
  'python-dataclass-kw-only': 'Keyword-only construction makes field names visible at every call site. This reduces argument-order mistakes and lets the class evolve without silently reinterpreting positional calls.',
  'python-dataclass-slots': 'Slots replace the usual per-instance attribute dictionary with a fixed layout. They can reduce memory and catch undeclared attributes, but they also change inheritance, weak-reference, and dynamic-attribute behavior.',
  'python-dataclass-replace': '`replace` expresses a copy-with-changes operation at the dataclass level. Because it calls the initializer, normal initialization and `__post_init__` invariants apply to the new instance.',
  'python-dataclass-asdict': '`asdict` walks the full dataclass object graph and deep-copies non-dataclass leaves. That is convenient for small value records but costly or invalid for tensors, clients, locks, and other runtime objects.',

  // Pydantic v2
  'python-pydantic-boundary-choice': 'Use validation where trust changes. Internal code benefits from lightweight typed records, while serialized or user-controlled data needs parsing, rejection rules, and useful boundary errors.',
  'python-pydantic-dataclass': 'A Pydantic dataclass keeps the dataclass programming model while validating initialization. It does not become a `BaseModel`; wrap the type in a `TypeAdapter` for schema and serialization operations.',
  'python-pydantic-default-coercion': 'Coercion is useful at text-heavy boundaries, but it can hide upstream schema drift. Decide deliberately whether a boundary should normalize compatible values or reject anything with the wrong runtime type.',
  'python-pydantic-strict-mode': 'Strictness changes the boundary contract from “convert if possible” to “require the expected type.” It can be applied globally or narrowly where coercion would make bad input look valid.',
  'python-pydantic-extra-forbid': 'Ignoring unknown keys makes forward compatibility easy but also hides typos. For configuration files, forbidding extras usually gives a safer failure: the misspelled control cannot silently fall back to its default.',
  'python-pydantic-positive-field': 'Declarative constraints keep validation, generated schema, and error locations tied to the field. Prefer them over a custom validator when the rule is a standard local bound.',
  'python-pydantic-default-factory': 'The default factory runs for every successful model construction. It gives each model an independent mutable value and can also derive a default from already validated earlier fields when needed.',
  'python-pydantic-field-validator': 'A field validator owns one field’s normalization or local rule. Choose before-validation only when raw input must be transformed; after-validation can rely on the parsed field type.',
  'python-pydantic-model-validator': 'Cross-field rules belong at model scope because no single field owns the invariant. An after-model validator sees the completed typed model and can return it after checking the relationship.',
  'python-pydantic-assignment-validation': 'Construction-time validity does not guarantee lifetime validity when fields remain mutable. Assignment validation moves the boundary to each write, at an ongoing runtime cost.',
  'python-pydantic-io-api': 'Choose the API that matches the representation at the boundary. Python-object methods avoid an unnecessary JSON round trip; JSON methods own decoding or encoding when bytes or text are the real interface.',
  'python-pydantic-type-adapter': 'A type adapter gives an arbitrary annotation the same validation and serialization engine used by models. Build it once and reuse it rather than creating a wrapper class or rebuilding schemas per call.',
  'python-pydantic-private-attr': 'Private attributes hold operational state rather than validated domain data. They are omitted from schemas and dumps, so another mechanism must recreate them after deserialization.',
  'python-pydantic-arbitrary-tensor': 'Allowing an arbitrary class only proves that the object has that class. Tensor invariants—rank, shape, dtype, device, finiteness, or value range—still need explicit validation.',
  'python-pydantic-model-construct': 'Bypassing validation moves responsibility to the caller. Use construction without validation only after another trusted layer has established every invariant; otherwise invalid state enters the model silently.',
  'python-pydantic-model-copy': 'A shallow model copy creates a new outer model while retaining nested references. Choose `deep=True` only when nested isolation is required, because deep copying can be expensive or unsupported for runtime resources.',
  'python-pydantic-discriminated-union': 'The discriminator makes variant selection explicit and efficient. After Pydantic reads the tag, it validates only the matching model and can report errors against that concrete configuration shape.',

  // Reliability and I/O
  'python-narrow-exception': 'An exception boundary should catch only failures it can translate, retry, or recover from. Broad catches turn programming defects into misleading configuration errors and make the original fault harder to locate.',
  'python-explicit-exception-cause': 'Exception chaining keeps both abstraction levels: the outer error explains the failed operation, while the cause preserves the low-level reason and traceback needed to diagnose it.',
  'python-context-manager-cleanup': 'A context manager encodes acquisition and release as one protocol. Its exit method runs while an exception is propagating, so resource cleanup does not depend on every control-flow path remembering to close.',
  'python-assert-input-validation': 'Assertions document conditions that should already be true if the program is correct. External data can always be wrong, so validate it with an explicit branch and stable exception regardless of optimization mode.',
  'python-str-vs-bytes': 'Encoding maps Unicode text to bytes; decoding maps bytes to text. Make the codec explicit at the boundary so the rest of the program operates on one representation with known meaning.',
  'python-sorted-glob': 'Directory iteration order can vary across filesystems, runs, and machines. Sorting converts that environmental accident into a stable ingestion order before it can affect splits, hashes, or outputs.',
  'python-untrusted-pickle': 'Pickle reconstructs Python objects by executing a serialization program, not by parsing inert data. Authentication or a safer data format is required when the producer is outside the trust boundary.',
  'python-stable-experiment-id': 'The digest is stable only if the serialized bytes are stable. Canonicalize key order, numeric representation, and omitted defaults before hashing so equivalent configurations produce the same identity.',
  'python-multiprocessing-entrypoint': 'Spawn starts a fresh interpreter that imports the main module. The guard prevents child imports from recursively creating more processes, and top-level callables give pickle an importable name.',
  'python-concurrency-choice': 'Match the tool to the waiting model. Threads help when native or blocking work releases the interpreter, processes isolate CPU-bound Python execution, and `asyncio` needs nonblocking libraries plus explicit cooperative scheduling.',
};
