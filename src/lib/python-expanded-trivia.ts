import type { TriviaCard } from './trivia-decks';

/**
 * Coverage added after the practical 88-card core.
 *
 * These cards deliberately test one decision or failure mode at a time. The
 * short answer supports active recall, the explanation resolves the immediate
 * scenario, and the detail transfers the rule to production code.
 */
export const expandedPythonTriviaCards = [
  // Syntax and control flow
  {
    id: 'python-truthiness-contract',
    topic: 'Syntax & control flow',
    code: `values = [[], "False", 0, float("nan")]
result = [bool(value) for value in values]`,
    question: 'What value does `result` contain?',
    answer: '`[False, True, False, True]`',
    explanation: 'Empty containers and numeric zero are falsy; a nonempty string and NaN are truthy.',
    detail: 'Truthiness asks whether an object counts as empty or zero, not whether its text means false or its number is usable. Test domain conditions such as finiteness explicitly.',
  },
  {
    id: 'python-short-circuit-return-value',
    topic: 'Syntax & control flow',
    code: `cache = {}
result = cache.get("model") or load_model()`,
    question: 'What does `or` return in this expression?',
    answer: 'First truthy operand',
    acceptedAnswers: ['first truthy value', 'operand not bool'],
    explanation: '`or` returns an operand rather than converting the result to `bool`, and it evaluates left to right.',
    detail: 'Short-circuit operators combine control flow with value selection. They are compact defaults only when every falsy left-hand value truly means absent; otherwise use an explicit missing-value test.',
  },
  {
    id: 'python-chained-comparison-once',
    topic: 'Syntax & control flow',
    code: `valid = 0 < get_score() <= 1`,
    question: 'How many times is `get_score()` evaluated?',
    answer: 'Once',
    explanation: 'A chained comparison evaluates the middle expression once and combines both comparisons with short-circuiting.',
    detail: '`a < b <= c` behaves like `a < b and b <= c` except that `b` is evaluated once. This matters when the expression is costly or has side effects.',
  },
  {
    id: 'python-loop-else',
    topic: 'Syntax & control flow',
    code: `for item in items:
    if matches(item):
        break
else:
    raise LookupError("no match")`,
    question: 'When does the `else` block execute?',
    answer: 'When no `break` occurs',
    acceptedAnswers: ['normal loop exhaustion', 'loop completes without break'],
    explanation: 'Loop `else` runs after normal exhaustion, including an empty iterable, but not after `break`.',
    detail: 'The `else` belongs to the search outcome, not to the final condition test. It expresses “the loop exhausted without finding a reason to stop” without a separate flag.',
  },
  {
    id: 'python-comprehension-scope',
    topic: 'Syntax & control flow',
    code: `i = 99
values = [i * 2 for i in range(3)]
result = i`,
    question: 'What value does `result` contain in Python 3?',
    answer: '`99`',
    acceptedAnswers: ['99'],
    explanation: 'A comprehension has its own implicit scope, so its iteration variable does not overwrite the surrounding `i`.',
    detail: 'Comprehension variables are local to the comprehension frame, unlike variables assigned by an ordinary `for` loop. Names used by the expression still follow normal enclosing-scope lookup.',
  },
  {
    id: 'python-comprehension-filter-order',
    topic: 'Syntax & control flow',
    code: `pairs = [(x, y) for x in xs if ready(x) for y in expand(x) if valid(y)]`,
    question: 'How should the comprehension be read?',
    answer: 'Left to right',
    acceptedAnswers: ['nested loops left to right'],
    explanation: 'Read each `for` and `if` in order as the equivalent nested loop; later clauses can use earlier names.',
    detail: 'A comprehension is compact syntax for a precise nesting structure. When the left-to-right execution order is hard to simulate, expand it into ordinary loops for reviewability.',
  },
  {
    id: 'python-unpacking-star-target',
    topic: 'Syntax & control flow',
    code: `first, *middle, last = range(5)`,
    question: 'What value does `middle` contain?',
    answer: '`[1, 2, 3]`',
    acceptedAnswers: ['[1,2,3]'],
    explanation: 'The starred assignment target collects unmatched items into a new list.',
    detail: 'Extended unpacking validates the whole arity contract while naming the meaningful ends. The starred target is always a list, even when the source is a tuple or iterator.',
  },
  {
    id: 'python-walrus-scope',
    topic: 'Syntax & control flow',
    code: `if (batch := loader.next_batch()) is not None:
    consume(batch)`,
    question: 'Why can `:=` be useful here?',
    answer: 'Evaluate once and bind',
    acceptedAnswers: ['assignment expression', 'bind and test once'],
    explanation: 'The assignment expression stores the result while the surrounding condition tests that same object.',
    detail: 'Use an assignment expression when one computed value participates in both a condition and its body. If binding inside the expression hides control flow, a separate statement is clearer.',
  },
  {
    id: 'python-match-guard-order',
    topic: 'Syntax & control flow',
    code: `match event:
    case {"type": "score", "value": value} if value >= 0:
        handle(value)
    case _:
        reject(event)`,
    question: 'When is the guard `value >= 0` evaluated?',
    answer: 'After the pattern matches',
    acceptedAnswers: ['after structural match'],
    explanation: 'A case guard runs only after its pattern succeeds and its captured names have been bound.',
    detail: 'Structural pattern matching first checks shape and literals, then evaluates the guard. Cases are attempted top to bottom, so specific patterns must precede broad ones.',
  },
  {
    id: 'python-match-capture-gotcha',
    topic: 'Syntax & control flow',
    code: `RED = "red"
match color:
    case RED:
        handle_red()`,
    question: 'Why is bare `RED` not a constant-value pattern?',
    answer: 'It is a capture pattern',
    acceptedAnswers: ['capture', 'name capture'],
    explanation: 'A bare name in a pattern captures the subject; qualified names or literals express value patterns.',
    detail: 'Pattern syntax is not expression syntax. An unqualified name creates a new binding and matches broadly, which can make later cases unreachable; use an enum member or other dotted name.',
  },
  {
    id: 'python-augmented-assignment-mutation',
    topic: 'Syntax & control flow',
    code: `a = [1]
b = a
a += [2]`,
    question: 'What value does `b` contain?',
    answer: '`[1, 2]`',
    acceptedAnswers: ['[1,2]'],
    explanation: 'List `+=` mutates the existing list in place, so the alias `b` observes the extension.',
    detail: 'Augmented assignment first tries the in-place special method and may mutate; ordinary `a = a + ...` creates and rebinds a new list. The behavior depends on the operand type.',
  },
  {
    id: 'python-slice-copy',
    topic: 'Syntax & control flow',
    code: `items = [[1], [2]]
copy = items[:]
copy[0].append(9)`,
    question: 'What value does `items[0]` contain?',
    answer: '`[1, 9]`',
    acceptedAnswers: ['[1,9]'],
    explanation: 'A list slice creates a shallow outer copy and keeps references to the same nested lists.',
    detail: 'Slicing breaks only the list container edge. It is an efficient shallow copy when elements are immutable or intentionally shared, but it does not isolate a nested object graph.',
  },
  {
    id: 'python-pass-vs-ellipsis',
    topic: 'Syntax & control flow',
    code: `def interface() -> None:
    ...

def no_op() -> None:
    pass`,
    question: 'How do `...` and `pass` differ at runtime here?',
    answer: 'Both do nothing',
    acceptedAnswers: ['no runtime difference here'],
    explanation: '`pass` is a no-op statement; `...` evaluates the `Ellipsis` singleton and discards it in this position.',
    detail: 'Their runtime effect is equivalent in an empty body, but their intent differs. `pass` denotes an intentional no-op; ellipsis conventionally marks a stub or omitted implementation.',
  },

  // Collections and algorithms
  {
    id: 'python-list-append-amortized',
    topic: 'Collections & algorithms',
    code: `items = []
for sample in stream:
    items.append(sample)`,
    question: 'Why can an occasional `append` take much longer than the surrounding appends?',
    answer: 'The list must resize and copy its references',
    acceptedAnswers: ['dynamic array resize', 'resizing', 'it reallocates and copies'],
    explanation: 'A list is a dynamic array with spare capacity. Most appends fill one free slot, but a full array must allocate a larger block and copy its existing references before adding the new one.',
    detail: 'That over-allocation spreads rare linear resizes across many cheap appends, so a sequence of appends averages constant time per item. “Amortized `O(1)`” describes that average; it does not promise constant latency for every individual call.',
  },
  {
    id: 'python-list-front-pop',
    topic: 'Collections & algorithms',
    code: `while pending:
    item = pending.pop(0)`,
    question: 'Why can this queue become quadratic?',
    answer: '`pop(0)` shifts `O(n)` items',
    acceptedAnswers: ['front pop is O(n)', 'use deque'],
    explanation: 'Removing the first list element shifts every remaining reference; use `deque.popleft()` for a FIFO queue.',
    detail: 'The hidden unit of cost is not the returned element but the contiguous array behind it. Repeating a linear shift for every item turns a simple drain into `O(n²)` work.',
  },
  {
    id: 'python-membership-complexity',
    topic: 'Collections & algorithms',
    code: `blocked_ids = load_blocked_ids()
kept = [sample for sample in samples if sample.id not in blocked_ids]`,
    question: 'Which container should `blocked_ids` usually be when this performs many membership checks?',
    answer: 'A `set`',
    acceptedAnswers: ['set', 'a set'],
    explanation: 'A list scans its elements for each `in` check, while a set uses a hash table and usually finds a value in average constant time. Converting once can avoid repeating a linear scan for every sample.',
    detail: 'Build a set when repeated membership dominates and IDs have a stable hash contract. The conversion itself costs time and memory, so a tiny collection used once may still be clearer as a list.',
  },
  {
    id: 'python-dict-view-live',
    topic: 'Collections & algorithms',
    code: `keys = config.keys()
config["device"] = "cpu"
result = "device" in keys`,
    question: 'What value does `result` contain?',
    answer: '`True`',
    explanation: 'Dictionary views are dynamic windows onto the mapping rather than frozen list snapshots.',
    detail: 'A view avoids copying and tracks later mutations, which is useful for set-like comparison. Materialize `list(config)` when the consumer needs a stable snapshot independent of future writes.',
  },
  {
    id: 'python-dict-mutation-iteration',
    topic: 'Collections & algorithms',
    code: `for key in config:
    if obsolete(key):
        del config[key]`,
    question: 'Why is this loop unsafe?',
    answer: 'Dictionary size changes during iteration',
    acceptedAnswers: ['runtime error', 'mutating dict during iteration'],
    explanation: 'Adding or removing keys while iterating a dictionary can raise `RuntimeError`; iterate over `list(config)` instead.',
    detail: 'The iterator depends on the mapping structure remaining stable. Updating a value for an existing key is different from changing the key set, but collecting intended edits first is clearer.',
  },
  {
    id: 'python-setdefault-eager-default',
    topic: 'Collections & algorithms',
    code: `value = cache.setdefault(key, expensive_default())`,
    question: 'When is `expensive_default()` evaluated?',
    answer: 'Every call',
    acceptedAnswers: ['before setdefault', 'eagerly'],
    explanation: 'Python evaluates function arguments before calling `setdefault`, even when `key` already exists.',
    detail: '`setdefault` avoids two mapping operations but does not make default construction lazy. Use an explicit membership branch or a missing-value protocol when construction is expensive or has side effects.',
  },
  {
    id: 'python-defaultdict-read-inserts',
    topic: 'Collections & algorithms',
    code: `groups = defaultdict(list)
missing = groups["unknown"]`,
    question: 'What side effect does the lookup have?',
    answer: 'It inserts the missing key',
    acceptedAnswers: ['creates unknown key'],
    explanation: '`defaultdict.__getitem__` calls the factory and stores the result for a missing key.',
    detail: 'A default dictionary turns reading through brackets into a write. Use `get` when a probe must not change mapping length, serialized output, or later iteration.',
  },
  {
    id: 'python-counter-missing-zero',
    topic: 'Collections & algorithms',
    code: `counts = Counter({"cat": 2})
result = counts["dog"]`,
    question: 'What value does `result` contain?',
    answer: '`0`',
    acceptedAnswers: ['0'],
    explanation: 'A `Counter` returns zero for a missing key instead of raising `KeyError`.',
    detail: 'Zero counts can remain stored after subtraction even though they behave like absence in equality rules. Apply unary plus or delete entries when output should contain positive counts only.',
  },
  {
    id: 'python-chainmap-write-layer',
    topic: 'Collections & algorithms',
    code: `config = ChainMap(cli, environment, defaults)
config["batch_size"] = 64`,
    question: 'Which mapping receives the write?',
    answer: 'The first mapping',
    acceptedAnswers: ['cli', 'first map'],
    explanation: '`ChainMap` searches all mappings for reads but directs writes and deletions only to the first.',
    detail: 'A chain is a live layered view, not a merged snapshot. It preserves precedence and lets overrides remain separate from defaults, but mutations do not update the layer where a key was found.',
  },
  {
    id: 'python-stable-sort',
    topic: 'Collections & algorithms',
    code: `records.sort(key=lambda row: row.timestamp)
records.sort(key=lambda row: row.priority)`,
    question: 'Why does the second sort preserve timestamp order within each priority?',
    answer: 'Python sort is stable',
    acceptedAnswers: ['stable sort'],
    explanation: 'Equal keys retain their relative input order, so the earlier ordering survives within later ties.',
    detail: 'Stability lets several simple least-significant-to-most-significant sorts express a compound ordering. A tuple key is often clearer and performs one sort when all key components are available together.',
  },
  {
    id: 'python-sort-key-once',
    topic: 'Collections & algorithms',
    question: 'How often does `sorted(items, key=expensive_key)` call the key function per item?',
    answer: 'Once',
    acceptedAnswers: ['one time'],
    explanation: 'Python computes and caches one key per input item for the duration of the sort.',
    detail: 'Key-based sorting uses decorate-sort-undecorate behavior internally. Put expensive derivation in `key` rather than a comparator so comparison does not repeatedly recompute it.',
  },
  {
    id: 'python-heap-top-k',
    topic: 'Collections & algorithms',
    code: `top_100 = sorted(scores, reverse=True)[:100]`,
    question: 'What should replace this full sort when `scores` is huge and only 100 values are needed?',
    answer: '`heapq.nlargest(100, scores)`',
    acceptedAnswers: ['heapq nlargest', 'bounded heap'],
    explanation: '`heapq.nlargest(100, scores)` keeps only the strongest candidates instead of ordering every score. It avoids work when the requested `k` is much smaller than the total input size.',
    detail: 'A full sort costs `O(n log n)`. Heap selection is attractive when `k` is much smaller than `n`; for large `k`, implementation constants can make sorting competitive.',
  },
  {
    id: 'python-bisect-sorted-insert',
    topic: 'Collections & algorithms',
    question: 'Why is `bisect.insort` still `O(n)` despite its binary search?',
    answer: 'List insertion shifts elements',
    acceptedAnswers: ['insertion is O(n)', 'shifting'],
    explanation: 'Bisect finds the position in `O(log n)`, but the contiguous list must shift later references.',
    detail: 'Searching and updating have different cost centers. A sorted list is excellent for read-heavy binary search; frequent middle insertion calls for a different data structure or batched rebuild.',
  },
  {
    id: 'python-frozenset-key',
    topic: 'Collections & algorithms',
    code: `cache_key = {"decode", "resize"}
result = cache[cache_key]`,
    question: 'What immutable collection should replace `cache_key` so the unordered group can be used as a dictionary key?',
    answer: '`frozenset`',
    explanation: 'A mutable set is unhashable because changing its members would change its identity as a key. A `frozenset` fixes the members at construction and is hashable when those members are hashable.',
    detail: 'Use a tuple when order carries meaning and a frozenset when membership alone defines identity. Duplicate source elements disappear, so it is not a multiset key.',
  },
  {
    id: 'python-dict-union-precedence',
    topic: 'Collections & algorithms',
    code: `merged = defaults | overrides`,
    question: 'Which mapping wins when both contain the same key?',
    answer: '`overrides`',
    acceptedAnswers: ['right operand', 'right mapping'],
    explanation: 'Dictionary union returns a new dictionary and lets the right operand replace duplicate keys.',
    detail: 'Merge order is configuration precedence encoded in syntax. Nested dictionaries are still replaced as whole values; built-in union does not perform a recursive deep merge.',
  },
  {
    id: 'python-memoryview-zero-copy',
    topic: 'Collections & algorithms',
    code: `buffer = bytearray(b"abcd")
view = memoryview(buffer)[1:3]
view[0] = ord("X")`,
    question: 'What value does `buffer` contain after the write through `view`?',
    answer: '`bytearray(b"aXcd")`',
    acceptedAnswers: ['aXcd', 'bytearray(b"aXcd")'],
    explanation: 'A memory view exposes an existing binary buffer without copying it. The slice still refers to `buffer`, so changing the first byte of the view changes the original bytearray.',
    detail: 'Zero-copy access reduces allocation and bandwidth, but it extends the source buffer lifetime and preserves aliasing. Consumers must respect mutability, format, shape, and contiguity constraints.',
  },

  // Functions and iteration
  {
    id: 'python-positional-only-parameter',
    topic: 'Functions & iteration',
    code: `def normalize(value, /, *, mean=0.0, scale=1.0):
    ...

a = normalize(batch, mean=0.5, scale=0.2)
b = normalize(value=batch, mean=0.5, scale=0.2)
c = normalize(batch, 0.5, 0.2)`,
    question: 'Which call is valid: `a`, `b`, or `c`?',
    answer: '`a`',
    acceptedAnswers: ['a', 'only a'],
    explanation: 'The `/` makes `value` positional-only, so naming it makes `b` invalid. The `*` makes `mean` and `scale` keyword-only, so passing them by position makes `c` invalid.',
    detail: 'Binding markers protect an API from ambiguous calls. Positional-only hides an implementation parameter name; keyword-only makes control names visible and prevents same-typed arguments from being swapped.',
  },
  {
    id: 'python-args-kwargs-shape',
    topic: 'Functions & iteration',
    code: `def capture(*args, **kwargs):
    return args, kwargs

result = capture("train", 32, shuffle=True)`,
    question: 'What value does `result` contain?',
    answer: '`(("train", 32), {"shuffle": True})`',
    acceptedAnswers: ['((train, 32), {shuffle: true})', 'tuple and dict'],
    explanation: '`*args` collects extra positional arguments into a tuple. `**kwargs` collects extra named arguments into a dictionary, so the call separates the two positional values from `shuffle=True`.',
    detail: 'Variadic parameters trade a closed, inspectable signature for forwarding flexibility. Preserve explicit parameters for important controls, and validate forwarded keyword names at the boundary that owns them.',
  },
  {
    id: 'python-default-captures-old-global',
    topic: 'Functions & iteration',
    code: `rate = 0.1
def train(lr: float = rate) -> float:
    return lr
rate = 0.01
result = train()`,
    question: 'What value does `result` contain?',
    answer: '`0.1`',
    acceptedAnswers: ['0.1'],
    explanation: 'The default expression reads `rate` when the `def` statement executes, not when the function is called.',
    detail: 'Defaults are stored on the function object. This explains both mutable-default sharing and stale captured configuration; use `None` or a sentinel when the current value must be resolved per call.',
  },
  {
    id: 'python-nonlocal-rebinding',
    topic: 'Functions & iteration',
    code: `def counter():
    value = 0
    def increment():
        nonlocal value
        value += 1
        return value
    return increment`,
    question: 'What error would `increment()` raise if `nonlocal value` were removed?',
    answer: '`UnboundLocalError`',
    acceptedAnswers: ['UnboundLocalError', 'unbound local error'],
    explanation: 'Assignment normally makes `value` local to `increment`, so `value += 1` would try to read that uninitialized local name. `nonlocal` redirects the assignment to the `value` binding in `counter` instead.',
    detail: '`nonlocal` changes compile-time scope resolution for writes. Mutation of an enclosing mutable object needs no declaration because the binding itself is not being replaced.',
  },
  {
    id: 'python-yield-return-difference',
    topic: 'Functions & iteration',
    question: 'How does `yield` change a function call compared with `return`?',
    answer: 'It creates a resumable generator',
    acceptedAnswers: ['suspends frame', 'generator function'],
    explanation: 'Any `yield` makes the function a generator function; calling it returns an iterator before its body runs.',
    detail: '`yield` preserves the frame, instruction position, and locals between pulls. A generator `return` terminates iteration and can pass a final value to `yield from`, not to an ordinary loop.',
  },
  {
    id: 'python-yield-from-delegation',
    topic: 'Functions & iteration',
    question: 'What does `yield from child()` delegate beyond ordinary values?',
    answer: '`send`, `throw`, `close`, and return value',
    acceptedAnswers: ['full generator protocol'],
    explanation: 'It forwards the generator protocol and makes the child generator return value the expression result.',
    detail: 'A manual `for` loop forwards yielded items only. `yield from` composes coroutines at the protocol level, including exception flow and cleanup, though modern asynchronous work usually uses native coroutines.',
  },
  {
    id: 'python-iter-callable-sentinel',
    topic: 'Functions & iteration',
    code: `for chunk in iter(lambda: handle.read(8192), b""):
    process(chunk)`,
    question: 'When does this iterator stop?',
    answer: 'When the callable returns `b""`',
    acceptedAnswers: ['on sentinel'],
    explanation: 'The two-argument form of `iter` repeatedly calls the function until its result equals the sentinel.',
    detail: 'Callable-sentinel iteration turns a stateful pull API into a normal iterator without a manual infinite loop. Equality defines termination, so choose a sentinel the data cannot ambiguously produce early.',
  },
  {
    id: 'python-itertools-groupby-adjacent',
    topic: 'Functions & iteration',
    code: `groups = [(key, list(rows)) for key, rows in groupby(records, key=label)]`,
    question: 'Which records does `groupby` combine?',
    answer: 'Adjacent equal keys',
    acceptedAnswers: ['consecutive groups', 'adjacent records'],
    explanation: '`itertools.groupby` starts a new group whenever the key changes; it does not globally aggregate matching keys.',
    detail: 'Sort by the same grouping key first when all equal keys must meet. Group iterators share the underlying source, so materialize a group before advancing if it must survive.',
  },
  {
    id: 'python-itertools-tee-buffer',
    topic: 'Functions & iteration',
    question: 'What hidden cost can `itertools.tee(source, 2)` incur?',
    answer: 'Buffered lag between consumers',
    acceptedAnswers: ['memory buffer', 'unbounded buffering'],
    explanation: 'Tee stores values consumed by the faster iterator until the slower iterator catches up.',
    detail: 'The copies are independent cursors, not independent sources. If one cursor races far ahead or never finishes, retained values can approach the size of materializing the input.',
  },
  {
    id: 'python-any-all-short-circuit',
    topic: 'Functions & iteration',
    code: `items = [2, 4, 5, 8]
result = any(is_odd(item) for item in items)`,
    question: 'Which item causes `any` to stop calling `is_odd`?',
    answer: '`5`',
    acceptedAnswers: ['5', 'after 5', 'the first true result'],
    explanation: '`any` requests generator values until one is truthy, then stops. It tests `2`, `4`, and `5`; it never evaluates `8` or builds a temporary list of every predicate result.',
    detail: '`all` similarly stops at the first falsy result. Passing a prebuilt list defeats the compute and memory savings because every predicate runs before the reduction begins.',
  },
  {
    id: 'python-decorator-definition-time',
    topic: 'Functions & iteration',
    code: `@register("encoder")
def encode(sample):
    ...`,
    question: 'When are `register("encoder")` and its returned decorator executed?',
    answer: 'During function definition',
    acceptedAnswers: ['at import or def execution time'],
    explanation: 'Decorator expressions and application run when Python executes the `def`, commonly during module import.',
    detail: 'Decorators can therefore mutate registries before the function is ever called. That convenience creates import-time side effects, ordering dependencies, and test isolation concerns that a production registry must manage.',
  },
  {
    id: 'python-cache-keyword-order',
    topic: 'Functions & iteration',
    code: `@lru_cache
def score(*, split: str, metric: str):
    ...

score(split="val", metric="f1")
score(metric="f1", split="val")`,
    question: 'Can the two calls occupy different cache entries?',
    answer: 'Yes',
    explanation: 'The cache may treat distinct keyword argument orderings as distinct keys even when calls are semantically equivalent.',
    detail: 'Memoization keys encode call syntax as well as values. Canonicalize a public wrapper before the cached function when equivalent inputs can arrive in several spellings or orders.',
  },

  // Object model and OOP
  {
    id: 'python-new-before-init',
    topic: 'Object model & OOP',
    question: 'How do `__new__` and `__init__` divide object construction?',
    answer: 'Create / initialize',
    acceptedAnswers: ['__new__ creates __init__ initializes'],
    explanation: '`__new__` returns an instance; Python calls `__init__` only when that result is an instance of the class.',
    detail: 'Most mutable classes need only `__init__`. Customize `__new__` when the value must be determined before an immutable instance exists or when construction may return a cached or different object.',
  },
  {
    id: 'python-data-descriptor-precedence',
    topic: 'Object model & OOP',
    question: 'Which wins during attribute lookup: an instance `__dict__` entry or a data descriptor?',
    answer: 'Data descriptor',
    acceptedAnswers: ['descriptor with __set__'],
    explanation: 'A descriptor defining `__set__` or `__delete__` takes precedence over the instance dictionary.',
    detail: 'The order is data descriptor, instance dictionary, non-data descriptor or class attribute, then `__getattr__`. This is why a property setter cannot be bypassed by inserting the same name on the instance.',
  },
  {
    id: 'python-getattr-vs-getattribute',
    topic: 'Object model & OOP',
    question: 'When does `__getattr__` run compared with `__getattribute__`?',
    answer: 'Only after normal lookup fails',
    acceptedAnswers: ['getattribute always getattr fallback'],
    explanation: '`__getattribute__` handles every instance lookup; `__getattr__` is the missing-attribute fallback.',
    detail: 'Overriding `__getattribute__` can intercept everything and easily recurse; delegate with `object.__getattribute__`. Use `__getattr__` for narrow lazy or compatibility fallbacks whose absence should otherwise raise `AttributeError`.',
  },
  {
    id: 'python-special-method-class-lookup',
    topic: 'Object model & OOP',
    code: `obj.__len__ = lambda: 3
result = len(obj)`,
    question: 'Why can `len(obj)` still fail?',
    answer: 'Special methods are looked up on the type',
    acceptedAnswers: ['class lookup', 'implicit special method lookup'],
    explanation: 'Implicit protocol operations bypass ordinary instance lookup for many special methods such as `__len__`.',
    detail: 'Python resolves core syntax through the class so per-instance monkey patches cannot destabilize interpreter protocols. Define the special method on the type or use an explicit ordinary method.',
  },
  {
    id: 'python-eq-disables-hash',
    topic: 'Object model & OOP',
    code: `class Sample:
    def __eq__(self, other):
        return self.id == other.id

seen = {Sample()}`,
    question: 'What happens when Python tries to create `seen`?',
    answer: '`TypeError`: `Sample` is unhashable',
    acceptedAnswers: ['TypeError', 'unhashable', 'Sample is unhashable'],
    explanation: 'Defining value equality without a matching hash makes Python set `Sample.__hash__ = None`. Set insertion therefore raises `TypeError` rather than allowing equal objects to occupy inconsistent hash locations.',
    detail: 'Equal objects must have equal hashes, and a stored hash must remain stable. Value objects can hash the immutable fields used by equality; mutable value objects should remain unhashable.',
  },
  {
    id: 'python-notimplemented-protocol',
    topic: 'Object model & OOP',
    question: 'Why should an unsupported binary special method return `NotImplemented`?',
    answer: 'Allow reflected dispatch',
    acceptedAnswers: ['try other operand', 'fallback comparison'],
    explanation: 'The singleton tells Python to try the reflected operation or another defined fallback.',
    detail: '`NotImplemented` is a protocol result, not an exception. `NotImplementedError` instead signals that ordinary callable behavior is intentionally missing and stops the operation unless caught.',
  },
  {
    id: 'python-repr-vs-str',
    topic: 'Object model & OOP',
    question: 'How should `__repr__` and `__str__` differ?',
    answer: 'Developer detail / user text',
    acceptedAnswers: ['unambiguous vs readable'],
    explanation: '`repr` should aid debugging and, when practical, resemble reconstruction; `str` is the readable presentation.',
    detail: 'Containers use element representations, logs often need diagnostic identity, and f-strings select the protocol with `!r` or `!s`. Never expose secrets in either representation.',
  },
  {
    id: 'python-name-mangling',
    topic: 'Object model & OOP',
    code: `class Encoder:
    def __init__(self):
        self.__cache = {}`,
    question: 'What protection does `__cache` provide?',
    answer: 'Name mangling, not privacy',
    acceptedAnswers: ['_Encoder__cache', 'avoid subclass collision'],
    explanation: 'Python rewrites the name with the defining class, mainly to avoid accidental subclass collisions.',
    detail: 'Double-leading underscores do not enforce access control and can still be reached through the mangled name. A single underscore is the normal convention for non-public APIs.',
  },
  {
    id: 'python-abc-vs-protocol',
    topic: 'Object model & OOP',
    question: 'When should an interface use an `ABC` rather than a typing `Protocol`?',
    answer: 'Shared runtime contract or implementation',
    acceptedAnswers: ['nominal interface', 'runtime inheritance'],
    explanation: 'An abstract base class fits controlled inheritance and shared behavior; a protocol fits structural static compatibility.',
    detail: 'Choose nominal coupling when registration, runtime identity, protected hooks, or reusable code matter. Choose a protocol when consumers only need a shape and third-party types should participate without inheritance.',
  },
  {
    id: 'python-cooperative-super',
    topic: 'Object model & OOP',
    question: 'What must methods do for cooperative multiple inheritance with `super()`?',
    answer: 'Share a compatible signature and forward',
    acceptedAnswers: ['call super cooperatively', 'forward once'],
    explanation: 'Each implementation handles its part and calls `super` so the runtime MRO can visit every participant once.',
    detail: '`super()` means next in the MRO, not named parent. Cooperative classes commonly accept and forward keyword arguments so mixins can consume their own parameters without hard-coding the hierarchy.',
  },
  {
    id: 'python-property-setter-validation',
    topic: 'Object model & OOP',
    code: `class Run:
    def __init__(self) -> None:
        self._progress = 0.0

    @property
    def progress(self) -> float:
        return self._progress

    @progress.setter
    def progress(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError("progress must be between 0 and 1")
        self._progress = value

run = Run()
run.progress = 1.5`,
    question: 'Does the final assignment change `run._progress`?',
    answer: 'No; it raises `ValueError` first',
    acceptedAnswers: ['no', 'ValueError', 'the setter rejects it'],
    explanation: 'A property setter runs code whenever callers assign through the public attribute. This setter checks the range before updating `_progress`, so an invalid value cannot enter the object through `run.progress`.',
    detail: 'A property preserves attribute syntax while owning access. Keep the cost and failure mode attribute-like; an operation involving remote I/O or substantial work should be an explicit method.',
  },
  {
    id: 'python-cached-property-staleness',
    topic: 'Object model & OOP',
    code: `@cached_property
def index(self) -> Index:
    return build_index(self.records)`,
    question: 'What invalidation problem does `cached_property` create?',
    answer: 'The cached value can become stale',
    acceptedAnswers: ['stale cache', 'delete attribute to recompute'],
    explanation: 'After first access, the computed value is stored on the instance and reused until removed.',
    detail: 'Caching changes a derived value into retained state. Use it when dependencies are effectively immutable, or make invalidation explicit by deleting the cached attribute whenever source state changes.',
  },
  {
    id: 'python-slots-inheritance',
    topic: 'Object model & OOP',
    code: `class Point:
    __slots__ = ("x", "y")

class LabeledPoint(Point):
    pass

point = LabeledPoint()
point.label = "origin"`,
    question: 'Does the final assignment succeed?',
    answer: 'Yes',
    acceptedAnswers: ['yes', 'yes, the subclass has __dict__'],
    explanation: 'Slots apply to one class layout, not automatically to every subclass. Because `LabeledPoint` does not declare its own `__slots__`, Python normally gives its instances a `__dict__`, allowing new names such as `label`.',
    detail: 'Slots are a per-class layout decision, not a deep immutability guarantee. Inheritance, weak references, dynamic attributes, and multiple slotted bases complicate the memory benefit.',
  },
  {
    id: 'python-context-manager-suppression',
    topic: 'Object model & OOP',
    code: `class IgnoreErrors:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return True

with IgnoreErrors():
    raise ValueError("bad sample")

continue_training()`,
    question: 'Does execution reach `continue_training()`?',
    answer: 'Yes',
    acceptedAnswers: ['yes', 'the exception is suppressed'],
    explanation: '`__exit__` receives the exception details when the `with` block exits. Returning a truthy value tells Python that the manager handled the error, so the `ValueError` is suppressed and execution continues.',
    detail: 'Cleanup and suppression are separate decisions. Most resource managers release state and return false so failures remain visible; suppress only a narrow exception the abstraction genuinely handles.',
  },
  {
    id: 'python-iterator-protocol-class',
    topic: 'Object model & OOP',
    code: `class BatchCursor:
    def __iter__(self):
        return self

    def __next__(self):
        ...  # return next batch or raise StopIteration

cursor = BatchCursor()
same_object = iter(cursor) is cursor`,
    question: 'What value does `same_object` contain?',
    answer: '`True`',
    acceptedAnswers: ['true', 'True'],
    explanation: 'An iterator implements `__next__` and returns itself from `__iter__`. It is therefore one advancing cursor; iterating it again does not create an independent traversal.',
    detail: 'An iterable may instead create a fresh iterator each time. Separating container from cursor enables independent passes; returning `self` makes traversal state shared and normally one-shot.',
  },
  {
    id: 'python-metaclass-class-creation',
    topic: 'Object model & OOP',
    question: 'What does a Python `metaclass` customize?',
    answer: 'Class creation and behavior',
    acceptedAnswers: ['class of a class'],
    explanation: 'A metaclass constructs class objects and can inspect or transform the namespace during class definition.',
    detail: 'Metaclasses are appropriate for framework-wide class invariants or registration that must happen at definition time. Class decorators and `__init_subclass__` are simpler for many local transformations.',
  },
  {
    id: 'python-init-subclass-hook',
    topic: 'Object model & OOP',
    question: 'When does `__init_subclass__` run?',
    answer: 'Whenever a subclass is created',
    acceptedAnswers: ['subclass definition time'],
    explanation: 'The hook lets a base class validate or register direct and indirect subclasses at class creation time.',
    detail: 'It supplies many metaclass-style extension points without controlling the entire class-construction machinery. Cooperative implementations should accept keyword options and forward unknown ones with `super()`.',
  },

  // Typing
  {
    id: 'python-cast-no-runtime-conversion',
    topic: 'Typing',
    code: `value = cast(Model, payload)`,
    question: 'What does `cast(Model, payload)` do at runtime?',
    answer: 'Returns `payload` unchanged',
    acceptedAnswers: ['nothing', 'no validation'],
    explanation: '`typing.cast` informs a static checker but performs no check, conversion, or copy.',
    detail: 'A cast moves proof responsibility from the checker to the programmer. Use it only when an invariant was established elsewhere; use parsing, `isinstance`, or a validator when runtime data is uncertain.',
  },
  {
    id: 'python-newtype-runtime',
    topic: 'Typing',
    code: `UserId = NewType("UserId", int)
user_id = UserId(7)`,
    question: 'What runtime value and type does `user_id` have?',
    answer: '`7` as an `int`',
    acceptedAnswers: ['int 7', '7'],
    explanation: '`NewType` creates a static distinction, while the runtime call returns its argument with negligible overhead.',
    detail: 'New types prevent accidentally mixing same-representation concepts such as user and experiment IDs in checked code. They do not validate input or create a runtime subclass.',
  },
  {
    id: 'python-type-alias-vs-newtype',
    topic: 'Typing',
    code: `ExperimentId: TypeAlias = int
UserId = NewType("UserId", int)

def load_user(user_id: UserId) -> User: ...

experiment_id: ExperimentId = 7
load_user(experiment_id)`,
    question: 'Which declaration lets a type checker flag the final call?',
    answer: '`NewType`',
    acceptedAnswers: ['NewType', 'UserId'],
    explanation: 'A type alias is only another name for the same static type, so `ExperimentId` remains equivalent to `int`. `NewType` creates a distinct checked identity, allowing the checker to reject an ordinary integer where `UserId` is required.',
    detail: 'Use aliases to explain a complex shape and new types to prevent category mistakes. Neither mechanism validates external data, and both erase to ordinary Python behavior at runtime.',
  },
  {
    id: 'python-overload-implementation',
    topic: 'Typing',
    code: `@overload
def load(raw: bytes) -> Image: ...

@overload
def load(raw: Path) -> Image: ...

def load(raw: bytes | Path) -> Image:
    return decode(raw)`,
    question: 'Which function body executes when `load(path)` runs?',
    answer: 'The final undecorated implementation',
    acceptedAnswers: ['the last load', 'the concrete implementation', 'decode(raw)'],
    explanation: '`@overload` declarations exist for static call checking and have no usable runtime implementation. The final undecorated function accepts every declared case and is the only body Python calls.',
    detail: 'Overloads are useful when return type depends on arguments in ways a union cannot express. The implementation must accept every declared case and still validate unexpected runtime calls as needed.',
  },
  {
    id: 'python-param-spec-decorator',
    topic: 'Typing',
    question: 'Which typing construct preserves an arbitrary wrapped `Callable` signature?',
    answer: '`ParamSpec`',
    acceptedAnswers: ['typing.ParamSpec'],
    explanation: '`ParamSpec` carries positional and keyword parameter types through a higher-order callable.',
    detail: 'A plain `Callable[..., R]` preserves only the return type and loses call checking. Pair `ParamSpec` with a return `TypeVar` when a decorator forwards arguments unchanged.',
  },
  {
    id: 'python-self-return-type',
    topic: 'Typing',
    code: `class Builder:
    def configure(self, **options) -> Self:
        ...`,
    question: 'What relationship does `Self` preserve for subclasses?',
    answer: 'Returns the receiver type',
    acceptedAnswers: ['subclass return type', 'same self type'],
    explanation: '`Self` means the concrete type of the current receiver rather than exactly the declaring base class.',
    detail: 'Fluent methods and alternative constructors can therefore retain subclass types without a manually bound type variable. The implementation must actually return a compatible instance.',
  },
  {
    id: 'python-never-exhaustiveness',
    topic: 'Typing',
    code: `Mode = Literal["train", "eval"]

def run(mode: Mode) -> None:
    if mode == "train":
        train()
    else:
        assert_never(mode)`,
    question: 'Why does a type checker report the final line as an error?',
    answer: '`"eval"` can still reach it',
    acceptedAnswers: ['eval is unhandled', 'the branches are not exhaustive', 'mode can be eval'],
    explanation: '`assert_never` marks a branch that should be statically unreachable. Because the function handles only `"train"`, the remaining `"eval"` member proves that the decision is incomplete.',
    detail: 'The check converts future union growth into a static failure at every incomplete decision site. At runtime, `assert_never` also raises if the impossible path is reached.',
  },
  {
    id: 'python-typeguard-narrowing',
    topic: 'Typing',
    question: 'What does a return annotation such as `TypeGuard[list[str]]` communicate?',
    answer: 'A successful predicate narrows the argument',
    acceptedAnswers: ['user-defined type narrowing', 'narrow on true'],
    explanation: 'When the predicate returns true, a checker treats the tested value as the specified narrower type.',
    detail: 'The annotation does not verify the predicate implementation. A false or incomplete runtime check can make the static model unsound, so the function must establish every claimed invariant.',
  },
  {
    id: 'python-runtime-checkable-protocol-limit',
    topic: 'Typing',
    code: `@runtime_checkable
class Transform(Protocol):
    def __call__(self, sample: Sample) -> Sample: ...

class Broken:
    def __call__(self) -> int:
        return 0

result = isinstance(Broken(), Transform)`,
    question: 'What value can `result` contain despite the incompatible call signature?',
    answer: '`True`',
    acceptedAnswers: ['true', 'True'],
    explanation: 'A runtime-checkable protocol tests whether required attribute names exist, not whether their annotations or signatures match. `Broken` has `__call__`, so the coarse runtime check can pass even though static checking should reject it as a `Transform`.',
    detail: 'A successful check is weaker than static protocol conformance. Use it for coarse feature detection, not as proof that argument types, return types, or semantic behavior match.',
  },
  {
    id: 'python-type-checking-guard',
    topic: 'Typing',
    code: `if TYPE_CHECKING:
    from heavy_package import Model`,
    question: 'When is the guarded import executed?',
    answer: 'By static analysis, not normal runtime',
    acceptedAnswers: ['not at runtime', 'type checker only'],
    explanation: '`TYPE_CHECKING` is false during ordinary execution and understood as true by static type checkers.',
    detail: 'The guard can avoid optional runtime dependencies or import cycles for annotations. Runtime consumers that evaluate annotations may still need the referenced name or a compatible deferred-annotation strategy.',
  },
  {
    id: 'python-generic-invariance',
    topic: 'Typing',
    code: `def add_pet(animals: list[Animal]) -> None:
    animals.append(Cat())

dogs: list[Dog] = [Dog()]
add_pet(dogs)`,
    question: 'Why does a type checker reject the final call?',
    answer: '`add_pet` could insert a `Cat` into `dogs`',
    acceptedAnswers: ['list is invariant', 'it could append a Cat', 'dogs would no longer contain only dogs'],
    explanation: '`list` is mutable: the function can both read animals and insert new ones. Treating `list[Dog]` as `list[Animal]` would allow `Cat()` into a list promised to contain only dogs, so mutable lists are invariant.',
    detail: 'Read-only producers can often be covariant because consumers only observe values. Mutable containers both produce and consume their element type, forcing a stricter invariant relationship.',
  },
  {
    id: 'python-annotation-runtime-resolution',
    topic: 'Typing',
    question: 'Which `typing` API should inspect annotations when forward references or deferred evaluation may exist?',
    answer: '`typing.get_type_hints()`',
    acceptedAnswers: ['get_type_hints'],
    explanation: '`get_type_hints` resolves supported string and forward annotations using the relevant namespaces.',
    detail: 'Reading `__annotations__` directly exposes storage format rather than a stable resolved contract. Resolution can execute code embedded in annotations, so do not process untrusted annotations.',
  },

  // Pydantic v2
  {
    id: 'python-pydantic-output-guarantee',
    topic: 'Pydantic v2',
    code: `class RunConfig(BaseModel):
    epochs: int

raw_epochs = "10"
config = RunConfig(epochs=raw_epochs)`,
    question: 'Why can `raw_epochs` be a `str` while `config.epochs` is an `int`?',
    answer: 'Pydantic guarantees the validated output, not an unchanged input',
    acceptedAnswers: ['Pydantic coerces input', 'validated output type', 'validation converts the string'],
    explanation: 'Default Pydantic validation may parse, copy, or coerce incoming values. Its contract is that the stored model value matches the declared field after validation, not that the original representation remains unchanged.',
    detail: 'This distinction explains why “validation” can transform data. If preserving exact input representation matters, select strict fields or retain the raw payload separately rather than assuming validation is observation-only.',
  },
  {
    id: 'python-pydantic-required-optional',
    topic: 'Pydantic v2',
    code: `class Job(BaseModel):
    cache_dir: Path | None`,
    question: 'Is `cache_dir` required when it accepts `None`?',
    answer: 'Yes',
    explanation: 'The union permits a null value, but omission requires a default such as `= None`.',
    detail: 'Pydantic v2 follows the typing distinction between nullable and optional-to-supply. State the default explicitly so schemas, call signatures, and runtime behavior agree.',
  },
  {
    id: 'python-pydantic-validate-default',
    topic: 'Pydantic v2',
    code: `class RetryPolicy(BaseModel):
    retries: int = Field(default="ten", validate_default=True)

policy = RetryPolicy()`,
    question: 'What happens when the caller omits `retries`?',
    answer: '`ValidationError`',
    acceptedAnswers: ['ValidationError', 'the invalid default is rejected'],
    explanation: '`validate_default=True` sends the declared default through the normal field validator when callers omit that field. Because `"ten"` cannot become an integer, even the no-argument construction fails.',
    detail: 'A default is authored code but can still drift away from its annotation during refactoring. Validating defaults closes the gap between explicitly supplied and omitted values at some construction cost.',
  },
  {
    id: 'python-pydantic-alias-boundary',
    topic: 'Pydantic v2',
    code: `class Config(BaseModel):
    model_name: str = Field(
        validation_alias="modelName",
        serialization_alias="model_name",
    )

config = Config.model_validate({"modelName": "resnet"})
payload = config.model_dump(by_alias=True)`,
    question: 'Which key appears in `payload`: `modelName` or `model_name`?',
    answer: '`model_name`',
    acceptedAnswers: ['model_name', '"model_name"'],
    explanation: 'A validation alias names the key accepted on input; a serialization alias names the key emitted when dumping by alias. The model can therefore read the old external spelling while writing the new one.',
    detail: 'Separate aliases are useful during schema migration because read and write contracts can move independently. Test both directions and avoid ambiguous alias collisions between fields.',
  },
  {
    id: 'python-pydantic-exclude-unset',
    topic: 'Pydantic v2',
    code: `class Patch(BaseModel):
    lr: float = 1e-3
    batch_size: int = 32

patch = Patch(lr=1e-4)
payload = patch.model_dump(exclude_unset=True)`,
    question: 'What value does `payload` contain?',
    answer: '`{"lr": 1e-4}`',
    acceptedAnswers: ['{"lr": 0.0001}', 'lr only', '{"lr": 1e-4}'],
    explanation: '`exclude_unset=True` keeps fields explicitly provided by the caller and omits fields that only received defaults. That preserves the PATCH distinction between “change `lr`” and “also overwrite `batch_size` with its default.”',
    detail: 'A partial update must distinguish “leave unchanged” from “write the default.” Explicit-field tracking preserves that intent, though later assignment can change which fields count as set.',
  },
  {
    id: 'python-pydantic-python-json-mode',
    topic: 'Pydantic v2',
    code: `event = Event(created_at=datetime(2026, 8, 26, tzinfo=UTC))
python_value = event.model_dump(mode="python")["created_at"]
json_value = event.model_dump(mode="json")["created_at"]`,
    question: 'What runtime types do `python_value` and `json_value` have?',
    answer: '`datetime` and `str`',
    acceptedAnswers: ['datetime, string', 'datetime and str'],
    explanation: 'Python mode retains native Python objects such as `datetime`. JSON mode converts supported fields to JSON-compatible values, so the same timestamp becomes a string suitable for a JSON encoder.',
    detail: 'Serialization mode is a representation boundary, not merely formatting. Choose it from the next consumer rather than dumping to JSON and parsing back to obtain normalized Python data.',
  },
  {
    id: 'python-pydantic-root-model',
    topic: 'Pydantic v2',
    code: `class Labels(RootModel[list[str]]):
    pass

labels = Labels.model_validate(["cat", "dog"])
payload = labels.model_dump()`,
    question: 'What value does `payload` contain?',
    answer: '`["cat", "dog"]`',
    acceptedAnswers: ['[cat, dog]', '["cat", "dog"]'],
    explanation: 'A root model validates one list, mapping, or scalar as the model itself. Dumping `Labels` therefore returns the bare list rather than wrapping it in an object such as `{"labels": [...]}`.',
    detail: 'Use a root model only when the wire schema truly is the contained value. A named field is easier to extend later with sibling metadata without breaking the payload shape.',
  },
  {
    id: 'python-pydantic-from-attributes',
    topic: 'Pydantic v2',
    code: `class UserSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    name: str

row = UserRow(name="Arun")
user = UserSchema.model_validate(row)`,
    question: 'Where does Pydantic read the value for `user.name`?',
    answer: 'From `row.name`',
    acceptedAnswers: ['row.name', 'the object attribute', 'from the attribute'],
    explanation: '`from_attributes=True` lets model validation read named attributes from an object instead of requiring a dictionary. This is useful for ORM rows, but attribute access may trigger lazy I/O or descriptor code.',
    detail: 'Attribute access can trigger descriptors, lazy database loads, or exceptions. Treat ORM-to-schema conversion as an I/O boundary and control what has already been loaded.',
  },
  {
    id: 'python-pydantic-revalidate-instances',
    topic: 'Pydantic v2',
    question: 'Why can passing an existing model instance to `model_validate` preserve invalid mutated state?',
    answer: 'Instances are not revalidated by default',
    acceptedAnswers: ['revalidate_instances never', 'trusted instance'],
    explanation: 'Pydantic normally assumes an existing instance is valid unless `revalidate_instances` is configured differently.',
    detail: 'Construction validation does not protect a mutable object forever. Combine assignment validation, frozen models, or instance revalidation according to where mutation is allowed and where trust changes.',
  },
  {
    id: 'python-pydantic-validator-order',
    topic: 'Pydantic v2',
    code: `class Config(BaseModel):
    epochs: int

    @field_validator("epochs", mode="before")
    @classmethod
    def inspect_epochs(cls, value):
        print(type(value).__name__)
        return value

Config(epochs="10")`,
    question: 'What does the validator print?',
    answer: '`str`',
    acceptedAnswers: ['str', 'string'],
    explanation: 'A before-validator receives the raw input before Pydantic parses the annotated type. An after-validator would instead receive the converted integer `10`.',
    detail: 'Raw input can have any shape, so before validators must be defensive. Prefer after validators for type-safe invariants and use before mode only when normalization must precede parsing.',
  },
  {
    id: 'python-pydantic-validation-context',
    topic: 'Pydantic v2',
    question: 'How can a Pydantic `validator` receive request-specific policy without storing it as a model field?',
    answer: 'Validation context',
    acceptedAnswers: ['context argument', 'ValidationInfo.context'],
    explanation: 'Pass `context` to a validation method and read it through `ValidationInfo` inside validators.',
    detail: 'Context keeps external policy separate from domain data, but it makes validation depend on an extra input. Test each policy and avoid hidden global state inside validators.',
  },
  {
    id: 'python-pydantic-computed-field',
    topic: 'Pydantic v2',
    code: `class Rectangle(BaseModel):
    width: float
    height: float

    @computed_field
    @property
    def area(self) -> float:
        return self.width * self.height

payload = Rectangle(width=3, height=4).model_dump()`,
    question: 'What value does `payload["area"]` contain?',
    answer: '`12.0`',
    acceptedAnswers: ['12', '12.0'],
    explanation: '`@computed_field` makes a derived property participate in serialization and the serialization schema. `area` is calculated from validated fields when the model is dumped; callers do not supply it as input state.',
    detail: 'A computed field derives output rather than accepting validated input. Keep it deterministic and cheap enough for serialization, and avoid pretending a derived value is caller-controlled state.',
  },
  {
    id: 'python-pydantic-model-serializer',
    topic: 'Pydantic v2',
    question: 'When should code use `@model_serializer` rather than several field serializers?',
    answer: 'Whole-model output logic',
    acceptedAnswers: ['cross-field serialization', 'model serializer'],
    explanation: 'A model serializer owns output whose representation depends on several fields or replaces the default model shape.',
    detail: 'Serialization is a public boundary contract. Prefer field-local serializers when fields are independent; a whole-model serializer is more powerful but can obscure schema correspondence and exclusions.',
  },
  {
    id: 'python-pydantic-json-schema',
    topic: 'Pydantic v2',
    question: 'Which `BaseModel` API produces its JSON Schema?',
    answer: '`model_json_schema()`',
    acceptedAnswers: ['model_json_schema'],
    explanation: 'The class method returns a JSON-compatible schema dictionary derived from fields, constraints, unions, and metadata.',
    detail: 'Generated schema documents the validator but does not prove every custom semantic rule is representable. Keep prose descriptions or downstream contract tests for invariants JSON Schema cannot express.',
  },
  {
    id: 'python-pydantic-concrete-collections',
    topic: 'Pydantic v2',
    code: `class Batch(BaseModel):
    sample_ids: list[int]

batch = Batch(sample_ids=(1, 2, 3))`,
    question: 'What concrete runtime type does `batch.sample_ids` have?',
    answer: '`list`',
    acceptedAnswers: ['list', 'a list'],
    explanation: 'The field annotation describes Pydantic\'s normalized output as well as acceptable input. Pydantic can accept the tuple and convert it to a concrete list without the broader checks needed to preserve an abstract `Sequence` result.',
    detail: 'Field annotations describe the normalized result as well as accepted input. A concrete collection reduces ambiguity and validation work; use an abstraction only when preserving that broader output contract matters.',
  },

  // Concurrency and parallelism
  {
    id: 'python-gil-practical-boundary',
    topic: 'Concurrency & parallelism',
    question: 'What does the traditional CPython `GIL` serialize?',
    answer: 'Execution of Python bytecode in one process',
    acceptedAnswers: ['one thread executes Python bytecode'],
    explanation: 'A GIL-enabled interpreter allows one thread at a time to execute Python bytecode, while blocking I/O and native code may release it.',
    detail: 'The GIL is neither a user-data lock nor a ban on all parallel work. Threads can overlap I/O, and C extensions can run native kernels concurrently; shared invariants still require synchronization.',
  },
  {
    id: 'python-free-threaded-build',
    topic: 'Concurrency & parallelism',
    question: 'What changes in a free-threaded `CPython` build?',
    answer: 'The GIL can be disabled',
    acceptedAnswers: ['threads can execute Python in parallel'],
    explanation: 'Free-threaded builds can run Python threads across cores, although incompatible extensions may re-enable the GIL.',
    detail: 'Parallel bytecode exposes races that accidental GIL serialization may have hidden. Write explicit synchronization now, test dependencies for free-threaded support, and measure its workload-specific overhead and benefit.',
  },
  {
    id: 'python-gil-no-race-safety',
    topic: 'Concurrency & parallelism',
    code: `if key not in cache:
    cache[key] = build(key)`,
    question: 'Why is this check-then-write unsafe even on a GIL-enabled interpreter?',
    answer: 'The logical operation is not atomic',
    acceptedAnswers: ['race condition', 'thread switch between steps'],
    explanation: 'Another thread can interleave after the check, and `build` may release the GIL or take arbitrary time.',
    detail: 'Thread safety belongs to the invariant spanning several operations, not to isolated container methods. Guard the whole transition, tolerate duplicate construction deliberately, or centralize ownership.',
  },
  {
    id: 'python-lock-context-manager',
    topic: 'Concurrency & parallelism',
    question: 'Why should `threading.Lock` normally be used with `with lock:`?',
    answer: 'Guaranteed release',
    acceptedAnswers: ['release in finally', 'context manager'],
    explanation: 'The context manager releases the lock on normal completion, return, or exception.',
    detail: 'Keep the critical section as small as the protected invariant allows, but not smaller. Releasing between a check and its update preserves responsiveness while destroying correctness.',
  },
  {
    id: 'python-rlock-reentrancy',
    topic: 'Concurrency & parallelism',
    question: 'When is `threading.RLock` required instead of `Lock`?',
    answer: 'The same thread must acquire it recursively',
    acceptedAnswers: ['reentrant acquisition', 'nested locked calls'],
    explanation: 'An RLock tracks owning thread and recursion depth; it releases fully only after matching releases.',
    detail: 'Reentrancy prevents self-deadlock when a locked method calls another method using the same lock. It does not permit other threads to enter or repair a confused ownership design.',
  },
  {
    id: 'python-condition-predicate-loop',
    topic: 'Concurrency & parallelism',
    code: `with condition:
    condition.wait_for(lambda: bool(queue))
    item = queue.popleft()`,
    question: 'Why must a condition wait recheck a predicate?',
    answer: 'Wake-up does not guarantee the state',
    acceptedAnswers: ['spurious wakeup', 'another thread consumed state'],
    explanation: 'A waiter may wake spuriously or lose the desired state to another thread before reacquiring the lock.',
    detail: 'A condition combines a lock, a state predicate, and notification. Notifications announce that state may have changed; the predicate, checked while holding the lock, decides whether progress is valid.',
  },
  {
    id: 'python-event-level-trigger',
    topic: 'Concurrency & parallelism',
    question: 'What coordination contract does `threading.Event` provide?',
    answer: 'A shared boolean signal',
    acceptedAnswers: ['set clear wait flag'],
    explanation: 'Once set, all current and future waiters pass until another thread explicitly clears the event.',
    detail: 'An event communicates state rather than transferring ownership or counting occurrences. It fits shutdown and readiness signals; use a queue or semaphore when every individual occurrence matters.',
  },
  {
    id: 'python-bounded-semaphore',
    topic: 'Concurrency & parallelism',
    question: 'Why use `BoundedSemaphore` for a pool of ten connections?',
    answer: 'Detect over-release',
    acceptedAnswers: ['capacity limit and over release error'],
    explanation: 'It limits concurrent acquisitions and raises if releases would exceed the initial capacity.',
    detail: 'A semaphore counts permits rather than protecting one owner. The bounded form turns an accounting bug into an immediate error, which is valuable when permits represent finite external resources.',
  },
  {
    id: 'python-barrier-parties',
    topic: 'Concurrency & parallelism',
    question: 'What happens when all parties call `threading.Barrier.wait()`?',
    answer: 'They are released together',
    acceptedAnswers: ['barrier opens'],
    explanation: 'A fixed number of threads rendezvous at the barrier, which can be reused for later phases.',
    detail: 'A barrier coordinates phases, not mutual exclusion. Use a timeout and handle `BrokenBarrierError` so one failed participant does not leave every other thread blocked forever.',
  },
  {
    id: 'python-thread-daemon-exit',
    topic: 'Concurrency & parallelism',
    question: 'What guarantee is missing for a `daemon` thread at interpreter shutdown?',
    answer: 'Graceful completion and cleanup',
    acceptedAnswers: ['abruptly stopped', 'not joined'],
    explanation: 'The process may exit when only daemon threads remain, without waiting for their work or cleanup.',
    detail: 'Daemon status is suitable for disposable background assistance, not durable writes. Production workers need a stop signal, exception reporting, and an explicit `join` during shutdown.',
  },
  {
    id: 'python-thread-join-exception',
    topic: 'Concurrency & parallelism',
    question: 'Does `thread.join()` re-raise an exception that escaped the target function?',
    answer: 'No',
    explanation: 'Join waits for termination; uncaught thread exceptions go through `threading.excepthook` rather than the joining thread.',
    detail: 'A raw thread has no result channel. Use a `Future`, queue, or explicit error holder when the parent must observe success, return values, and failures as part of the task contract.',
  },
  {
    id: 'python-queue-task-done',
    topic: 'Concurrency & parallelism',
    question: 'What must each `queue.Queue.get()` consumer call so `queue.join()` can finish?',
    answer: '`task_done()`',
    explanation: 'The queue tracks unfinished work separately from item removal; each completed item must decrement that count.',
    detail: 'Call `task_done` in a `finally` block after ownership transfers to the worker. Forgetting it blocks join forever; calling it too often raises because accounting becomes inconsistent.',
  },
  {
    id: 'python-thread-local-vs-contextvar',
    topic: 'Concurrency & parallelism',
    question: 'Why is `ContextVar` usually better than thread-local storage for async request state?',
    answer: 'Context follows async tasks',
    acceptedAnswers: ['task-local context', 'thread local mixes coroutines'],
    explanation: 'Several coroutines share one event-loop thread, while each task receives its own context-variable view.',
    detail: 'Thread-local state is keyed by operating-system thread and works for thread-bound code. Context variables follow logical execution contexts across awaits and can be copied into other execution boundaries deliberately.',
  },
  {
    id: 'python-threadpool-future-exception',
    topic: 'Concurrency & parallelism',
    question: 'When does a `ThreadPoolExecutor` task exception reach the caller?',
    answer: 'When its future result is retrieved',
    acceptedAnswers: ['future.result()', 'result call'],
    explanation: 'The executor stores the exception in the `Future`; `future.result()` re-raises it in the observing thread.',
    detail: 'Submitting work is not observing work. Retain futures and inspect results, use `as_completed`, or consume `map` output so background failures cannot disappear behind successful submission.',
  },
  {
    id: 'python-as-completed-order',
    topic: 'Concurrency & parallelism',
    question: 'How does `as_completed(futures)` order results?',
    answer: 'Completion order',
    acceptedAnswers: ['finish order'],
    explanation: 'It yields each future as that task finishes rather than preserving submission order.',
    detail: 'Completion order reduces latency for streaming independent results. If downstream output must be deterministic, retain an input index and restore order at the presentation boundary.',
  },
  {
    id: 'python-executor-context-cleanup',
    topic: 'Concurrency & parallelism',
    question: 'What does leaving `with ThreadPoolExecutor() as executor:` guarantee?',
    answer: 'Shutdown and wait for pending work',
    acceptedAnswers: ['executor shutdown wait'],
    explanation: 'The context manager calls shutdown and waits for submitted tasks to complete before exiting.',
    detail: 'Resource lifetime includes worker threads and queued tasks, not only the executor object. For early failure, decide whether pending futures should finish, be cancelled, or receive cooperative cancellation.',
  },
  {
    id: 'python-processpool-pickling',
    topic: 'Concurrency & parallelism',
    question: 'Why do nested functions and lambdas often fail in `ProcessPoolExecutor`?',
    answer: 'Workers need picklable importable callables',
    acceptedAnswers: ['not picklable', 'top-level function required'],
    explanation: 'Process workers serialize tasks and generally need to import a top-level function by module name.',
    detail: 'A process boundary copies or serializes arguments, results, and errors. Design a small explicit message contract and keep large shared read-only data out of every task payload when possible.',
  },
  {
    id: 'python-process-start-method',
    topic: 'Concurrency & parallelism',
    question: 'Why should code not assume all `multiprocessing` workers inherit the parent memory state?',
    answer: 'Start methods differ',
    acceptedAnswers: ['spawn vs fork', 'fresh interpreter'],
    explanation: 'Spawn and forkserver create fresh interpreter state, while fork copies the parent process with different safety trade-offs.',
    detail: 'Portable multiprocessing initializes worker state explicitly and protects process creation with the main guard. Relying on inherited globals creates platform-specific behavior and can copy unsafe threaded-library state.',
  },
  {
    id: 'python-async-blocking-call',
    topic: 'Concurrency & parallelism',
    code: `async def handler():
    data = requests.get(url).json()
    return data`,
    question: 'Why does this coroutine block unrelated tasks?',
    answer: '`requests.get` blocks the event-loop thread',
    acceptedAnswers: ['blocking I/O in async', 'not async client'],
    explanation: '`async def` alone creates no concurrency; control switches only when the task reaches a cooperating await.',
    detail: 'Every call on the async path must respect the event-loop contract. Use an asynchronous client or move unavoidable blocking work with `asyncio.to_thread` while preserving timeouts and cancellation expectations.',
  },
  {
    id: 'python-async-create-task-reference',
    topic: 'Concurrency & parallelism',
    question: 'Why should fire-and-forget `asyncio.create_task()` calls be retained and observed?',
    answer: 'Keep task lifetime and exceptions visible',
    acceptedAnswers: ['strong reference', 'observe failures'],
    explanation: 'Unstructured background tasks can outlive their owner, lose exceptions, or be garbage-collected without a clear shutdown path.',
    detail: 'Prefer structured concurrency with `TaskGroup`. When detached work is intentional, keep a strong-reference set, remove completed tasks, report failures, and cancel or await them at shutdown.',
  },
  {
    id: 'python-taskgroup-failure',
    topic: 'Concurrency & parallelism',
    question: 'What happens when one child in an `asyncio.TaskGroup` fails?',
    answer: 'Sibling tasks are cancelled and failures are grouped',
    acceptedAnswers: ['structured cancellation', 'ExceptionGroup'],
    explanation: 'The group waits for all children, cancels remaining work after a non-cancellation failure, and raises collected errors as an exception group.',
    detail: 'Structured concurrency makes child lifetime match lexical scope. Failure cannot silently strand sibling tasks, and the caller receives a complete error boundary instead of hunting detached tasks.',
  },
  {
    id: 'python-gather-failure-semantics',
    topic: 'Concurrency & parallelism',
    question: 'Does default `asyncio.gather` cancel every other awaitable when one raises?',
    answer: 'No',
    explanation: 'The first exception is propagated to the waiter, but other submitted awaitables normally continue unless separately cancelled.',
    detail: 'This differs from task-group fail-fast structure. Choose gather when independent results may continue and you own their observation; choose a task group when sibling fate should be coupled.',
  },
  {
    id: 'python-cancelled-error-cleanup',
    topic: 'Concurrency & parallelism',
    question: 'What should a coroutine do after catching `asyncio.CancelledError` for cleanup?',
    answer: 'Normally re-raise it',
    acceptedAnswers: ['propagate cancellation'],
    explanation: 'Cancellation is a control-flow request; swallowing it can break task groups, timeouts, and shutdown.',
    detail: 'Put idempotent cleanup in `finally`, then let cancellation propagate unless the coroutine deliberately converts the contract. Cancellation can arrive at any await, so protect invariants across suspension points.',
  },
  {
    id: 'python-async-timeout-cancellation',
    topic: 'Concurrency & parallelism',
    question: 'What does an `asyncio.timeout(...)` context do when its deadline expires?',
    answer: 'Cancels the current task and raises `TimeoutError` outside',
    acceptedAnswers: ['timeout cancellation'],
    explanation: 'The context converts its internal cancellation into `TimeoutError` when control exits the timeout block.',
    detail: 'A timeout is a cancellation boundary, so called coroutines must be cancellation-safe. It does not guarantee an external service stopped work unless that protocol also supports cancellation.',
  },
  {
    id: 'python-asyncio-lock-thread-limit',
    topic: 'Concurrency & parallelism',
    question: 'Can `asyncio.Lock` safely coordinate ordinary threads?',
    answer: 'No',
    explanation: 'Asyncio synchronization primitives coordinate tasks on an event loop and are not thread-safe primitives.',
    detail: 'Use `threading.Lock` for cross-thread shared memory. Crossing from a thread into an event loop requires thread-safe scheduling APIs; do not mix primitives merely because their method names match.',
  },

  // Performance and optimization
  {
    id: 'python-profile-before-optimize',
    topic: 'Performance & optimization',
    question: 'What should happen before optimizing a suspected Python `hot path`?',
    answer: 'Measure with representative profiling',
    acceptedAnswers: ['profile first', 'benchmark'],
    explanation: 'A profiler identifies where elapsed time, allocation, or blocking actually accumulates under realistic input.',
    detail: 'Optimization spends complexity budget. Define the target metric, preserve a correctness test, benchmark representative workloads, and keep the simpler implementation when the measured improvement is irrelevant.',
  },
  {
    id: 'python-string-join',
    topic: 'Performance & optimization',
    code: `text = "".join(parts)`,
    question: 'Why is `join` preferred for many string fragments?',
    answer: 'One planned allocation',
    acceptedAnswers: ['avoids repeated copies', 'linear construction'],
    explanation: 'Strings are immutable, so repeated concatenation can allocate and copy growing intermediate results.',
    detail: '`join` makes the separator and collection boundary explicit while allowing the runtime to size the result. Small fixed concatenations remain readable and may already be optimized.',
  },
  {
    id: 'python-generator-sum-memory',
    topic: 'Performance & optimization',
    question: 'What does `sum(transform(x) for x in items)` save over summing a list comprehension?',
    answer: 'Temporary list memory',
    acceptedAnswers: ['streaming memory'],
    explanation: 'The generator produces one transformed value at a time for the reduction.',
    detail: 'Laziness reduces peak memory but does not make the transformation itself faster, and Python generator resumption has overhead. Measure when the input is small or a vectorized native operation exists.',
  },
  {
    id: 'python-localize-membership-index',
    topic: 'Performance & optimization',
    code: `allowed = set(allowed_ids)
selected = [item for item in items if item.id in allowed]`,
    question: 'When does building `allowed` repay its cost?',
    answer: 'When membership is repeated enough',
    acceptedAnswers: ['many lookups', 'amortize set build'],
    explanation: 'A one-time `O(m)` set build can replace repeated linear scans with average constant-time lookups.',
    detail: 'Include index construction, memory, hashing, and reuse in the cost model. Converting a tiny list for one membership test adds work without changing the meaningful bottleneck.',
  },
  {
    id: 'python-copy-avoidance-aliasing',
    topic: 'Performance & optimization',
    question: 'What correctness cost accompanies avoiding a `defensive copy`?',
    answer: 'Shared mutable aliasing',
    acceptedAnswers: ['caller and callee share state'],
    explanation: 'Reusing storage saves allocation and bandwidth but lets either owner observe or cause later mutation.',
    detail: 'Copy avoidance is an ownership decision. Document who may mutate and how long the view remains valid; copy at trust boundaries where isolation matters more than throughput.',
  },
  {
    id: 'python-slots-memory-tradeoff',
    topic: 'Performance & optimization',
    question: 'When can `__slots__` materially reduce memory?',
    answer: 'Many small instances with fixed fields',
    acceptedAnswers: ['avoid per instance dict'],
    explanation: 'Slots can replace each instance attribute dictionary with a fixed layout.',
    detail: 'The gain depends on object count and field shape, while inheritance and tooling costs are permanent. Measure total resident memory before redesigning ordinary classes around slots.',
  },
  {
    id: 'python-cache-memory-retention',
    topic: 'Performance & optimization',
    question: 'What memory risk does an unbounded `@cache` create?',
    answer: 'It retains every key and result',
    acceptedAnswers: ['unbounded growth', 'memory leak-like retention'],
    explanation: 'The cache holds strong references to arguments and return values until cleared or the function is discarded.',
    detail: 'Caching shifts cost from compute to memory and invalidation. Bound cardinality with `lru_cache(maxsize=...)`, expose cache metrics, and clear state when the underlying world changes.',
  },
  {
    id: 'python-builtins-native-loop',
    topic: 'Performance & optimization',
    question: 'Why can built-ins such as `sum`, `min`, and `any` outperform an equivalent Python loop?',
    answer: 'Less Python-level dispatch',
    acceptedAnswers: ['implemented in optimized runtime code', 'C loop'],
    explanation: 'A built-in can keep iteration and reduction inside optimized interpreter or native implementation paths.',
    detail: 'Prefer a built-in when it expresses the exact operation; clarity and speed align. Do not contort domain logic into dense built-in combinations whose semantics are harder to verify.',
  },
  {
    id: 'python-vectorization-boundary',
    topic: 'Performance & optimization',
    question: 'Why can a native `vectorized` operation beat a Python element loop?',
    answer: 'Work stays in compiled kernels',
    acceptedAnswers: ['amortizes interpreter overhead', 'native loop'],
    explanation: 'One Python call can dispatch many element operations to contiguous optimized code that may release the GIL.',
    detail: 'Vectorization can add large temporaries or force awkward layouts. Include allocation, memory bandwidth, and transfer costs rather than counting only fewer lines of Python.',
  },
  {
    id: 'python-microbenchmark-setup',
    topic: 'Performance & optimization',
    question: 'Why should `timeit` setup exclude unrelated construction from the timed statement?',
    answer: 'Measure only the target operation',
    acceptedAnswers: ['isolate benchmark'],
    explanation: 'Repeated setup work can dominate a microbenchmark and reverse the apparent comparison.',
    detail: 'A microbenchmark answers a narrow mechanism question, not end-to-end latency. Control warm-up, input size, cache state, and result use, then confirm important gains in the real workload.',
  },
  {
    id: 'python-logging-lazy-format',
    topic: 'Performance & optimization',
    code: `logger.debug("batch=%s stats=%s", batch_id, expensive_stats)`,
    question: 'What formatting work does parameterized logging defer?',
    answer: 'Message interpolation',
    acceptedAnswers: ['string formatting until emitted'],
    explanation: 'The logging framework formats the template only if a handler emits the record at that level.',
    detail: 'Arguments themselves are still evaluated before the call, so `expensive_stats()` would run even when debug is disabled. Guard genuinely expensive diagnostic computation with a level check.',
  },
  {
    id: 'python-process-overhead-break-even',
    topic: 'Performance & optimization',
    question: 'Why can a `ProcessPoolExecutor` slow down many tiny CPU tasks?',
    answer: 'Serialization and scheduling dominate',
    acceptedAnswers: ['process overhead', 'pickling overhead'],
    explanation: 'Each task crosses process boundaries and competes for startup, queueing, serialization, and result-transfer time.',
    detail: 'Parallel speedup requires enough independent work per message. Batch small items, initialize reusable worker state once, and measure wall time including data movement rather than worker compute alone.',
  },

  // Runtime, imports, and testing
  {
    id: 'python-import-executes-once',
    topic: 'Runtime, imports & testing',
    question: 'Why does `import` of the same module twice normally not rerun its top-level code?',
    answer: '`sys.modules` cache',
    acceptedAnswers: ['module cache'],
    explanation: 'The first import creates and executes a module object; later imports normally reuse the cached object.',
    detail: 'A module is shared process state, so import-time registries and singletons persist across importers. Tests that mutate them need explicit reset boundaries rather than assuming a fresh import.',
  },
  {
    id: 'python-main-module-guard',
    topic: 'Runtime, imports & testing',
    question: 'What does `if __name__ == "__main__":` separate?',
    answer: 'Script execution from import behavior',
    acceptedAnswers: ['entrypoint guard'],
    explanation: 'The guarded block runs when the file is the entry module, not when another module imports it normally.',
    detail: 'Keep reusable definitions import-safe and put side effects behind a small main function. The boundary also prevents recursive process creation under multiprocessing spawn.',
  },
  {
    id: 'python-circular-import-partial-module',
    topic: 'Runtime, imports & testing',
    question: 'Why can a circular import observe a missing name from a module that exists in `sys.modules`?',
    answer: 'The module is only partially initialized',
    acceptedAnswers: ['import cycle', 'top-level execution not finished'],
    explanation: 'Python caches the module object before finishing its body, so the cycle can return that incomplete namespace.',
    detail: 'Import cycles usually reveal ownership that points both ways. Move shared contracts to a lower-level module, depend on interfaces, or defer a narrow import only when restructuring is unjustified.',
  },
  {
    id: 'python-import-star-all',
    topic: 'Runtime, imports & testing',
    question: 'What does module-level `__all__` control?',
    answer: 'Names exported by wildcard import',
    acceptedAnswers: ['from module import star'],
    explanation: 'It defines the public names copied by `from module import *`; it does not make other attributes private.',
    detail: 'Explicit imports keep dependencies searchable and avoid collisions, so wildcard imports are rarely suitable for application code. `__all__` remains useful for a curated package facade.',
  },
  {
    id: 'python-reference-count-cycles',
    topic: 'Runtime, imports & testing',
    question: 'Why does CPython need cyclic `garbage collection` in addition to reference counting?',
    answer: 'Cycles can keep nonzero reference counts',
    acceptedAnswers: ['unreachable cycles'],
    explanation: 'Objects that reference each other can become unreachable while each still holds the other alive by count.',
    detail: 'Object-finalization timing is an implementation detail and cycles delay it further. Use context managers for files, locks, sockets, and transactions whose release must be deterministic.',
  },
  {
    id: 'python-weakref-cache',
    topic: 'Runtime, imports & testing',
    question: 'When is `weakref.WeakValueDictionary` useful for a cache?',
    answer: 'Cached values should not extend object lifetime',
    acceptedAnswers: ['weak cache', 'garbage collect when no strong refs'],
    explanation: 'Entries disappear when no strong references to their values remain elsewhere.',
    detail: 'Weak references are lifecycle observation, not deterministic eviction. An object can vanish between accesses, and not every built-in type supports weak references, so callers must tolerate cache misses.',
  },
  {
    id: 'python-mock-where-used',
    topic: 'Runtime, imports & testing',
    code: `from client import fetch

def load():
    return fetch()`,
    question: 'Where should a test patch `fetch` for this module?',
    answer: 'Where the name is looked up',
    acceptedAnswers: ['patch module_under_test.fetch', 'where used'],
    explanation: 'The module under test holds its own binding, so patching the original definition may not replace that local reference.',
    detail: 'Mocking follows Python name binding, not conceptual ownership. Prefer injecting dependencies when practical; it makes the seam explicit and reduces brittle patch-path knowledge.',
  },
  {
    id: 'python-test-observable-behavior',
    topic: 'Runtime, imports & testing',
    question: 'Why should a `unit test` prefer public behavior over private call counts?',
    answer: 'Refactoring should not break the contract test',
    acceptedAnswers: ['test behavior not implementation'],
    explanation: 'Private interaction assertions couple the test to one implementation even when externally visible behavior remains correct.',
    detail: 'Mock an interaction when that interaction is itself the contract, such as one durable write or no network call. Otherwise assert outputs, state transitions, and boundary errors.',
  },
  {
    id: 'python-random-seed-scope',
    topic: 'Runtime, imports & testing',
    question: 'Why is setting only `random.seed(0)` insufficient for full ML pipeline reproducibility?',
    answer: 'Libraries and workers have separate RNG state',
    acceptedAnswers: ['multiple random generators', 'not all nondeterminism'],
    explanation: 'NumPy, PyTorch, subprocesses, workers, and nondeterministic algorithms can each require separate controls.',
    detail: 'A seed initializes a generator; it does not serialize scheduling, filesystem order, library versions, or hardware kernels. Record the complete reproducibility envelope and expected tolerance.',
  },
  {
    id: 'python-warning-vs-exception',
    topic: 'Runtime, imports & testing',
    question: 'When should an API emit `warnings.warn` instead of raising an exception?',
    answer: 'The operation remains valid but needs attention',
    acceptedAnswers: ['nonfatal compatibility issue', 'deprecation'],
    explanation: 'Warnings report conditions such as deprecation while allowing the current operation to finish.',
    detail: 'A warning is filterable process-level communication, so include a precise category and useful stack level. Invalid state or unsafe continuation still requires an exception.',
  },

  // Reliability, errors, and I/O
  {
    id: 'python-bare-except-baseexception',
    topic: 'Reliability & I/O',
    question: 'Why is bare `except:` broader than `except Exception:`?',
    answer: 'It also catches `BaseException` control signals',
    acceptedAnswers: ['KeyboardInterrupt SystemExit'],
    explanation: 'A bare handler catches interrupts and interpreter-exit signals in addition to ordinary application failures.',
    detail: 'Catch only failures the boundary can recover from, translate, or retry. Broad handling hides programming defects and can make a process ignore the user or supervisor asking it to stop.',
  },
  {
    id: 'python-finally-return-overrides',
    topic: 'Reliability & I/O',
    code: `def result():
    try:
        return 1
    finally:
        return 2`,
    question: 'What value does `result()` return?',
    answer: '`2`',
    acceptedAnswers: ['2'],
    explanation: 'A control-flow statement in `finally` overrides the pending return or exception from `try`.',
    detail: 'Use `finally` for cleanup, not a second decision about outcome. Returning or raising there can silently erase the failure the caller most needs to see.',
  },
  {
    id: 'python-exception-group-except-star',
    topic: 'Reliability & I/O',
    question: 'What does `except* OSError` handle inside an `ExceptionGroup`?',
    answer: 'All matching subgroup exceptions',
    acceptedAnswers: ['matching OSErrors'],
    explanation: 'The group is split so the handler receives matching failures while unmatched failures continue propagating.',
    detail: 'Concurrent work can fail in several independent places, so one linear exception is incomplete. Preserve the group structure and handle only the subset for which one recovery policy is valid.',
  },
  {
    id: 'python-eafp-race',
    topic: 'Reliability & I/O',
    code: `if path.exists():
    data = path.read_bytes()`,
    question: 'Why can this look-before-you-leap check still fail?',
    answer: 'The filesystem can change after the check',
    acceptedAnswers: ['TOCTOU race', 'time of check time of use'],
    explanation: 'Another actor can remove or replace the path between `exists` and `read_bytes`.',
    detail: 'Attempt the operation and handle the specific failure when the operation itself defines truth. Prechecks remain useful for user guidance but cannot provide an atomic external guarantee.',
  },
  {
    id: 'python-atomic-file-replace',
    topic: 'Reliability & I/O',
    question: 'How should `Path`-based code avoid exposing a half-written configuration file?',
    answer: 'Write a sibling temporary file, then `os.replace`',
    acceptedAnswers: ['atomic replace', 'temp file rename'],
    explanation: 'Complete and flush the new file before atomically replacing the destination on the same filesystem.',
    detail: 'Atomic replacement protects readers from partial content, but crash durability may also require syncing the file and containing directory. Cross-filesystem moves do not share the same atomic guarantee.',
  },
  {
    id: 'python-json-number-limits',
    topic: 'Reliability & I/O',
    question: 'Why can a `JSON` numeric round trip lose domain meaning?',
    answer: 'JSON has no Python-specific numeric types',
    acceptedAnswers: ['Decimal datetime not preserved', 'type information loss'],
    explanation: 'JSON represents a small set of interoperable values and cannot preserve tuples, decimals, datetimes, or custom classes without a schema.',
    detail: 'Serialization needs an explicit wire contract, not an assumption that arbitrary Python objects survive. Define representations, ranges, time zones, and decoder validation at the boundary.',
  },
  {
    id: 'python-datetime-aware',
    topic: 'Reliability & I/O',
    question: 'Why should persisted `datetime` event timestamps normally be timezone-aware?',
    answer: 'They identify an unambiguous instant',
    acceptedAnswers: ['include timezone', 'UTC aware'],
    explanation: 'A naive datetime has no offset or zone context, so the same wall time can refer to different instants.',
    detail: 'Store an aware UTC instant for ordering and transport, and retain a named local zone separately when future civil-time rules or presentation meaning matter.',
  },
  {
    id: 'python-decimal-money',
    topic: 'Reliability & I/O',
    code: `price = Decimal("0.10")`,
    question: 'Why construct a `Decimal` from text rather than `0.10`?',
    answer: 'Avoid importing binary float approximation',
    acceptedAnswers: ['exact decimal input', 'string constructor'],
    explanation: 'A string preserves the intended decimal digits; a float already contains a binary approximation.',
    detail: 'Decimal arithmetic follows an explicit context for precision and rounding, which fits financial or exact decimal rules. It is not a universal performance replacement for binary floating point.',
  },
  {
    id: 'python-secrets-token',
    topic: 'Reliability & I/O',
    question: 'Which module should generate an authentication token: `random` or `secrets`?',
    answer: '`secrets`',
    explanation: '`secrets` uses operating-system cryptographic randomness; `random` is deterministic and designed for simulation.',
    detail: 'Security needs unpredictability, not merely a large-looking output space. Use purpose-built token helpers and compare sensitive digests with constant-time functions where timing leakage matters.',
  },
  {
    id: 'python-subprocess-shell-injection',
    topic: 'Reliability & I/O',
    question: 'Why prefer `subprocess.run([program, arg], shell=False)` for untrusted arguments?',
    answer: 'Avoid shell parsing and injection',
    acceptedAnswers: ['argument vector', 'no shell injection'],
    explanation: 'An argument list passes boundaries directly to the program instead of interpreting shell metacharacters.',
    detail: 'Quoting rules are platform-specific and easy to get wrong. Invoke the executable directly, validate allowable operations, set timeouts, and inspect the return code and captured diagnostics.',
  },
] satisfies TriviaCard[];
