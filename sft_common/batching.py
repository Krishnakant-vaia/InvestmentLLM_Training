"""How training rows are grouped into micro-batches. Two modes (config: training.batching_mode):

padded  - rows in a micro-batch are padded to the longest one, so the GPU processes
          (rows x longest row) tokens. Rows of similar length are grouped so padding stays small.
          A long row goes alone (anything next to it would be padded to its length).
          Safe with every model.

packed  - rows are laid end to end with no padding (each row keeps its own positions and
          attention), so the GPU processes (sum of row lengths) tokens. The longest remaining
          row starts a micro-batch, then the space left is filled "best fit": each time with
          the LARGEST remaining row that still fits, until nothing fits (e.g. 8.5k + 2.7k +
          0.6k = ~12k). Small rows are kept for the last small gaps, so micro-batches end up
          almost exactly full and nothing trains alone.
          Only correct if the model keeps rows apart; sft_train.py checks that before training.

In both modes: rows are never cut or merged (each row is one whole example in exactly one
micro-batch), every row is used once per epoch, and every micro-batch stays within
`max_tokens` (its cost, as defined by the mode) and `max_rows`.
"""
import math
import random
from collections import defaultdict

MODES = ("padded", "packed")


def cost(lengths: list[int], batch: list[int], mode: str) -> int:
    """Tokens the GPU processes for one micro-batch in the given mode."""
    if mode == "packed":
        return sum(lengths[i] for i in batch)
    return len(batch) * max(lengths[i] for i in batch)


class _Sampler:
    """Batch sampler for DataLoader(batch_sampler=...): yields lists of row indices.

    Deterministic for a given (lengths, seed, epoch), so `--resume` replays the same order
    and the Trainer can skip the micro-batches it has already trained on.
    """

    mode = ""

    def __init__(self, lengths: list[int], max_tokens: int, max_rows: int, seed: int, shuffle: bool = True):
        self.lengths = list(lengths)          # token count of each row, by dataset index
        self.max_tokens = max_tokens          # token budget per micro-batch (see `cost`)
        self.max_rows = max_rows              # hard cap on rows per micro-batch
        self.seed, self.shuffle = seed, shuffle
        self.epoch = 0                        # changes the shuffle each epoch (set by the Trainer)
        self._cache: tuple[int, list[list[int]]] | None = None  # (epoch, batches) built once per epoch

    def set_epoch(self, epoch: int) -> None:
        """Called by the Trainer/accelerate at the start of each epoch."""
        self.epoch = epoch

    def batches(self) -> list[list[int]]:
        """All micro-batches for the current epoch, as lists of row indices."""
        if not (self._cache and self._cache[0] == self.epoch):
            rng = random.Random(self.seed + self.epoch)
            order = list(range(len(self.lengths)))
            if self.shuffle:
                rng.shuffle(order)            # random order first, so ties are broken randomly
            out = self._group(order)
            if self.shuffle:
                rng.shuffle(out)              # mix short/long and domain/tool micro-batches across the run
            self._cache = (self.epoch, out)
        return self._cache[1]

    def _group(self, order: list[int]) -> list[list[int]]:
        raise NotImplementedError

    def __iter__(self):
        yield from self.batches()

    def __len__(self) -> int:
        # Number of micro-batches; the Trainer uses it to compute the number of optimizer steps.
        return len(self.batches())


class PaddedSampler(_Sampler):
    """padded mode: group rows of similar length, keep (rows x longest) <= max_tokens."""

    mode = "padded"

    def __init__(self, *args, chunk_size: int = 1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunk_size = chunk_size          # rows sorted together; smaller = more random, more padding

    def _group(self, order: list[int]) -> list[list[int]]:
        out: list[list[int]] = []
        for start in range(0, len(order), self.chunk_size):
            # Sort each chunk by length so neighbours have similar lengths (little padding).
            chunk = sorted(order[start:start + self.chunk_size], key=self.lengths.__getitem__)
            current: list[int] = []
            for i in chunk:                   # ascending, so row i is the longest in `current` + i
                fits = (len(current) + 1) * self.lengths[i] <= self.max_tokens and len(current) < self.max_rows
                if current and not fits:
                    out.append(current)       # close this micro-batch; row i starts the next one
                    current = []
                current.append(i)
            if current:
                out.append(current)
        return out


class _LengthIndex:
    """The rows not yet placed, grouped by length, answering one question fast:
    "which remaining row is the longest one with at most `gap` tokens?"

    A Fenwick (binary indexed) tree over lengths counts rows per length, so both the lookup
    and the removal take O(log L) time: ~0.2 s for 208k rows instead of hours for a scan.
    """

    def __init__(self, lengths: list[int], order: list[int]):
        self.size = max(lengths, default=0) + 1
        self.tree = [0] * (self.size + 1)
        self.buckets: dict[int, list[int]] = defaultdict(list)   # length -> row indices (random order)
        for i in order:
            self.buckets[lengths[i]].append(i)
            self._add(lengths[i], 1)

    def _add(self, length: int, delta: int) -> None:
        k = length + 1                                            # tree is 1-based; length 0 is allowed
        while k <= self.size:
            self.tree[k] += delta
            k += k & -k

    def _count_upto(self, length: int) -> int:
        """Number of remaining rows with length <= `length`."""
        k, total = min(length, self.size - 1) + 1, 0
        while k > 0:
            total += self.tree[k]
            k -= k & -k
        return total

    def _kth(self, k: int) -> int:
        """Length of the k-th shortest remaining row (1-based)."""
        pos, step = 0, 1 << self.size.bit_length()
        while step:
            if pos + step <= self.size and self.tree[pos + step] < k:
                pos += step
                k -= self.tree[pos]
            step >>= 1
        return pos                                                # pos+1 in the tree = length pos

    def take_largest_upto(self, gap: int) -> int | None:
        """Remove and return the longest remaining row with length <= gap (None if there is none)."""
        if gap < 0:
            return None
        n = self._count_upto(gap)
        if n == 0:
            return None
        length = self._kth(n)                                     # the n-th shortest = longest that fits
        self._add(length, -1)
        return self.buckets[length].pop()


class PackedSampler(_Sampler):
    """packed mode: longest remaining row first, then fill the gap best-fit (largest row that fits)."""

    mode = "packed"

    def _group(self, order: list[int]) -> list[list[int]]:
        # Rows of equal length come out in the random order of `order`, so ties are random.
        index = _LengthIndex(self.lengths, order)
        out: list[list[int]] = []
        while True:
            seed = index.take_largest_upto(index.size)            # the longest remaining row starts it
            if seed is None:
                return out                                        # every row placed
            current, gap = [seed], self.max_tokens - self.lengths[seed]
            # Best fit: keep adding the largest remaining row that fits the gap left.
            while len(current) < self.max_rows:
                i = index.take_largest_upto(gap)
                if i is None:
                    break                                         # no remaining row fits: close it
                current.append(i)
                gap -= self.lengths[i]
            out.append(current)


def make_sampler(mode: str, lengths: list[int], max_tokens: int, max_rows: int, seed: int,
                 shuffle: bool = True) -> _Sampler:
    """The sampler for a batching mode ("padded" or "packed")."""
    if mode not in MODES:
        raise ValueError(f"training.batching_mode must be one of {MODES}, got {mode!r}")
    cls = PackedSampler if mode == "packed" else PaddedSampler
    return cls(lengths, max_tokens=max_tokens, max_rows=max_rows, seed=seed, shuffle=shuffle)


def verify_coverage(n_rows: int, batches: list[list[int]]) -> None:
    """Raise ValueError unless every row 0..n_rows-1 is in exactly one micro-batch
    (proof that grouping neither lost nor duplicated a QA pair)."""
    seen = [0] * n_rows
    for b in batches:
        for i in b:
            if not 0 <= i < n_rows:
                raise ValueError(f"micro-batch refers to row {i}, but there are only {n_rows} rows")
            seen[i] += 1
    missing, repeated = seen.count(0), sum(1 for c in seen if c > 1)
    if missing or repeated:
        raise ValueError(f"grouping lost {missing} rows and repeated {repeated} rows")


def batch_plan(batches: list[list[int]], lengths: list[int], row_ids: list[int], sources: list[str]):
    """One record per micro-batch: which QA pairs (line numbers in train.jsonl) train together."""
    for k, b in enumerate(batches):
        yield {"micro_batch": k, "rows": [row_ids[i] for i in b], "sources": [sources[i] for i in b],
               "tokens": [lengths[i] for i in b], "total": sum(lengths[i] for i in b)}


def batch_stats(lengths: list[int], batches: list[list[int]], mode: str, max_tokens: int) -> dict:
    """Summary of a grouping: how full the micro-batches are (vs the `max_tokens` budget) and
    how much of the GPU work is padding."""
    real = sum(lengths[i] for b in batches for i in b)                    # tokens actually in the data
    processed = sum(cost(lengths, b, mode) for b in batches)              # tokens the GPU processes
    return {
        "micro_batches": len(batches),
        "rows": sum(len(b) for b in batches),
        "avg_rows_per_micro_batch": round(sum(len(b) for b in batches) / max(len(batches), 1), 1),
        "max_rows_in_a_micro_batch": max((len(b) for b in batches), default=0),
        "max_tokens_in_a_micro_batch": max((cost(lengths, b, mode) for b in batches), default=0),
        "avg_fill_pct": round(100 * processed / max(len(batches) * max_tokens, 1), 1),  # of the budget
        "single_row_micro_batches": sum(len(b) == 1 for b in batches),
        "padding_pct": round(100 * (processed - real) / max(processed, 1), 1),
    }


def heaviest_batches(lengths: list[int], batches: list[list[int]], mode: str) -> dict[str, list[int]]:
    """The two micro-batches most likely to run out of memory, for --memory-check."""
    return {
        "heaviest micro-batch (most tokens)": max(batches, key=lambda b: cost(lengths, b, mode)),
        "micro-batch with the most rows": max(batches, key=len),
    }


def optimizer_steps(n_micro_batches: int, grad_accum: int, epochs: float) -> int:
    """Optimizer steps for a run: each step accumulates `grad_accum` micro-batches."""
    return math.ceil(n_micro_batches / grad_accum) * math.ceil(epochs)


def length_report(sources: list[str], lengths: list[int], limit: int) -> dict[str, dict]:
    """Per-source row counts: kept, dropped (longer than `limit` tokens), and the longest row."""
    rep: dict[str, dict] = defaultdict(lambda: {"rows": 0, "dropped": 0, "max_tokens": 0})
    for src, n in zip(sources, lengths):
        r = rep[src]
        r["rows"] += 1
        r["max_tokens"] = max(r["max_tokens"], n)
        if n > limit:
            r["dropped"] += 1
    return dict(rep)
