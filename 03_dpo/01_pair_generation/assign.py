"""Step 2: chunks -> assignments. Code, not the LLM, decides which question types and defects each chunk gets.

    python assign.py --chunks work/corpus/chunks.jsonl --out work/run/assignments.jsonl [--limit-chunks 200]

One line per LLM call: {call_id, chunk_id, types, defects}. A `Balancer` keeps the long-run share of every question type and
every defect at its target (largest shortfall first), so the answer-position and type biases cannot drift. A type is never
repeated for the same chunk across passes; numerical types are only assigned to chunks that contain figures; Drafting and the
four calculation types get a call each, the others share a call (at most MAX_PAIRS_PER_CALL).
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import config
from score_pairs import COMPATIBLE, EXPECTED_GATES, MAX_PAIRS_PER_CALL, ONE_PAIR_PER_CALL

COMPATIBLE_DEFECTS = set(EXPECTED_GATES)


class Balancer:
    """Picks keys so that their long-run shares match the weights: the key furthest below its target share goes next."""

    def __init__(self, weights: dict[str, float], seed: int):
        self.w = {k: v for k, v in weights.items() if v > 0}
        self.total_w = sum(self.w.values())
        self.count, self.n, self.rng = Counter(), 0, random.Random(seed)

    def next(self, allowed=None) -> str:
        keys = [k for k in self.w if allowed is None or k in allowed]
        if not keys:
            raise ValueError("no allowed key")
        best = max(keys, key=lambda k: (self.w[k] / self.total_w * (self.n + 1) - self.count[k], self.rng.random()))
        self.count[best] += 1
        self.n += 1
        return best


def sample_chunks(chunks: list[dict], limit: int | None, seed: int = config.SEED) -> list[dict]:
    """All chunks, or a seeded sample of `limit` that takes turns across folders (so every folder is in a smoke test) and
    reserves about 1 in 8 places for chunks with figures (so the numerical question types are exercised too)."""
    ordered = sorted(chunks, key=lambda c: c["chunk_id"])
    if not limit or limit >= len(ordered):
        return ordered
    rng = random.Random(seed)
    rng.shuffle(ordered)

    def take(pool: list[dict], n: int) -> list[dict]:
        by_folder: dict[str, list[dict]] = defaultdict(list)
        for c in pool:
            by_folder[c["folder"]].append(c)
        folders, picked = sorted(by_folder), []
        while len(picked) < n and any(by_folder.values()):
            for f in folders:
                if by_folder[f] and len(picked) < n:
                    picked.append(by_folder[f].pop(0))
        return picked

    figures = take([c for c in ordered if c["n_quantities"] >= config.MIN_QUANTITIES_FOR_NUMERIC], max(2, limit // 8))
    chosen = {c["chunk_id"] for c in figures}
    return figures + take([c for c in ordered if c["chunk_id"] not in chosen], limit - len(figures))


def assign(chunks: list[dict], seed: int = config.SEED, passes: int = config.PASSES, per_pass: int = config.TYPES_PER_PASS,
           type_mix: dict[str, float] = config.TYPE_MIX, defect_targets: dict[str, float] | None = None,
           styles: list[str] | None = None, adjustment_words=None, cross_marker: str | None = None) -> list[dict]:
    """adjustment_words: the pattern a chunk must match for headline_number / wrong_adjustment (default config.ADJUSTMENT_WORDS).
    cross_marker: when given, cross_section_conflict is assigned only to a chunk holding at least two of these markers (two labelled sections)."""
    defect_targets = config.DEFECT_TARGETS if defect_targets is None else defect_targets    # read at call time, not import time
    adjustment_words = config.ADJUSTMENT_WORDS if adjustment_words is None else adjustment_words
    styles = config.STYLES if styles is None else styles
    unknown = set(type_mix) - set(COMPATIBLE)
    if unknown:
        raise ValueError(f"unknown question types in TYPE_MIX: {sorted(unknown)}")
    cap = 1 / (passes * per_pass)
    for t, w in type_mix.items():
        if w / sum(type_mix.values()) > cap + 1e-9:
            raise ValueError(f"{t} is {w / sum(type_mix.values()):.1%} of TYPE_MIX but a type is used once per chunk, "
                             f"so it cannot exceed {cap:.1%} (passes x types per pass = {passes * per_pass})")
    types = Balancer(type_mix, seed)
    over = {d: w for d, w in defect_targets.items() if w > config.MAX_DEFECT_SHARE}
    if over:
        raise ValueError(f"DEFECT_TARGETS above MAX_DEFECT_SHARE ({config.MAX_DEFECT_SHARE:.0%}): {over}")
    assignable = {d for t in type_mix for d in COMPATIBLE[t]}
    if set(defect_targets) - set(COMPATIBLE_DEFECTS) or set(assignable) - set(defect_targets):
        raise ValueError(f"DEFECT_TARGETS must cover exactly the defects the chosen question types allow: {sorted(assignable)}")
    defects = Balancer(defect_targets, seed + 1)
    used, calls, rng = defaultdict(set), [], random.Random(seed)
    style_rng = random.Random(seed + 2)       # separate stream: drawing a style never changes which types and defects are assigned
    for p in range(passes):
        order = sorted(chunks, key=lambda c: c["chunk_id"])
        rng.shuffle(order)
        for c in order:
            allowed = {t for t in type_mix if t not in used[c["chunk_id"]]
                       and (t not in config.NUMERIC_TYPES_NEED_FIGURES or c["n_quantities"] >= config.MIN_QUANTITIES_FOR_NUMERIC)}
            picks = []
            while len(picks) < per_pass and allowed - set(picks):
                picks.append(types.next(allowed - set(picks)))
            used[c["chunk_id"]] |= set(picks)
            text = c.get("text")              # a file with no mention of adjustments cannot take headline_number / wrong_adjustment
            blocked = set() if text is None or adjustment_words.search(text) else set(config.ADJUSTMENT_DEFECTS)
            if cross_marker and text is not None and text.count(cross_marker) < 2:
                blocked.add("cross_section_conflict")      # a contradiction between sections needs two sections in the document
            pairs = [(t, defects.next(set(COMPATIBLE[t]) - blocked)) for t in picks]
            single = [x for x in pairs if x[0] in ONE_PAIR_PER_CALL]
            multi = [x for x in pairs if x[0] not in ONE_PAIR_PER_CALL]
            groups = [[x] for x in single] + [multi[i:i + MAX_PAIRS_PER_CALL] for i in range(0, len(multi), MAX_PAIRS_PER_CALL)]
            for j, g in enumerate(groups):
                calls.append({"call_id": f"{c['chunk_id']}-p{p}-{j}", "chunk_id": c["chunk_id"],
                              "types": [t for t, _ in g], "defects": [d for _, d in g], "style": style_rng.choice(styles)})
    return calls


def assign_grid(chunks: list[dict], seed: int = config.SEED, types: list[str] | None = None, styles: list[str] | None = None,
                neutral_style: str | None = None) -> list[dict]:
    """Tear-sheet mode: every chunk gets each compatible (type, defect) pair once, one pair per call (a call cannot hold a type twice).
    Drafting calls draw a layout from `styles`; the other types get `neutral_style`."""
    types = config.TEARSHEET_TYPES if types is None else types
    styles = config.TEARSHEET_STYLES if styles is None else styles
    neutral_style = config.TEARSHEET_NEUTRAL_STYLE if neutral_style is None else neutral_style
    unknown = set(types) - set(COMPATIBLE)
    if unknown:
        raise ValueError(f"unknown question types: {sorted(unknown)}")
    rng, calls = random.Random(seed + 2), []
    for c in sorted(chunks, key=lambda c: c["chunk_id"]):
        k = 0
        for t in types:
            for d in COMPATIBLE[t]:
                calls.append({"call_id": f"{c['chunk_id']}-g{k}", "chunk_id": c["chunk_id"], "types": [t], "defects": [d],
                              "style": rng.choice(styles) if t in ("Drafting", "CIM Drafting") else neutral_style})
                k += 1
    return calls


def assign_balanced(chunks: list[dict], per_chunk: int = 5, seed: int = config.SEED, type_mix: dict[str, float] | None = None,
                    defect_targets: dict[str, float] | None = None, styles: list[str] | None = None, adjustment_words=None,
                    cross_marker: str | None = None, neutral_style: str | None = None) -> list[dict]:
    """Up to `per_chunk` pairs per chunk, one pair per call (a call cannot hold a type twice). A type may repeat inside a chunk with a
    different defect, never the same (type, defect) twice. Types and defects are balanced across the whole run by their target shares.
    A numerical type needs a chunk with figures; headline_number / wrong_adjustment need `adjustment_words` in the text, and
    cross_section_conflict needs `cross_marker` twice (two labelled sections). Drafting types draw a layout from `styles`."""
    type_mix = config.CIM_TYPE_MIX if type_mix is None else type_mix
    defect_targets = config.CIM_DEFECT_TARGETS if defect_targets is None else defect_targets
    adjustment_words = config.CIM_BASIS_WORDS if adjustment_words is None else adjustment_words
    styles = config.STYLES if styles is None else styles
    neutral_style = config.TEARSHEET_NEUTRAL_STYLE if neutral_style is None else neutral_style
    unknown = set(type_mix) - set(COMPATIBLE)
    if unknown:
        raise ValueError(f"unknown question types: {sorted(unknown)}")
    over = {d: w for d, w in defect_targets.items() if w > config.MAX_DEFECT_SHARE}
    if over:
        raise ValueError(f"defect targets above MAX_DEFECT_SHARE ({config.MAX_DEFECT_SHARE:.0%}): {over}")
    types, defects = Balancer(type_mix, seed), Balancer(defect_targets, seed + 1)
    rng, calls = random.Random(seed + 2), []
    for c in sorted(chunks, key=lambda c: c["chunk_id"]):
        text = c.get("text")
        blocked = set() if text is None or adjustment_words.search(text) else set(config.ADJUSTMENT_DEFECTS)
        if cross_marker and text is not None and text.count(cross_marker) < 2:
            blocked.add("cross_section_conflict")
        figures = c["n_quantities"] >= config.MIN_QUANTITIES_FOR_NUMERIC
        taken: set[tuple[str, str]] = set()
        for k in range(per_chunk):
            free = {t: [d for d in COMPATIBLE[t] if d in defect_targets and d not in blocked and (t, d) not in taken]
                    for t in type_mix if figures or t not in config.NUMERIC_TYPES_NEED_FIGURES}
            free = {t: ds for t, ds in free.items() if ds}
            if not free:
                break
            t = types.next(set(free))
            d = defects.next(set(free[t]))
            taken.add((t, d))
            calls.append({"call_id": f"{c['chunk_id']}-b{k}", "chunk_id": c["chunk_id"], "types": [t], "defects": [d],
                          "style": rng.choice(styles) if t in ("Drafting", "CIM Drafting") else neutral_style})
    return calls


def summary(calls: list[dict]) -> str:
    n_pairs = sum(len(c["types"]) for c in calls)
    types = Counter(t for c in calls for t in c["types"])
    defects = Counter(d for c in calls for d in c["defects"])
    lines = [f"[assign] {len(calls)} calls, {n_pairs} pairs requested, {len({c['chunk_id'] for c in calls})} chunks"]
    lines += [f"    type   {t:28} {n:>5} ({n / n_pairs:.1%})" for t, n in types.most_common()]
    lines += [f"    defect {d:28} {n:>5} ({n / n_pairs:.1%})" for d, n in defects.most_common()]
    return "\n".join(lines)


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-chunks", type=int, help="seeded sample of N chunks (smoke / pilot)")
    ap.add_argument("--deck", action="store_true", help="management-presentation mix (config.DECK_ASSIGN)")
    ap.add_argument("--tearsheet", action="store_true", help="tear-sheet grid: every compatible type x defect once per chunk")
    a = ap.parse_args()
    chunks = sample_chunks(read_jsonl(a.chunks), a.limit_chunks)
    calls = assign_grid(chunks) if a.tearsheet else assign(chunks, **(config.DECK_ASSIGN if a.deck else {}))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(c) + "\n" for c in calls)
    print(summary(calls))


if __name__ == "__main__":
    main()
