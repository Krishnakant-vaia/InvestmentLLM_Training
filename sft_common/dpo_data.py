"""DPO preference pairs: read, validate, render exactly like SFT, split, and collate.

Input rows (JSONL): {"prompt": text or chat messages, "preferred": text, "not_preferred": text, "source": label (optional)}.
`source_type` is accepted as an alias of `source`.

    read_pairs(path)                          -> raw rows
    prepare(rows, tokenizer, opts, seed)      -> Prepared(train, val_by_source, report, dropped)
    collate_pairs(batch, pad_id)              -> input_ids / attention_mask / labels for the trainer

What `prepare` guarantees (each is a bug of the old notebook, see README_DPO.md):
  * the prompt is rendered with the model's own chat template, the same text SFT trained on, and both answers
    end on <|im_end|>; an answer with a <think> block is rendered the way the template renders reasoning;
  * a pair is dropped whole, never cut, if it is invalid, too long, or its answers differ too much in length;
  * the split is by PROMPT: no question appears in both train and val.
"""
import json
import random
import statistics as st
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import torch

from .chat_format import ASSISTANT_PART, ChatFormat, split_think

IGNORE = -100


@dataclass
class Prepared:
    train: list[dict]                                   # encoded pairs (see `encode_pair`)
    val: dict[str, list[dict]]                          # source -> encoded pairs
    report: list[str]                                   # human-readable summary, printed and saved by the trainer
    dropped: Counter = field(default_factory=Counter)   # reason -> number of pairs


# ---- reading --------------------------------------------------------------------------------

def read_pairs(path) -> list[dict]:
    """Rows of a pairs JSONL, with the source label normalised and the line number kept (`line`)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, text in enumerate(f):
            if not text.strip():
                continue
            r = json.loads(text)
            r["source"] = str(r.get("source") or r.get("source_type") or "unknown")
            r["line"] = n
            rows.append(r)
    return rows


# ---- one pair -------------------------------------------------------------------------------

def _messages(prompt) -> list[dict] | None:
    """The chat messages of a prompt: text becomes one user message; a list must end on a user turn."""
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}] if prompt.strip() else None
    if isinstance(prompt, list) and prompt and all(isinstance(m, dict) and "role" in m for m in prompt):
        return prompt if prompt[-1]["role"] == "user" and any(m.get("content") for m in prompt) else None
    return None


def _assistant(text: str) -> dict:
    """An assistant turn from answer text; a leading <think>...</think> becomes the turn's reasoning."""
    reasoning, answer = split_think(text)
    return {"role": "assistant", "content": answer, "reasoning": reasoning}


def has_think(text: str) -> bool:
    return split_think(text)[0] != ""


def encode_pair(fmt: ChatFormat, row: dict) -> dict | str:
    """Token ids of one pair, or the reason it cannot be used (a string).

    Returns {"prompt_ids", "chosen_ids", "rejected_ids", "source", "line"}: the prompt (up to and including
    the assistant marker, shared by both answers) and each answer's tokens ending on <|im_end|>.
    """
    msgs = _messages(row.get("prompt"))
    chosen, rejected = row.get("preferred"), row.get("not_preferred")          # internal names below: the preferred / not-preferred answer
    if msgs is None or not isinstance(chosen, str) or not isinstance(rejected, str) or not chosen.strip() or not rejected.strip():
        return "missing or empty prompt/preferred/not_preferred"
    if chosen.strip() == rejected.strip():
        return "preferred identical to not_preferred"
    texts = [fmt.render(msgs + [_assistant(a)]) for a in (chosen, rejected)]
    cuts = [t.rfind(ASSISTANT_PART) + len(ASSISTANT_PART) for t in texts]
    if min(cuts) < len(ASSISTANT_PART) or texts[0][:cuts[0]] != texts[1][:cuts[1]]:
        return "prompt renders differently for the two answers"
    tok = fmt.tok
    prompt_ids = tok(texts[0][:cuts[0]], add_special_tokens=False)["input_ids"]
    # The template ends a turn with "<|im_end|>\n"; the trailing newline is not part of what the model writes.
    ends = [t[c:].removesuffix("\n") for t, c in zip(texts, cuts)]
    chosen_ids, rejected_ids = (tok(e, add_special_tokens=False)["input_ids"] for e in ends)
    return {"prompt_ids": prompt_ids, "chosen_ids": chosen_ids, "rejected_ids": rejected_ids,
            "source": row["source"], "line": row["line"], "group": str(row.get("group") or "")}


def side_tokens(p: dict) -> int:
    """Tokens of the longer side (prompt + answer) of an encoded pair: what padding makes both sides cost."""
    return len(p["prompt_ids"]) + max(len(p["chosen_ids"]), len(p["rejected_ids"]))


# ---- a whole file ---------------------------------------------------------------------------

def _split_by_prompt(pairs: list[dict], fraction: float, cap: int, seed: int) -> tuple[list[dict], list[dict]]:
    """(train, val) of one source, split by prompt so a question is never on both sides."""
    groups = defaultdict(list)
    for p in pairs:                                        # a row may name its own group (a document, a question template)
        groups[p["group"] or tuple(p["prompt_ids"])].append(p)
    keys = sorted(groups, key=str)                           # sorted first: the shuffle is reproducible
    random.Random(seed).shuffle(keys)
    target = round(len(pairs) * fraction) if len(keys) >= 10 else 0     # val size in PAIRS, filled with whole groups
    val, train = [], []
    want = min(target, cap)
    for k in keys:                                         # a group goes to val whole, and only if it does not swamp it
        to_val = len(val) < want and len(val) + len(groups[k]) <= 1.5 * want
        (val if to_val else train).extend(groups[k])
    return train, val


def prepare(rows: list[dict], tokenizer, opts: dict, seed: int) -> Prepared:
    """Validate and encode every pair, split train/val, and describe the result.

    opts: max_pair_tokens, max_length_ratio, drop_think_mismatch, val_fraction, val_cap_per_source.
    """
    fmt, dropped = ChatFormat(tokenizer), Counter()
    kept, per_source = [], defaultdict(Counter)
    for r in rows:
        per_source[r["source"]]["rows"] += 1
        out = encode_pair(fmt, r)
        if isinstance(out, str):
            reason = out
        elif opts["drop_think_mismatch"] and has_think(r["preferred"]) != has_think(r["not_preferred"]):
            reason = "only one answer has a <think> block"
        elif max(len(out["chosen_ids"]), len(out["rejected_ids"])) > opts["max_length_ratio"] * max(1, min(len(out["chosen_ids"]), len(out["rejected_ids"]))):
            reason = f"answers differ more than {opts['max_length_ratio']}x in length"
        elif side_tokens(out) > opts["max_pair_tokens"]:
            reason = f"longer than {opts['max_pair_tokens']} tokens"
        else:
            kept.append(out)
            continue
        dropped[reason] += 1
        per_source[r["source"]]["dropped"] += 1

    train, val = [], {}
    for src in sorted({p["source"] for p in kept}):
        t, v = _split_by_prompt([p for p in kept if p["source"] == src], opts["val_fraction"], opts["val_cap_per_source"], seed)
        train += t
        if v:
            val[src] = v
    random.Random(seed).shuffle(train)
    return Prepared(train, val, _report(rows, kept, train, val, per_source, dropped, opts), dropped)


def _report(rows, kept, train, val, per_source, dropped, opts) -> list[str]:
    lines = [f"[dpo data] {len(rows)} pairs read, {len(kept)} usable, {len(rows) - len(kept)} dropped whole (never cut)"]
    lines += [f"    dropped: {n:>6}  {why}" for why, n in dropped.most_common()]
    for src, c in sorted(per_source.items()):
        lines.append(f"    source {src:12} read={c['rows']:>6} dropped={c['dropped']:>5} "
                     f"train={sum(p['source'] == src for p in train):>6} val={len(val.get(src, []))}")
    if kept:
        c_len, r_len = [len(p["chosen_ids"]) for p in kept], [len(p["rejected_ids"]) for p in kept]
        longer = sum(c > r for c, r in zip(c_len, r_len)) / len(kept)
        lines.append(f"    answer tokens: preferred median {st.median(c_len):.0f} / not-preferred median {st.median(r_len):.0f}; "
                     f"preferred is the longer one in {longer:.0%} of pairs; longest side {max(map(side_tokens, kept))} of {opts['max_pair_tokens']}")
        if longer > 0.7 or longer < 0.3:
            lines.append(f"    WARNING: preferred is longer in {longer:.0%} of pairs. DPO will partly learn "
                         f"{'verbosity' if longer > 0.5 else 'brevity'} instead of quality: rebalance the pairs.")
    n_prompts = lambda ps: len({tuple(p["prompt_ids"]) for p in ps})  # noqa: E731
    key = lambda p: p["group"] or tuple(p["prompt_ids"])  # noqa: E731
    both = {key(p) for p in train} & {key(p) for v in val.values() for p in v}
    lines.append(f"    split by prompt: train {len(train)} pairs / {n_prompts(train)} prompts, "
                 f"val {sum(map(len, val.values()))} pairs; groups (prompt, document or template) in both: {len(both)}")
    return lines


# ---- batching -------------------------------------------------------------------------------

def collate_pairs(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    """B encoded pairs -> 2B right-padded sequences: rows 0..B-1 are the chosen sides, B..2B-1 the rejected.

    labels = the answer tokens only (-100 on the prompt and on padding), so only what the model writes is scored.
    """
    seqs = [(p["prompt_ids"], p[f"{side}_ids"]) for side in ("chosen", "rejected") for p in batch]
    width = max(len(a) + len(b) for a, b in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    labels = torch.full((len(seqs), width), IGNORE, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, (prompt, answer) in enumerate(seqs):
        n = len(prompt) + len(answer)
        ids[i, :n] = torch.tensor(prompt + answer)
        labels[i, len(prompt):n] = torch.tensor(answer)
        mask[i, :n] = 1
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}
