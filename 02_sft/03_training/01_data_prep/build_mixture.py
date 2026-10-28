"""Build train/val from the S3 pool, used exactly as provided.

The pool is several sources concatenated in a known order (config: data.pool.blocks);
each row is labelled with its block's source, only for the per-source val split and reports.

Output (data/processed/):
    train.jsonl / val.jsonl  rows: {"source": str, "messages": [...], "text": str}
    stats.json               per-source counts and skipped rows

`text` is the exact string the trainer sees; `messages` is kept for evaluation.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import step_root  # noqa: E402,F401  (shared sft_common + this step as config root)
from sft_common.chat_format import ChatFormat
from sft_common.config import load_config, resolve
from sft_common.pretuned import load_tokenizer
from sft_common.sources import pool_row


def read_jsonl(path: Path):
    """Yield one item per non-blank line: the parsed row, or None if it is not valid JSON.

    Bad lines yield None (not skipped) so row positions, and therefore block labels, stay exact.
    """
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    yield None


def block_ends(blocks: list[dict]) -> list[tuple[str, int | None]]:
    """[(source, exclusive end row index)]; end None = to the end of the file (last block only)."""
    ends, total = [], 0
    for i, b in enumerate(blocks):
        if b["rows"] is None:
            if i != len(blocks) - 1:
                raise ValueError("only the last block may have rows: null")
            ends.append((b["source"], None))
        else:
            total += b["rows"]
            ends.append((b["source"], total))
    return ends


def collect(rows, blocks: list[dict], fmt: ChatFormat, stats: dict) -> dict[str, list[dict]]:
    """Label each row by position, adapt and render it. Returns examples grouped by source."""
    ends = block_ends(blocks)
    out = {src: [] for src, _ in ends}
    skipped = {src: 0 for src, _ in ends}
    for i, row in enumerate(rows):
        src = next((s for s, end in ends if end is None or i < end), None)
        if src is None:
            raise ValueError(f"pool has more rows than data.pool.blocks covers (row {i})")
        msgs = pool_row(row) if isinstance(row, dict) else None
        if msgs is None:
            skipped[src] += 1
            continue
        out[src].append({"source": src, "messages": msgs, "text": fmt.render(msgs)})
    for src in out:
        stats[src] = {"kept": len(out[src]), "skipped": skipped[src]}
        print(f"[{src}] kept {len(out[src])}, skipped {skipped[src]}")
    return out


def split(examples: list[dict], val_fraction: float, rng: random.Random):
    rng.shuffle(examples)
    n_val = max(1, int(len(examples) * val_fraction)) if examples else 0
    return examples[n_val:], examples[:n_val]


def write_jsonl(path: Path, rows: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    full_cfg = load_config()
    cfg = full_cfg["data"]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool-file", default=str(resolve(cfg["pool"]["local_path"])))
    ap.add_argument("--out-dir", default=str(resolve(cfg["out_dir"])))
    a = ap.parse_args()

    pool_path = Path(a.pool_file)
    if not pool_path.exists():
        sys.exit(f"Missing {pool_path}. Run download_s3.py first.")

    rng = random.Random(cfg["seed"])
    stats: dict = {}
    # Text is rendered with the pre-tuned model's own chat template (tokenizer files only).
    fmt = ChatFormat(load_tokenizer(full_cfg))
    by_source = collect(read_jsonl(pool_path), cfg["pool"]["blocks"], fmt, stats)

    train, val = [], []
    for examples in by_source.values():  # split per source so every source is in val
        t, v = split(examples, cfg["val_fraction"], rng)
        train += t
        val += v
    rng.shuffle(train)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val.jsonl", val)
    stats.update(train_total=len(train), val_total=len(val))
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"train={len(train)} val={len(val)} -> {out}")


if __name__ == "__main__":
    main()
