"""CPU preflight for a DPO run: everything the GPU job does with the data, without a GPU and without the 18 GB model.

    python 02_training/preflight.py            # downloads only the tokenizer files of dpo.base, and the pairs file + manifest

Checks, in the order the trainer does them: S3 assets exist; the pairs file matches its manifest; the pairs are grouped (pairs/shape.py);
every pair is encoded with the model's own chat template, bad / too-long pairs dropped whole, train/val split; the micro-batch plan
(every pair in exactly one micro-batch); optimizer steps and a rough run time. Prints the same report the trainer prints at startup.
"""
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import step_root  # noqa: E402,F401  (shared sft_common + this step as config root)

from transformers import AutoTokenizer  # noqa: E402

from pairs.shape import shape  # noqa: E402
from pairs.source import load_pairs  # noqa: E402
from sft_common.batching import batch_stats, make_sampler, optimizer_steps, verify_coverage  # noqa: E402
from sft_common.config import load_config, resolve  # noqa: E402
from sft_common.dpo_data import prepare  # noqa: E402
from sft_common.s3_fetch import fetch_file  # noqa: E402
from sft_common.s3_models import parse_uri  # noqa: E402

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "config.json", "generation_config.json")
SFT_TOKENS_PER_S = 6000          # as in dpo_train.py: SFT's measured speed on one H100; DPO ~4/3 of the work per token


def main() -> int:
    cfg = load_config()
    d = cfg["dpo"]
    base = d["base"]["s3_uri"].rstrip("/") + "/"
    tok_dir = resolve("models/preflight_tokenizer")
    for name in TOKENIZER_FILES:
        fetch_file(base + name, tok_dir / name)
    bucket, prefix = parse_uri(base)
    print(f"[preflight] base model {base} (tokenizer files only: {', '.join(TOKENIZER_FILES)})")

    d["data"]["local_path"] = str(resolve(d["data"]["local_path"]))
    rows, source_report = load_pairs(d["data"])
    rows, shape_report = shape(rows, d["data"].get("shape") or {})
    prepared = prepare(rows, AutoTokenizer.from_pretrained(tok_dir), d["data"] | {"max_pair_tokens": d["max_pair_tokens"]}, cfg["data"]["seed"])
    print("\n".join(source_report + shape_report + prepared.report))
    if not prepared.train:
        print("[preflight] FAILED: no usable training pairs")
        return 1

    train = prepared.train
    lengths = [len(p["prompt_ids"]) + max(len(p["chosen_ids"]), len(p["rejected_ids"])) for p in train]
    prompt_share = sum(len(p["prompt_ids"]) for p in train) / sum(lengths)
    sampler = make_sampler("padded", lengths, max_tokens=d["tokens_per_micro_batch"] // 2, max_rows=d["max_rows_per_micro_batch"],
                           seed=cfg["data"]["seed"], shuffle=True)
    batches = sampler.batches()
    verify_coverage(len(lengths), batches)
    s = batch_stats(lengths, batches, "padded", d["tokens_per_micro_batch"] // 2)
    steps = optimizer_steps(s["micro_batches"], d["grad_accum_steps"], d["epochs"])
    processed = sum(len(b) * max(lengths[i] for i in b) for b in batches) * 2
    hours = processed * d["epochs"] * 4 / 3 / SFT_TOKENS_PER_S / 3600
    answer_tokens = [len(p["chosen_ids"]) + len(p["rejected_ids"]) for p in train]
    print(f"[preflight] {len(train)} train pairs, {sum(map(len, prepared.val.values()))} val pairs in {len(prepared.val)} groups")
    print(f"    tokens per pair side: median {st.median(lengths):.0f}, max {max(lengths)}; the shared prompt is {prompt_share:.0%} of them "
          f"(answers: median {st.median(answer_tokens):.0f} tokens for both sides together)")
    print(f"    every pair is in exactly one micro-batch: {s['micro_batches']} micro-batches, fill {s['avg_fill_pct']}%, padding {s['padding_pct']}%, "
          f"{d['grad_accum_steps']} micro-batches per step -> {steps} optimizer steps for {d['epochs']} epoch(s)")
    print(f"    rough ETA {hours:.1f} h on one H100 (from SFT's measured speed +33% for the reference pass; NOT measured for DPO)")
    print("[preflight] OK: the data path is ready. Next: python 06_daytona/daytona_run.py dpo --max-steps 10")
    return 0


if __name__ == "__main__":
    sys.exit(main())
