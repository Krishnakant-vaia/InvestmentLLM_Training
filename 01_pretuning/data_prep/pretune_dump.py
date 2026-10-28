import argparse
import hashlib
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from botocore.config import Config as BotoConfig

# CHUNK_TARGET_TOKENS is deliberately equal to PACK_SEQ_LEN. Chunking to a
# smaller size and then having pack_chunks glue pieces back together bought
# nothing — it destroyed long-range document structure for no benefit and
# left the packer's greedy fill with wasted tail space. Chunking now only
# exists to split sections that are themselves larger than one training
# sequence.
CHUNK_TARGET_TOKENS = 8192

# Overlap is a retrieval idiom (protects a RAG chunk from cutting a fact in
# half). In continued pretraining it is pure loss — it inflates the token
# bill by ~12%, and because pack_chunks concatenates adjacent chunks in
# order, the overlapping span lands twice inside the same training
# sequence, giving the model a memorization signal on chunk boundaries.
CHUNK_OVERLAP_TOKENS = 0

# Only sections genuinely larger than a training sequence get split.
MAX_SECTION_TOKENS = 8192

# MUST match max_seq_len in the trainer config exactly (training/train_pretune.py's
# MAX_SEQ_LENGTH). If the trainer runs at a different length, every sequence
# here is truncated at train time and part of the corpus is silently discarded.
PACK_SEQ_LEN = 8192
PACK_SEPARATOR = "<|endoftext|>"

# ~61,355 real documents in the corpus (counted live 2026-08-31) -> 0.005
# gives ~307 held-out documents, plenty for a stable loss/perplexity curve
# without over-holding-out.
VAL_FRACTION = 0.005
SPLIT_SEED = 42
MODEL_NAME = "Qwen/Qwen3.5-9B-Base"

# Concurrent S3 document fetches. Raised from 64 to 256 after confirming on
# the real pod: file-descriptor headroom (1024 limit, ~68 FDs used at 64
# workers), 128 CPUs (irrelevant anyway for I/O-wait threads), and ~2TB RAM
# all had large margin to spare. Drop back down if this starts throttling or
# hitting "Too many open files". Purely a loading-speed lever --
# chunking/dedup/split/pack are untouched by this.
S3_FETCH_WORKERS = 256
# No default S3 prefix on purpose: a hardcoded, date-stamped default silently
# becomes the destination for every future run that forgets to override it,
# clobbering or mixing with whatever's already there. --s3-prefix is required
# (checked in main()) so every real run makes an explicit, conscious choice.
# Example: s3://ft-vaia/Corpus/Invest_dataset/08272026_PreTuningResults/

# Deliberately narrow: only patterns specific enough that real financial-document
# prose is unlikely to trigger them by accident. An earlier version also matched
# bare ALL-CAPS lines and "N. Capitalized text" — both fire constantly on real
# filings (balance-sheet line items, signature blocks, disclaimers, numbered
# prose), over-segmenting text at false boundaries. Under-detecting (falling
# back to fixed-size chunking) is a safer failure mode than corrupting good
# chunks with spurious splits, so those two patterns were dropped rather than
# tuned.
#
# FIXED (2026-08-31): the ITEM/PART branches were uppercase-only with no
# case-insensitivity, so "Item 1A. Risk Factors" — realistic casing in real
# filings — never matched; detect_sections silently returned [] on nearly
# every real document and everything fell through to fixed-size chunking.
# Using an explicit (?:ITEM|Item)/(?:PART|Part) alternation rather than a
# blanket re.IGNORECASE keeps the pattern narrow: lowercase "item 3 of the
# agreement" inside running prose still won't match.
_HEADING_RE = re.compile(
    r"^(?:#{1,6}\s+\S.*"                            # markdown-style headings
    r"|(?:ITEM|Item)\s+\d+[A-Z]?\.?\s*.*"           # "Item 1A. Risk Factors"
    r"|(?:PART|Part)\s+[IVXLC]+\b.*)$",             # "PART II"
    re.MULTILINE,
)

# S3 error codes that will never succeed on retry. Retrying these just burns
# the full backoff before failing anyway — at real corpus scale (tens of
# thousands of documents) even a small permanent-failure rate wastes hours.
_PERMANENT_S3_ERRORS = {
    "NoSuchKey", "NoSuchBucket", "AccessDenied", "403", "404",
    "InvalidAccessKeyId", "SignatureDoesNotMatch", "InvalidBucketName",
}


def detect_sections(text: str) -> list[str]:
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return []
    sections = []
    # Text before the first heading used to be dropped silently — keep it as
    # its own section, since on some document types that preamble is real
    # content (a cover page's summary text), not just boilerplate.
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(preamble)
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append(text[start:end].strip())
    return sections


def split_fixed_size(text: str, target_tokens: int, overlap_tokens: int, tokenizer) -> list[str]:
    if overlap_tokens >= target_tokens:
        raise ValueError(
            f"overlap_tokens ({overlap_tokens}) must be < target_tokens ({target_tokens}); "
            "otherwise the window never advances."
        )
    token_ids = tokenizer.encode(text)
    if len(token_ids) <= target_tokens:
        return [text]
    pieces = []
    start = 0
    step = target_tokens - overlap_tokens
    while start < len(token_ids):
        end = min(start + target_tokens, len(token_ids))
        pieces.append(tokenizer.decode(token_ids[start:end]))
        if end == len(token_ids):
            break
        start += step
    return pieces


def chunk_document(
    doc_key: str,
    text: str,
    tokenizer,
    target_tokens: int,
    overlap_tokens: int,
    max_section_tokens: int,
) -> list[dict]:
    sections = detect_sections(text) or [text]  # no headings detected -> whole doc is one section
    chunks = []
    idx = 0
    for section in sections:
        section_tokens = len(tokenizer.encode(section))
        if section_tokens <= max_section_tokens:
            # Token count is already exact here, so carry it rather than
            # re-encoding this text in pack_chunks and again in the stats report.
            sized_pieces = [(section, section_tokens)]
        else:
            sized_pieces = [
                (piece, len(tokenizer.encode(piece)))
                for piece in split_fixed_size(section, target_tokens, overlap_tokens, tokenizer)
            ]
        for piece, n_tokens in sized_pieces:
            if not piece.strip():
                continue
            chunks.append({
                "doc_key": doc_key,
                "chunk_id": f"{doc_key}-c{idx}",
                "chunk_text": piece,
                "n_tokens": n_tokens,
            })
            idx += 1
    return chunks


def dedupe_chunks(chunks: list[dict]) -> tuple[list[dict], int]:
    """Drops exact-duplicate chunk text, keeping the first occurrence.

    Financial corpora are full of repeated boilerplate — disclaimers,
    risk-factor language copied year over year, the same company's filings
    across periods. This must run BEFORE split_by_document, or identical
    text ends up on both sides of the train/val boundary and validation
    loss/perplexity becomes an unreliable signal (the model may just be
    recalling memorized boilerplate, not demonstrating real held-out
    performance).

    Exact-match only. Near-duplicate removal (MinHash/LSH) is a separate
    pass and is not implemented here."""
    seen: set[str] = set()
    kept = []
    for chunk in chunks:
        digest = hashlib.sha256(chunk["chunk_text"].encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        kept.append(chunk)
    return kept, len(chunks) - len(kept)


def split_by_document(chunks: list[dict], val_fraction: float, seed: int) -> dict[str, list[dict]]:
    doc_keys = sorted({c["doc_key"] for c in chunks})
    rng = random.Random(seed)
    shuffled_docs = doc_keys[:]
    rng.shuffle(shuffled_docs)

    n_val_docs = round(len(shuffled_docs) * val_fraction)
    val_doc_set = set(shuffled_docs[:n_val_docs])

    splits: dict[str, list[dict]] = {"train": [], "val": []}
    for chunk in chunks:
        target = "val" if chunk["doc_key"] in val_doc_set else "train"
        splits[target].append(chunk)
    return splits


def pack_chunks(chunks: list[dict], tokenizer, max_seq_len: int, separator: str) -> list[dict]:
    sep_tokens = len(tokenizer.encode(separator))
    packed: list[dict] = []
    current_texts: list[str] = []
    current_tokens = 0

    def flush():
        if current_texts:
            packed.append({"text": separator.join(current_texts), "n_tokens": current_tokens})

    for chunk in chunks:
        chunk_tokens = chunk["n_tokens"]  # carried from chunk_document, not recomputed

        if chunk_tokens > max_seq_len:
            flush()
            current_texts.clear()
            current_tokens = 0
            # kept whole; truncated at train time if needed
            packed.append({"text": chunk["chunk_text"], "n_tokens": chunk_tokens})
            continue

        added = chunk_tokens + (sep_tokens if current_texts else 0)
        if current_texts and current_tokens + added > max_seq_len:
            flush()
            current_texts.clear()
            current_tokens = 0

        current_texts.append(chunk["chunk_text"])
        current_tokens += chunk_tokens + (sep_tokens if len(current_texts) > 1 else 0)

    flush()
    return packed


def _is_permanent_s3_error(exc: Exception) -> bool:
    try:
        code = str(exc.response["Error"]["Code"])  # type: ignore[attr-defined]
    except Exception:
        return False
    return code in _PERMANENT_S3_ERRORS


def with_retries(fn, *args, retries: int = 3, backoff_seconds: float = 1.0, **kwargs):
    """Calls fn(*args, **kwargs), retrying on transient exceptions with
    exponential backoff, up to `retries` extra attempts. Re-raises
    immediately on permanent errors (missing key, bad credentials) and
    re-raises the last exception if every attempt fails."""
    import time

    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if _is_permanent_s3_error(exc):
                raise
            last_exc = exc
            if attempt < retries:
                time.sleep(backoff_seconds * (2 ** attempt))
    raise last_exc


def load_resume_log(path: str) -> set[str]:
    """Returns the set of doc_keys already recorded as processed in a prior
    (possibly interrupted) run. Missing file means a fresh run."""
    import os

    if not path or not os.path.isfile(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def append_resume_log(path: str, doc_keys: list[str]) -> None:
    """Records doc_keys as processed. Call this ONLY after their output has
    been durably written — see the note in main()."""
    with open(path, "a", encoding="utf-8") as f:
        for doc_key in doc_keys:
            f.write(doc_key + "\n")


def load_document_text(doc_key: str, s3_client) -> str:
    """Reads the already-extracted plain text for a document. Adjust the
    bucket/key convention below to match wherever your upstream extraction
    step actually stores it — this assumes doc_key is itself an s3:// path
    to the extracted text.

    Takes an injected client: constructing boto3.client("s3") per document
    costs ~100ms in service-model loading — at the real corpus's ~61,000
    documents that's roughly 1.7 hours of pure overhead if done per-call."""
    bucket, key = doc_key.replace("s3://", "").split("/", 1)
    obj = with_retries(s3_client.get_object, retries=3, backoff_seconds=1.0, Bucket=bucket, Key=key)
    return obj["Body"].read().decode("utf-8")


def load_documents_concurrently(doc_keys: list[str], s3_client, max_workers: int) -> tuple[list[tuple[str, str]], list[str]]:
    """Fetches doc_keys concurrently via a thread pool. Returns (docs, failed_doc_keys).

    Uses executor.map rather than as_completed specifically so docs comes
    back in the exact same order as doc_keys, even though the fetches
    themselves run concurrently -- this keeps everything downstream
    (chunking, splitting, packing) byte-for-byte identical to the old
    sequential loop's behavior, just faster."""

    def _load_one(doc_key: str) -> tuple[str, str | None, str | None]:
        try:
            return doc_key, load_document_text(doc_key, s3_client), None
        except Exception as exc:
            return doc_key, None, str(exc)

    docs: list[tuple[str, str]] = []
    failed: list[str] = []
    if not doc_keys:
        return docs, failed

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for i, (doc_key, text, error) in enumerate(pool.map(_load_one, doc_keys)):
            if error:
                print(f"WARNING: failed to load {doc_key}, skipping: {error}")
                failed.append(doc_key)
            else:
                docs.append((doc_key, text))
            if (i + 1) % 1000 == 0:
                print(f"Loaded {len(docs)} / {i + 1} documents scanned")
    return docs, failed


def generate_stats_report(splits: dict, packed_splits: dict, n_duplicates_dropped: int) -> dict:
    """Sums the n_tokens field chunk_document/pack_chunks already computed,
    rather than re-tokenizing text a second (or third) time — no tokenizer
    needed here."""
    return {
        "duplicate_chunks_dropped": n_duplicates_dropped,
        "chunk_counts": {name: len(chunks) for name, chunks in splits.items()},
        "chunk_token_counts": {
            name: sum(c["n_tokens"] for c in chunks) for name, chunks in splits.items()
        },
        "packed_sequence_counts": {name: len(seqs) for name, seqs in packed_splits.items()},
        "packed_token_counts": {
            name: sum(seq["n_tokens"] for seq in seqs) for name, seqs in packed_splits.items()
        },
        "mean_packed_fill": {
            name: (
                round(sum(s["n_tokens"] for s in seqs) / (len(seqs) * PACK_SEQ_LEN), 4)
                if seqs else 0.0
            )
            for name, seqs in packed_splits.items()
        },
    }


def render_markdown(report: dict) -> str:
    lines = ["# Pre-Tuning Dump Stats Report", ""]
    lines += [f"Duplicate chunks dropped: **{report['duplicate_chunks_dropped']}**", ""]
    lines += ["## Chunk Counts", ""]
    for k, v in report["chunk_counts"].items():
        lines.append(f"- **{k}**: {v}")
    lines += ["", "## Chunk Token Counts", ""]
    for k, v in report["chunk_token_counts"].items():
        lines.append(f"- **{k}**: {v}")
    lines += ["", "## Packed Sequences", ""]
    for k, v in report["packed_sequence_counts"].items():
        lines.append(f"- **{k}**: {v}")
    lines += ["", "## Packed Token Counts", ""]
    for k, v in report["packed_token_counts"].items():
        lines.append(f"- **{k}**: {v}")
    lines += ["", f"## Mean Fill (of {PACK_SEQ_LEN}-token sequences)", ""]
    for k, v in report["mean_packed_fill"].items():
        lines.append(f"- **{k}**: {v:.1%}")
    return "\n".join(lines)


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_NAME)


def build_output_files(packed_splits: dict, report_markdown: str) -> dict[str, str]:
    """Builds the output files' content in memory: {filename: content}. Only
    the "text" field is written — pack_chunks' internal "n_tokens" field
    (used by generate_stats_report) is deliberately not leaked into the
    training data."""
    files = {}
    for name, seqs in packed_splits.items():
        files[f"{name}.jsonl"] = "".join(json.dumps({"text": seq["text"]}) + "\n" for seq in seqs)
    files["report.md"] = report_markdown
    return files


def load_local_docs(dir_path: str) -> list[tuple[str, str]]:
    """Reads every .txt file in dir_path as (filename, text) — a stand-in
    for S3 document loading, used for offline dry runs."""
    import os

    docs = []
    for filename in sorted(os.listdir(dir_path)):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(dir_path, filename)
        with open(path, "r", encoding="utf-8") as f:
            docs.append((filename, f.read()))
    return docs


def write_output_files_locally(files: dict[str, str], out_dir: str) -> list[str]:
    """Writes each {filename: content} pair to out_dir — the dry-run
    equivalent of upload_outputs_to_s3, so results can be inspected without
    touching S3."""
    import os

    os.makedirs(out_dir, exist_ok=True)
    written = []
    for filename, content in files.items():
        path = os.path.join(out_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        written.append(path)
    return written


def check_existing_keys(files: dict[str, str], s3_uri_prefix: str, s3_client) -> list[str]:
    """Returns the filenames that already exist at the target S3 prefix —
    used to refuse silently clobbering a previous run's output."""
    if not s3_uri_prefix.endswith("/"):
        s3_uri_prefix += "/"
    bucket, prefix = s3_uri_prefix.replace("s3://", "").split("/", 1)
    existing = []
    for filename in files:
        key = prefix + filename
        try:
            s3_client.head_object(Bucket=bucket, Key=key)
            existing.append(filename)
        except Exception:
            pass  # not found (or any other error) -> treat as not existing
    return existing


def upload_outputs_to_s3(files: dict[str, str], s3_uri_prefix: str, s3_client) -> list[str]:
    """Uploads each {filename: content} pair to s3_uri_prefix + filename.
    Takes an injected s3_client so this is testable without real AWS creds.

    NOTE: put_object is a single-part upload and caps at 5 GB. The real
    corpus is ~0.74 GB of raw text (sampled 2026-08-31), so this isn't a
    live risk today — but if the corpus grows an order of magnitude,
    sharded output (train-00000-of-000NN.jsonl) or multipart upload would
    be required."""
    if not s3_uri_prefix.endswith("/"):
        s3_uri_prefix += "/"
    bucket, prefix = s3_uri_prefix.replace("s3://", "").split("/", 1)
    uploaded = []
    for filename, content in files.items():
        key = prefix + filename
        with_retries(
            s3_client.put_object, retries=3, backoff_seconds=1.0,
            Bucket=bucket, Key=key, Body=content.encode("utf-8"),
        )
        uploaded.append(f"s3://{bucket}/{key}")
    return uploaded


def build_wandb_config(**overrides) -> dict:
    """Returns this run's tunable-constant config for W&B, with any keyword
    overrides layered on top. No duplication risk here (unlike the earlier
    standalone wandb_tracking/wandb_logger.py) since this reads directly
    from this script's own live constants."""
    config = {
        "model_name": MODEL_NAME,
        "chunk_target_tokens": CHUNK_TARGET_TOKENS,
        "chunk_overlap_tokens": CHUNK_OVERLAP_TOKENS,
        "max_section_tokens": MAX_SECTION_TOKENS,
        "pack_seq_len": PACK_SEQ_LEN,
        "pack_separator": PACK_SEPARATOR,
        "val_fraction": VAL_FRACTION,
        "split_seed": SPLIT_SEED,
        "s3_fetch_workers": S3_FETCH_WORKERS,
    }
    config.update(overrides)
    return config


def log_run_to_wandb(
    wandb_module,
    project: str,
    config: dict,
    report: dict,
    artifact_uris: list[str],
    artifact_name: str = "pretuning-dataset",
) -> None:
    """Logs this run's config and stats to W&B, and — if artifact_uris is
    non-empty (a real S3 upload happened) — registers the output as a
    versioned dataset Artifact referencing those S3 locations rather than
    re-uploading content. Wrapped in try/finally so the run always finishes
    (exit_code=1 on failure, re-raising) instead of being left "running"
    forever on the dashboard if something fails partway through."""
    run = wandb_module.init(project=project, config=config, job_type="data-prep")
    try:
        run.log(report)
        if artifact_uris:
            artifact = wandb_module.Artifact(name=artifact_name, type="dataset")
            for uri in artifact_uris:
                artifact.add_reference(uri)
            run.log_artifact(artifact)
    except Exception:
        run.finish(exit_code=1)
        raise
    else:
        run.finish(exit_code=0)


def main():
    parser = argparse.ArgumentParser(description="Chunk, split, and pack finance documents for pre-tuning.")
    parser.add_argument("--doc-keys-from", help="Path to a text file, one S3 doc_key per line")
    parser.add_argument(
        "--local-docs-dir",
        help="Directory of local .txt files to use instead of S3 (for offline dry runs)",
    )
    parser.add_argument(
        "--s3-prefix",
        help="S3 URI prefix to upload train.jsonl/val.jsonl/report.md to (required unless --dry-run)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip the S3 upload; write outputs to --dry-run-out-dir instead",
    )
    parser.add_argument(
        "--dry-run-out-dir", default="./dry_run_output",
        help="Local directory for --dry-run output",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Allow overwriting files that already exist at --s3-prefix",
    )
    parser.add_argument(
        "--resume-log",
        help="Path to a file tracking already-processed doc_keys (S3 mode only). Entries are "
             "written only after a run's output is uploaded, so an interrupted run correctly "
             "reprocesses everything. A resumed run therefore writes ONLY the new documents, "
             "so point --s3-prefix at a fresh prefix for each resumed run and merge afterwards.",
    )
    parser.add_argument(
        "--wandb", action="store_true",
        help="Log this run's config/stats to W&B, and (unless --dry-run) register the "
             "output as a versioned dataset Artifact referencing its S3 location",
    )
    parser.add_argument("--wandb-project", default="vaia-financial-pretuning")
    parser.add_argument(
        "--wandb-dataset-name", default="pretuning-dataset",
        help="Artifact name — give unrelated corpus slices/configs distinct names so "
             "their Artifact versions don't pile up together under one lineage",
    )
    args = parser.parse_args()

    if not args.doc_keys_from and not args.local_docs_dir:
        parser.error("one of --doc-keys-from or --local-docs-dir is required")
    if not args.dry_run and not args.s3_prefix:
        parser.error("--s3-prefix is required unless --dry-run is set")

    tokenizer = load_tokenizer()

    docs: list[tuple[str, str]] = []
    failed_doc_keys: list[str] = []
    loaded_doc_keys: list[str] = []

    if args.local_docs_dir:
        docs = load_local_docs(args.local_docs_dir)
    else:
        import boto3
        # max_pool_connections must track S3_FETCH_WORKERS -- botocore's
        # default pool size is 10, which silently caps concurrency at 10 in
        # flight no matter how large the thread pool is, starving the rest
        # of the worker threads. Same class of fix the SFT pipeline's own
        # requests.Session pool sizing already documented.
        read_client = boto3.client("s3", config=BotoConfig(max_pool_connections=S3_FETCH_WORKERS))
        with open(args.doc_keys_from, "r", encoding="utf-8") as f:
            doc_keys = [line.strip() for line in f if line.strip()]
        already_processed = load_resume_log(args.resume_log) if args.resume_log else set()
        pending_doc_keys = [k for k in doc_keys if k not in already_processed]
        docs, failed_doc_keys = load_documents_concurrently(pending_doc_keys, read_client, S3_FETCH_WORKERS)
        loaded_doc_keys = [doc_key for doc_key, _ in docs]

    if failed_doc_keys:
        print(f"WARNING: {len(failed_doc_keys)} document(s) failed to load and were skipped.")
        for doc_key in failed_doc_keys[:20]:
            print(f"  - {doc_key}")
        if len(failed_doc_keys) > 20:
            print(f"  ... and {len(failed_doc_keys) - 20} more")

    all_chunks = []
    for doc_key, text in docs:
        all_chunks.extend(
            chunk_document(doc_key, text, tokenizer, CHUNK_TARGET_TOKENS, CHUNK_OVERLAP_TOKENS, MAX_SECTION_TOKENS)
        )

    all_chunks, n_dropped = dedupe_chunks(all_chunks)
    if n_dropped:
        print(f"Dropped {n_dropped} exact-duplicate chunk(s) before splitting.")

    splits = split_by_document(all_chunks, VAL_FRACTION, SPLIT_SEED)
    packed_splits = {
        name: pack_chunks(chunks, tokenizer, PACK_SEQ_LEN, PACK_SEPARATOR)
        for name, chunks in splits.items()
    }

    report = generate_stats_report(splits, packed_splits, n_dropped)
    files = build_output_files(packed_splits, render_markdown(report))

    artifact_uris: list[str] = []
    if args.dry_run:
        for path in write_output_files_locally(files, args.dry_run_out_dir):
            print(f"[dry run] Wrote {path}")
    else:
        import boto3
        s3_client = boto3.client("s3")
        if not args.overwrite:
            existing = check_existing_keys(files, args.s3_prefix, s3_client)
            if existing:
                parser.error(
                    f"Refusing to overwrite existing S3 object(s): {existing}. "
                    "Pass --overwrite to proceed anyway."
                )
        for uri in upload_outputs_to_s3(files, args.s3_prefix, s3_client):
            print(f"Uploaded {uri}")
            if uri.endswith((".jsonl",)):
                artifact_uris.append(uri)

        # Resume log is written only now, after the output is durably in S3 —
        # see the note in --resume-log's help text above.
        if args.resume_log and loaded_doc_keys:
            append_resume_log(args.resume_log, loaded_doc_keys)
            print(f"Recorded {len(loaded_doc_keys)} doc_key(s) in {args.resume_log}")

    if args.wandb:
        import wandb
        log_run_to_wandb(
            wandb, args.wandb_project, build_wandb_config(), report, artifact_uris,
            artifact_name=args.wandb_dataset_name,
        )
        print(f"Logged run to W&B project '{args.wandb_project}'.")


if __name__ == "__main__":
    main()
