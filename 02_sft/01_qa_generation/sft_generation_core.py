"""Chunk documents straight from S3 and generate SFT training data from them, one
model (gpt-oss-120b on Cloudflare Workers AI), as many parallel workers as the
account's rate limit allows, with incremental JSONL checkpointing so re-running
after new documents land only processes what's new.

Documents are selected by capability code (see sft_config.CAPABILITY_GROUPS); one
run covers all 8 capabilities.

Output records are written in the format Qwen3.5-9B expects for fine-tuning: the
source chunk plus the question in the user turn, and the model's own
<think>...</think>-wrapped reasoning followed by the answer in the assistant turn.

Sized for a large real corpus: documents are streamed in batches rather than loaded
up front, HTTP connections are pooled and reused, authentication failures abort
instead of retrying across the whole run, truncated model replies are salvaged
rather than discarded, and a genuinely empty (but valid) model reply -- the prompt's
own "return [] if the document doesn't support a good question" instruction -- is
treated as success, not an error.
"""
import argparse
import hashlib
import json
import os
import random
import threading
import time
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import requests
from botocore.config import Config as BotoConfig

_write_lock = threading.Lock()

# Set when the run hits something no amount of retrying will fix (bad credentials,
# account disabled, unrecognized model). Workers check it and bail instead of each
# burning MAX_RETRIES against a wall.
_abort = threading.Event()
_abort_reason = []


class FatalAPIError(Exception):
    """Not retryable, and not survivable: stop the run."""


class PermanentChunkError(Exception):
    """Not retryable, but only kills this one chunk."""


def _build_session(max_workers):
    """One pooled session for the whole run. A bare requests.post() per call opens
    a fresh TCP+TLS connection every time; across hundreds of thousands of calls
    that is both slow and a real risk of local port exhaustion."""
    session = requests.Session()
    session.mount(
        "https://",
        requests.adapters.HTTPAdapter(pool_connections=max_workers, pool_maxsize=max_workers),
    )
    return session


def _s3_client(config):
    return boto3.client(
        "s3",
        region_name=config.AWS_REGION,
        config=BotoConfig(retries={"max_attempts": 5, "mode": "adaptive"},
                          max_pool_connections=config.S3_FETCH_WORKERS),
    )


# --- capability-based corpus split ---

def extract_capability_code(key: str, s3_prefix: str) -> str | None:
    """First token before '_' in the first path segment after s3_prefix.

    e.g. '<prefix>/AR_AUTONOMOUS_CLOSE_THE_LOOP_EXECUTION - .../doc.md' -> 'AR'.
    Returns None if the segment is empty or has no leading token.
    """
    relative = key[len(s3_prefix):] if key.startswith(s3_prefix) else key
    relative = relative.lstrip("/")
    first_segment = relative.split("/", 1)[0]
    code = first_segment.split("_", 1)[0].strip()
    return code or None


def assign_group(key: str, s3_prefix: str, capability_groups: dict) -> str | None:
    """Returns the capability-group name this key belongs to, or None if its
    capability code isn't recognized by any group."""
    code = extract_capability_code(key, s3_prefix)
    if code is None:
        return None
    for group_name, codes in capability_groups.items():
        if code in codes:
            return group_name
    return None


def chunk_id(key, index):
    """Stable id for an (S3 key, chunk index) pair, independent of run order."""
    hash_key = f"{key}::{index}"
    return hashlib.sha1(hash_key.encode("utf-8")).hexdigest()[:16]


def iter_chunks(text, size, overlap):
    # An overlap >= size would leave `start` where it was and loop forever, emitting
    # chunks until memory runs out. Both are env-tunable, so clamp rather than trust.
    overlap = max(0, min(overlap, size - 1))
    if len(text) <= size:
        if text.strip():
            yield text
        return
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        piece = text[start:end]
        if piece.strip():
            yield piece
        if end == len(text):
            break
        start = end - overlap


def list_s3_keys(s3, bucket, s3_prefix, input_extensions, capability_group, capability_groups):
    """Lists keys under s3_prefix, splits them by capability group.

    Returns (matched_keys, unmatched_counts) where matched_keys are only the keys
    belonging to capability_group, and unmatched_counts is a dict of
    {unrecognized_code: count} for keys whose capability code isn't in ANY group --
    those documents are skipped entirely, not processed by either machine.
    """
    matched = []
    unmatched = defaultdict(int)
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if os.path.splitext(key)[1].lower() not in input_extensions:
                continue
            group = assign_group(key, s3_prefix, capability_groups)
            if group is None:
                code = extract_capability_code(key, s3_prefix) or "<none>"
                unmatched[code] += 1
                continue
            if group == capability_group:
                matched.append(key)
    return matched, dict(unmatched)


def fetch_and_chunk(s3, bucket, key, chunk_chars, chunk_overlap_chars, max_chunks_per_doc):
    """Fetch one document and split it. Returns (chunks, n_chunks, error_or_None)."""
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        text = body.decode("utf-8", errors="ignore")
    except Exception as e:  # noqa: BLE001
        return [], 0, str(e)

    pieces = list(iter_chunks(text, chunk_chars, chunk_overlap_chars))
    if max_chunks_per_doc:
        pieces = pieces[:max_chunks_per_doc]
    n = len(pieces)
    return [(chunk_id(key, i), key, piece, n) for i, piece in enumerate(pieces)], n, None


def fetch_batch(s3, bucket, keys, chunk_chars, chunk_overlap_chars, max_chunks_per_doc, fetch_workers):
    """Fetch + chunk a batch of documents in parallel. Bounded by DOC_BATCH_SIZE."""
    chunks = []
    failed = 0
    with ThreadPoolExecutor(max_workers=fetch_workers) as pool:
        for got, _n, err in pool.map(
            lambda k: fetch_and_chunk(s3, bucket, k, chunk_chars, chunk_overlap_chars, max_chunks_per_doc), keys
        ):
            if err:
                failed += 1
                continue
            chunks.extend(got)
    return chunks, failed


def load_progress(output_path):
    """Replay the output file to find what's already generated.

    Returns (done_chunk_ids, fully_done_source_keys). Knowing a source is *fully*
    done lets the next run skip re-downloading it entirely.
    """
    done = set()
    seen_per_source = defaultdict(set)
    expected_per_source = {}
    if not os.path.exists(output_path):
        return done, set()
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = rec.get("_chunk_id")
            if not cid:
                continue
            done.add(cid)
            src = rec.get("_source")
            n = rec.get("_n_chunks")
            if src and isinstance(n, int) and n > 0:
                seen_per_source[src].add(cid)
                expected_per_source[src] = n
    complete = {src for src, n in expected_per_source.items() if len(seen_per_source[src]) >= n}
    return done, complete


def log_line(message, run_log_path):
    """Prints AND persists one line to run_log_path. A logging failure must never
    take the run down -- if the log file can't be written (disk full, permissions),
    the message still reaches the console."""
    print(message, flush=True)
    try:
        with open(run_log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    except OSError:
        pass


def document_is_settled(source_key, attempted_per_source, expected_pending_per_source):
    """True once every chunk this run intended to attempt for source_key has been
    attempted (succeeded, was a legitimate empty result, or failed all retries) --
    the boundary at which it's safe to honor a pause request without leaving this
    document half-processed."""
    expected = expected_pending_per_source.get(source_key, 0)
    return expected > 0 and len(attempted_per_source.get(source_key, ())) >= expected


def append_examples(records, output_path):
    with _write_lock:
        with open(output_path, "a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_error(cid, source_key, error, log_path):
    with _write_lock:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"chunk_id": cid, "source": source_key, "error": error, "ts": time.time()},
                               ensure_ascii=False) + "\n")


def verify_model_available(model_id, account_id, api_token, session, timeout=30):
    """Confirms model_id is actually in this account's Workers AI model catalog
    before the run starts, rather than discovering a renamed/unavailable model
    only after burning retries against real chunks.

    Cloudflare's docs don't specify the exact field name each model object uses for
    its slug, so this searches the full JSON text of the response for model_id
    rather than assuming one specific key.
    """
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/models/search"
    try:
        resp = session.get(
            url,
            headers={"Authorization": f"Bearer {api_token}"},
            params={"search": model_id},
            timeout=timeout,
        )
    except requests.RequestException as e:
        raise FatalAPIError(f"model catalog check failed: {e}") from e

    if resp.status_code in (401, 403):
        raise FatalAPIError(f"HTTP {resp.status_code} checking model catalog (auth/permission): {resp.text[:300]}")
    if resp.status_code != 200:
        raise FatalAPIError(f"HTTP {resp.status_code} checking model catalog: {resp.text[:300]}")

    return model_id in resp.text


def call_gpt_oss(prompt, session, model_id, account_id, api_token, system_prompt, reasoning_effort,
                  max_output_tokens, timeout):
    """One generation call. Returns (text, truncated). Raises on failure.

    Errors are classified rather than lumped together: an expired token is not a
    transient blip and must not be retried across the whole run, while a 429 or a
    5xx is exactly what the retry loop exists for.
    """
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model_id}"
    try:
        resp = session.post(
            url,
            headers={"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"},
            json={
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "reasoning": {"effort": reasoning_effort},
                "temperature": 0.4,
                "max_tokens": max_output_tokens,
            },
            timeout=timeout,
        )
    except requests.RequestException as e:
        raise RuntimeError(f"request failed: {e}") from e

    if resp.status_code in (401, 403):
        raise FatalAPIError(f"HTTP {resp.status_code} (auth/permission): {resp.text[:300]}")
    if resp.status_code == 429:
        retry_after = resp.headers.get("Retry-After")
        raise RuntimeError(f"HTTP 429 rate limited (retry-after={retry_after}): {resp.text[:200]}")
    if resp.status_code == 400:
        raise PermanentChunkError(f"HTTP 400: {resp.text[:300]}")
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    if data.get("success") is False:
        raise RuntimeError(f"workers-ai error: {json.dumps(data.get('errors', data))[:400]}")
    result = data.get("result")
    if not isinstance(result, dict):
        raise RuntimeError(f"unrecognized response shape: {json.dumps(data)[:300]}")

    choices = result.get("choices")
    if choices:
        truncated = choices[0].get("finish_reason") == "length"
        return choices[0]["message"].get("content", ""), truncated
    if "response" in result:
        return result["response"], False
    raise RuntimeError(f"unrecognized response shape: {json.dumps(result)[:300]}")


def iter_json_objects(text):
    """Yield complete top-level {...} objects, brace-counting outside of strings.

    A truncated reply still yields every pair that came through complete, instead of
    parsing as nothing at all.
    """
    depth = 0
    start = None
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    yield text[start:i + 1]
                    start = None


def parse_examples(raw_text):
    """Returns a list of {question_type, difficulty, section_context, question,
    thinking, answer} dicts (question_type/difficulty/section_context optional,
    question/thinking/answer required to keep a pair).

    A genuinely empty result (no objects found, or objects found but none had the
    three required fields, when the reply was a deliberate `[]`) is a VALID outcome
    -- returns an empty list, does not raise -- per the prompt's own instruction to
    return [] when a chunk doesn't support a good question. Only a reply with zero
    parseable JSON objects at all still counts as a real parse failure... except an
    empty array parses to zero objects too, so the distinguishing signal is whether
    the raw reply, stripped, is a bare `[]` (or `[ ]`) -- a deliberate empty array --
    versus prose/garbage that never even formed a `[...]` shape.
    """
    stripped = raw_text.strip()
    is_deliberate_empty = stripped in ("[]", "[ ]")

    out = []
    found_any_object = False
    for blob in iter_json_objects(raw_text):
        found_any_object = True
        try:
            item = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        thinking = str(item.get("thinking", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if question and thinking and answer:
            out.append({
                "question_type": str(item.get("question_type", "")).strip(),
                "difficulty": str(item.get("difficulty", "")).strip(),
                "section_context": str(item.get("section_context", "")).strip(),
                "question": question,
                "thinking": thinking,
                "answer": answer,
            })

    if not out and not found_any_object and not is_deliberate_empty:
        raise ValueError("no complete question/thinking/answer objects in model output")
    return out


def build_record(cid, source_key, chunk_text, n_chunks, ex, model_id, trained_system_prompt):
    """One training example in Qwen3.5's expected shape.

    The chunk goes in the user turn (open-book: the model learns to reason over
    provided context, matching how these pairs were generated and how retrieval
    works at inference). `thinking` is used as-is in the assistant turn -- the
    generation prompt already requires the model to emit it pre-wrapped in
    <think>...</think>, so it is NOT re-wrapped here (that would double-wrap it).
    """
    return {
        "messages": [
            {"role": "system", "content": trained_system_prompt},
            {"role": "user", "content": f"<document>\n{chunk_text}\n</document>\n\n{ex['question']}"},
            {"role": "assistant", "content": f"{ex['thinking']}\n\n{ex['answer']}"},
        ],
        "_chunk_id": cid,
        "_source": source_key,
        "_n_chunks": n_chunks,
        "_question_type": ex.get("question_type", ""),
        "_difficulty": ex.get("difficulty", ""),
        "_section_context": ex.get("section_context", ""),
        "_generated_by": model_id,
        "_ts": time.time(),
    }


def upload_output_to_s3(local_path, s3_prefix, s3_client):
    """Full-file upload of local_path to s3_prefix (bucket+key parsed from the
    s3://bucket/prefix/ form). Returns the resulting s3:// URI.

    A durability mirror against process/pod loss -- local_path stays authoritative
    for resume (load_progress reads local disk only).
    """
    assert s3_prefix.startswith("s3://"), f"expected an s3:// prefix, got {s3_prefix!r}"
    without_scheme = s3_prefix[len("s3://"):]
    bucket, _, prefix = without_scheme.partition("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    filename = os.path.basename(local_path)
    key = f"{prefix}{filename}"
    with open(local_path, "rb") as f:
        s3_client.put_object(Bucket=bucket, Key=key, Body=f.read())
    return f"s3://{bucket}/{key}"


def process_chunk(cid, source_key, chunk_text, n_chunks, config, session):
    """Returns (n_written, outcome) where outcome is one of:
    None (real success, n_written may be 0 or more if [] was legitimate),
    "empty" (n_written == 0, a deliberate [] -- not an error),
    or an error message string (n_written == 0, a real failure)."""
    if _abort.is_set():
        return 0, "aborted"
    prompt = config.GENERATION_INSTRUCTIONS.format(chunk_text=chunk_text)
    last_err = None
    for attempt in range(1, config.MAX_RETRIES + 1):
        if _abort.is_set():
            return 0, "aborted"
        try:
            raw, truncated = call_gpt_oss(
                prompt, session, config.MODEL_ID, config.CLOUDFLARE_ACCOUNT_ID, config.CLOUDFLARE_API_TOKEN,
                config.SYSTEM_PROMPT, config.REASONING_EFFORT, config.MAX_OUTPUT_TOKENS, config.REQUEST_TIMEOUT_SECONDS,
            )
            examples = parse_examples(raw)
            if not examples:
                return 0, "empty"
            records = [build_record(cid, source_key, chunk_text, n_chunks, ex, config.MODEL_ID, config.TRAINED_SYSTEM_PROMPT)
                       for ex in examples]
            append_examples(records, config.OUTPUT_PATH)
            if truncated:
                append_error(cid, source_key, f"truncated reply: kept {len(records)} pairs", config.LOG_PATH)
            return len(records), None
        except FatalAPIError as e:
            last_err = str(e)
            if not _abort.is_set():
                _abort_reason.append(last_err)
                _abort.set()
            append_error(cid, source_key, last_err, config.LOG_PATH)
            return 0, last_err
        except PermanentChunkError as e:
            last_err = str(e)
            append_error(cid, source_key, last_err, config.LOG_PATH)
            return 0, last_err
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            if attempt < config.MAX_RETRIES:
                delay = min(config.RETRY_BASE_SECONDS ** attempt, 30)
                time.sleep(delay + random.uniform(0, delay * 0.5))
    append_error(cid, source_key, last_err, config.LOG_PATH)
    return 0, last_err


def main(config):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", type=str, default="", help="override config.S3_PREFIX")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N pending chunks (smoke test)")
    ap.add_argument("--manifest", type=str, default=os.environ.get("SFT_KEY_ALLOWLIST", ""),
                     help="path to a debt_manifest.jsonl-style file (one JSON object per line "
                          "with a 'key' field); when set, only these S3 keys are considered for "
                          "this capability group instead of the full S3_PREFIX listing. Omit for "
                          "today's full-scan behavior (main-set runs are unaffected).")
    args = ap.parse_args()

    if args.prefix:
        config.S3_PREFIX = args.prefix

    for path in (config.OUTPUT_PATH, config.LOG_PATH, config.RUN_LOG_PATH):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def _log(message):
        log_line(message, config.RUN_LOG_PATH)

    # A pause request from a prior run that was never consumed (e.g. the process
    # was killed before it noticed) must not silently re-pause this run instantly --
    # clear it before we even start looking for it.
    if os.path.exists(config.PAUSE_SENTINEL_PATH):
        os.remove(config.PAUSE_SENTINEL_PATH)

    session = _build_session(config.MAX_WORKERS)

    _log(f"verifying model {config.MODEL_ID} against the Cloudflare account's catalog...")
    try:
        if not verify_model_available(config.MODEL_ID, config.CLOUDFLARE_ACCOUNT_ID, config.CLOUDFLARE_API_TOKEN, session):
            _log(f"ABORTED: model {config.MODEL_ID} was not found in this account's Workers AI catalog. "
                 "Nothing was attempted.")
            return
    except FatalAPIError as e:
        _log(f"ABORTED: {e}\nNothing was attempted.")
        return
    _log("model verified.")

    s3 = _s3_client(config)
    _log(f"listing s3://{config.S3_BUCKET}/{config.S3_PREFIX} for capability group "
         f"'{config.CAPABILITY_GROUP}' ({sorted(config.CAPABILITY_GROUPS[config.CAPABILITY_GROUP])}) ...")
    keys, unmatched = list_s3_keys(s3, config.S3_BUCKET, config.S3_PREFIX, config.INPUT_EXTENSIONS,
                                    config.CAPABILITY_GROUP, config.CAPABILITY_GROUPS)
    _log(f"{len(keys)} documents found for this group")
    if unmatched:
        _log(f"NOTE: skipped {sum(unmatched.values())} documents with unrecognized capability codes: {unmatched}")

    if args.manifest:
        allowlist = set()
        with open(args.manifest, "r", encoding="utf-8") as mf:
            for line in mf:
                line = line.strip()
                if not line:
                    continue
                allowlist.add(json.loads(line)["key"])
        before = len(keys)
        keys = [k for k in keys if k in allowlist]
        _log(f"--manifest {args.manifest}: restricted {before} -> {len(keys)} keys "
             f"({len(allowlist)} keys in this group's manifest slice)")

    done_ids, complete_sources = load_progress(config.OUTPUT_PATH)
    if done_ids:
        _log(f"resuming - {len(done_ids)} chunks already generated, "
             f"{len(complete_sources)} documents fully complete")
    keys = [k for k in keys if k not in complete_sources]
    _log(f"{len(keys)} documents to scan this run")

    completed = chunks_seen = total_examples = errors = empty_chunks = fetch_failures = 0
    last_s3_upload_at = 0
    start = time.time()
    stop_early = False
    paused = False
    # Live, this-run-only tracking of which chunks have been attempted per source,
    # so a pause request can be honored right after the document currently being
    # worked on finishes, not after the whole in-flight batch (up to DOC_BATCH_SIZE
    # documents) drains.
    attempted_per_source = defaultdict(set)

    def maybe_upload_to_s3(force=False):
        nonlocal last_s3_upload_at
        if not force and completed - last_s3_upload_at < config.S3_UPLOAD_INTERVAL_CHUNKS:
            return
        last_s3_upload_at = completed
        try:
            for path in (config.OUTPUT_PATH, config.LOG_PATH, config.RUN_LOG_PATH):
                if os.path.exists(path):
                    upload_output_to_s3(path, config.S3_OUTPUT_PREFIX, s3)
            _log(f"[s3 sync @ {completed} chunks] uploaded to {config.S3_OUTPUT_PREFIX}")
        except Exception as e:  # noqa: BLE001
            _log(f"WARNING: S3 upload failed (will retry at next interval): {e}")

    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as gen_pool:
        for batch_start in range(0, len(keys), config.DOC_BATCH_SIZE):
            if _abort.is_set() or stop_early or paused:
                break
            batch_keys = keys[batch_start:batch_start + config.DOC_BATCH_SIZE]
            batch_chunks, failed = fetch_batch(s3, config.S3_BUCKET, batch_keys, config.CHUNK_CHARS,
                                                config.CHUNK_OVERLAP_CHARS, config.MAX_CHUNKS_PER_DOC,
                                                config.S3_FETCH_WORKERS)
            fetch_failures += failed

            pending = [c for c in batch_chunks if c[0] not in done_ids]
            if args.limit:
                room = args.limit - chunks_seen
                if room <= 0:
                    break
                pending = pending[:room]
            if not pending:
                continue
            chunks_seen += len(pending)
            # How many of THIS run's pending chunks belong to each source -- a
            # source is "settled" once attempted_per_source reaches this count,
            # regardless of how many chunks it originally had (earlier chunks may
            # already be done from a prior run and so never appear in `pending`).
            expected_pending_per_source = Counter(c[1] for c in pending)

            # Submitted in small windows (MAX_WORKERS at a time) rather than the
            # whole batch (up to DOC_BATCH_SIZE documents' worth of chunks) at
            # once, so a pause request is noticed within roughly one window, not
            # after waiting for hundreds of documents already in flight.
            for w_start in range(0, len(pending), config.MAX_WORKERS):
                if _abort.is_set():
                    break
                window = pending[w_start:w_start + config.MAX_WORKERS]
                futures = {gen_pool.submit(process_chunk, cid, key, text, n, config, session): (cid, key)
                           for cid, key, text, n in window}
                for fut in as_completed(futures):
                    cid, key = futures[fut]
                    n_written, outcome = fut.result()
                    completed += 1
                    total_examples += n_written
                    attempted_per_source[key].add(cid)
                    if outcome == "empty":
                        empty_chunks += 1
                    elif outcome:
                        errors += 1
                        _log(f"[{completed}] ERROR {key}: {outcome[:150]}")
                    if completed % 20 == 0:
                        elapsed = time.time() - start
                        rate = completed / elapsed if elapsed > 0 else 0
                        pairs_hr = total_examples / elapsed * 3600 if elapsed > 0 else 0
                        _log(f"[docs {min(batch_start + config.DOC_BATCH_SIZE, len(keys))}/{len(keys)}] "
                             f"{completed} chunks, {total_examples} examples, {empty_chunks} empty, {errors} errors, "
                             f"{rate:.2f} chunks/s (~{pairs_hr:,.0f} pairs/hr)")
                    maybe_upload_to_s3()

                if args.limit and chunks_seen >= args.limit:
                    stop_early = True

                # Pause is only honored once every document touched by this window
                # has been fully attempted -- never mid-document.
                if os.path.exists(config.PAUSE_SENTINEL_PATH):
                    window_sources = {key for _, key, _, _ in window}
                    if all(document_is_settled(src, attempted_per_source, expected_pending_per_source)
                           for src in window_sources):
                        os.remove(config.PAUSE_SENTINEL_PATH)  # consume it, so the next run starts clean
                        paused = True

                if stop_early or paused:
                    break

    maybe_upload_to_s3(force=True)

    if paused:
        docs_settled_this_run = sum(
            1 for src, expected in expected_pending_per_source.items()
            if document_is_settled(src, attempted_per_source, expected_pending_per_source)
        )
        _log(f"\nPAUSED by request. {completed} chunks attempted this run, {total_examples} examples written, "
             f"{empty_chunks} empty, {errors} errors, {docs_settled_this_run} documents in the final batch settled "
             f"before stopping.")
        _log(f"Resume anytime by re-running the same command -- it picks up exactly where this left off "
             f"via {config.OUTPUT_PATH}. No cleanup needed.")
        return

    if _abort.is_set():
        _log(f"\nABORTED: {_abort_reason[0] if _abort_reason else 'fatal error'}")
        _log("Nothing further was attempted. Fix the above and re-run - "
             "everything already written is kept and will be skipped.")

    elapsed = max(time.time() - start, 1e-6)
    _log(f"\nDone. {completed} chunks attempted, {total_examples} examples written, "
         f"{empty_chunks} chunks legitimately empty, {errors} errors, {fetch_failures} documents unreadable.")
    _log(f"Elapsed {elapsed/60:.1f} min ({total_examples/elapsed*3600:,.0f} pairs/hr).")
    _log(f"Output: {config.OUTPUT_PATH}")
    if errors or fetch_failures:
        _log(f"Errors logged to: {config.LOG_PATH}")
