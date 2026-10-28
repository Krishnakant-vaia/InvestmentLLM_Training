"""Downloads Glaive function-calling-v2 and Smoltalk (config=all, train split) from
HuggingFace as parquet, converts each to our {"messages": [{"role","content"}]} shape,
and appends them after the (already stripped) 178k pool to build the merged training file.

No LLM calls anywhere in this script -- both source schemas are machine-parseable, so
conversion is deterministic text/JSON reshaping. Parallelism is used only for the
network-bound parquet downloads (a thread pool), which is safe and doesn't load the CPU.

  python3 -u scripts/merge_pool_sources.py
"""
import concurrent.futures
import json
import random
import re
import sys

import pyarrow.parquet as pq
import requests

sys.path.insert(0, ".")

SEED = 42
GLAIVE_TARGET = 18000
SMOLTALK_TARGET = 12000

GLAIVE_PARQUET = "https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
SMOLTALK_SHARDS = [
    f"https://huggingface.co/datasets/HuggingFaceTB/smoltalk/resolve/refs%2Fconvert%2Fparquet/all/train/{i:04d}.parquet"
    for i in range(9)
]

STRIPPED_POOL = "merged_output/sft_data_finetune_pool_178k_stripped.jsonl"
OUT = "merged_output/sft_data_finetune_pool_merged.jsonl"
MANIFEST = "merged_output/finetune_pool_merge_manifest.json"

DOWNLOAD_DIR = "merged_output/hf_downloads"


def _download(url, dest):
    import os
    if os.path.exists(dest):
        return dest
    r = requests.get(url, timeout=120, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    return dest


def download_all():
    import os
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    jobs = [(GLAIVE_PARQUET, f"{DOWNLOAD_DIR}/glaive_0000.parquet")]
    jobs += [(url, f"{DOWNLOAD_DIR}/smoltalk_{i:04d}.parquet") for i, url in enumerate(SMOLTALK_SHARDS)]
    paths = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(_download, url, dest): dest for url, dest in jobs}
        for fut in concurrent.futures.as_completed(futs):
            dest = fut.result()
            print(f"downloaded {dest}", flush=True)
            paths[dest] = dest
    return (f"{DOWNLOAD_DIR}/glaive_0000.parquet",
            [f"{DOWNLOAD_DIR}/smoltalk_{i:04d}.parquet" for i in range(9)])


# ---- Glaive conversion -------------------------------------------------------
_TURN = re.compile(r"(USER|ASSISTANT|FUNCTION RESPONSE):\s*(.*?)(?=(?:USER|ASSISTANT|FUNCTION RESPONSE):|\Z)", re.DOTALL)


def glaive_row_to_messages(system_text, chat_text):
    system_text = system_text[len("SYSTEM:"):].strip() if system_text.startswith("SYSTEM:") else system_text.strip()
    messages = [{"role": "system", "content": system_text}]
    for tag, body in _TURN.findall(chat_text):
        body = body.replace("<|endoftext|>", "").strip()
        if not body:
            continue
        role = "assistant" if tag == "ASSISTANT" else "user"  # FUNCTION RESPONSE -> user (see design notes)
        messages.append({"role": role, "content": body})
    return messages


def build_glaive(path, n, seed):
    table = pq.read_table(path, columns=["system", "chat"])
    total = table.num_rows
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(total), n))
    sys_col = table.column("system")
    chat_col = table.column("chat")
    out = []
    for i in idx:
        messages = glaive_row_to_messages(str(sys_col[i]), str(chat_col[i]))
        if len(messages) < 2:  # malformed row, no parsed turns beyond system
            continue
        out.append({"messages": messages})
    print(f"glaive: {total} rows available -> sampled {len(idx)} -> {len(out)} converted", flush=True)
    return out


# ---- Smoltalk conversion ------------------------------------------------------
def smoltalk_source_counts(paths):
    counts = {}
    for p in paths:
        col = pq.read_table(p, columns=["source"]).column("source")
        for v in col:
            s = str(v)
            counts[s] = counts.get(s, 0) + 1
    return counts


def build_smoltalk(paths, n, seed):
    counts = smoltalk_source_counts(paths)
    sources = sorted(counts)
    quota = {s: n // len(sources) for s in sources}
    for s in sources[: n % len(sources)]:
        quota[s] += 1
    print(f"smoltalk: {len(sources)} sub-sources, total rows {sum(counts.values())}", flush=True)
    for s in sources:
        print(f"  {s}: available={counts[s]} quota={min(quota[s], counts[s])}", flush=True)

    rng = random.Random(seed)
    # reservoir-sample row indices per source across all shards in one pass
    reservoirs = {s: [] for s in sources}
    seen = {s: 0 for s in sources}
    row_cursor = 0
    for p in paths:
        table = pq.read_table(p, columns=["messages", "source"])
        src_col = table.column("source")
        msg_col = table.column("messages")
        for i in range(table.num_rows):
            s = str(src_col[i])
            k = quota.get(s, 0)
            if k == 0:
                continue
            seen[s] += 1
            item = (row_cursor, msg_col[i])
            if len(reservoirs[s]) < k:
                reservoirs[s].append(item)
            else:
                j = rng.randint(0, seen[s] - 1)
                if j < k:
                    reservoirs[s][j] = item
            row_cursor += 1

    out = []
    for s in sources:
        for _, msg_val in reservoirs[s]:
            messages = [{"role": m["role"], "content": m["content"]} for m in msg_val.as_py()]
            out.append({"messages": messages})
    print(f"smoltalk: sampled {len(out)} total", flush=True)
    return out


def main():
    glaive_path, smoltalk_paths = download_all()

    glaive_records = build_glaive(glaive_path, GLAIVE_TARGET, SEED)
    smoltalk_records = build_smoltalk(smoltalk_paths, SMOLTALK_TARGET, SEED)

    counts = {}
    with open(OUT, "w", encoding="utf-8", newline="\n") as out:
        n_pool = 0
        with open(STRIPPED_POOL, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                out.write(line if line.endswith("\n") else line + "\n")
                n_pool += 1
        counts["our_pool"] = n_pool
        start = n_pool
        for r in glaive_records:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts["glaive"] = len(glaive_records)
        mid = start + len(glaive_records)
        for r in smoltalk_records:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts["smoltalk"] = len(smoltalk_records)
        end = mid + len(smoltalk_records)

    manifest = {
        "total": end,
        "line_ranges_0indexed": {
            "our_pool": [0, n_pool],
            "glaive_function_calling_v2": [start, mid],
            "smoltalk_all": [mid, end],
        },
        "counts": counts,
        "seed": SEED,
    }
    json.dump(manifest, open(MANIFEST, "w", encoding="utf-8"), indent=2)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
