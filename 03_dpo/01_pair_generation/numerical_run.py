"""Judge the numerical pairs capability by capability on the GPU box, with S3 checkpoints per capability.

    python numerical_run.py --run numerical-1 [--per-capability N] [--resume]

Reads <OUT_ROOT>numerical-input/ (capabilities.json + one jsonl per capability, written by prep_numerical.py), starts vLLM,
then for each capability: builds the judge input (question, golden as reference_solution, two answers on random sides), judges
it with DPO_pair_judge_prompt_numerical.txt, and writes
    <OUT_ROOT><run>/<sheet>/<capability>/judge/        manifest + verdicts/part-*.jsonl checkpoints (checkpoints.py)
    <OUT_ROOT><run>/<sheet>/<capability>/final/judged_pairs.jsonl   every pair with both verdicts and flags (nothing dropped)
    <OUT_ROOT><run>/final/report.json                   per-capability counts, scores and measured time
--per-capability N judges only the first N pairs of each capability (calibration: use a separate --run name).
A capability that is already complete on S3 is skipped, so --resume after an eviction continues where it stopped.
"""
import argparse
import json
import os
import time
from pathlib import Path

import config
from schema import normalize_row
import judge
import prep_numerical
import s3util
from generate import read_jsonl
from pipeline import server_is_up, start_server, wait_ready

NUM_SEQS, NUM_WORKERS = 128, 100                   # short pairs: an H200 holds far more concurrent sequences than the document judge used
INPUT_PREFIX = f"{config.OUT_ROOT}numerical-input/"


def pull_input(s3, work: Path) -> list[dict]:
    caps = json.loads(s3util.get_text(s3, INPUT_PREFIX + "capabilities.json"))
    for c in caps:
        if c["file"]:
            s3util.download(s3, INPUT_PREFIX + c["file"], work / "input" / c["file"])
    return caps


def cap_run(run: str, cap: dict) -> str:
    return f"{run}/{prep_numerical.slug(cap['sheet'])}/{prep_numerical.slug(cap['capability'])}"


def verdict_flags(accepted_side: dict, rejected_side: dict) -> list[str]:
    flags = []
    if accepted_side["score"] < 7:
        flags.append("judge: preferred answer fails")
    if rejected_side["score"] >= 7:
        flags.append("judge: not-preferred answer passes")
    return flags


def distribution(verdicts: list[dict]) -> dict:
    out = {}
    for s in range(1, 11):
        out[s] = sum(v["score"] == s for v in verdicts)
    return {k: v for k, v in out.items() if v}


def judge_capability(s3, a, cap: dict, work: Path, url: str, workers: int) -> dict:
    rows = [normalize_row(r, accepted_is_answer=True) for r in read_jsonl(work / "input" / cap["file"])]     # old inputs: accepted / rejected
    if a.per_capability:
        rows = rows[:a.per_capability]
    run = cap_run(a.run, cap)
    lines, key = prep_numerical.build_judge_input(rows)
    d = work / "caps" / prep_numerical.slug(cap["capability"])
    d.mkdir(parents=True, exist_ok=True)
    (d / "judge_in.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines), encoding="utf-8")
    started = time.time()
    try:
        judge.run(d / "judge_in.jsonl", d / "judge_out.jsonl", workers=workers, url=url, s3_run=run, prompt_path=config.JUDGE_PROMPT_NUMERICAL)
    except RuntimeError as e:                                       # checkpoints made for a different input: never mixed
        raise SystemExit(f"[numerical] {cap['capability']}: {e}")
    seconds = time.time() - started
    verdicts = {v["id"]: v for v in read_jsonl(d / "judge_out.jsonl")}
    out, pref, notpref = [], [], []
    for i, (r, k) in enumerate(zip(rows, key)):
        v = verdicts.get(i)
        if not v:
            out.append({**r, "judge_preferred": None, "judge_not_preferred": None, "flags": ["no judge verdict"]})
            continue
        win, lose = (v["a"], v["b"]) if k["preferred_side"] == "a" else (v["b"], v["a"])
        pref.append(win)
        notpref.append(lose)
        out.append({**r, "judge_preferred": win, "judge_not_preferred": lose, "flags": verdict_flags(win, lose)})
    text = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out)
    s3util.put_text(s3, f"{config.OUT_ROOT}{run}/final/judged_pairs.jsonl", text)
    stat = {"sheet": cap["sheet"], "capability": cap["capability"], "pairs": len(rows), "judged": len(pref),
            "seconds_this_session": round(seconds, 1), "pairs_per_min": round(len(pref) / max(seconds, 1) * 60, 1),
            "preferred_pass": sum(v["score"] >= 7 for v in pref), "not_preferred_pass": sum(v["score"] >= 7 for v in notpref),
            "preferred_scores": distribution(pref), "not_preferred_scores": distribution(notpref)}
    s3util.put_text(s3, f"{config.OUT_ROOT}{run}/final/capability_report.json", json.dumps(stat, indent=2))
    print(f"[numerical] {cap['capability']}: {stat['judged']}/{stat['pairs']} judged, preferred pass {stat['preferred_pass']}, "
          f"not-preferred pass {stat['not_preferred_pass']}, {stat['pairs_per_min']} pairs/min", flush=True)
    return stat


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--per-capability", type=int, help="judge only the first N pairs of each capability (calibration)")
    ap.add_argument("--sheets", help="only these sheets, e.g. \"4,5\" (the sheet number before the dot)")
    ap.add_argument("--resume", action="store_true", help="accepted for the launcher; progress always resumes from S3")
    ap.add_argument("--server-url")
    a = ap.parse_args()
    work = config.HERE / "work" / a.run
    work.mkdir(parents=True, exist_ok=True)
    s3 = s3util.client()
    s3util.put_text(s3, f"{config.OUT_ROOT}{a.run}/LAYOUT.md",
                    "<sheet>/<capability>/judge/ = judge checkpoints; <sheet>/<capability>/final/judged_pairs.jsonl = every pair with verdicts; final/report.json = all capabilities\n")
    caps = pull_input(s3, work)
    if a.sheets:
        wanted = {x.strip() for x in a.sheets.split(",")}
        caps = [c for c in caps if c["sheet"].split(".")[0] in wanted]
        print(f"[numerical] sheets {sorted(wanted)}: {len(caps)} capabilities", flush=True)
    url, server = a.server_url or config.SERVER_URL, None
    workers = int(os.environ.get("DPO_NUM_WORKERS", NUM_WORKERS))
    stats = []
    try:
        if not a.server_url and not server_is_up(url):
            server = start_server(work / "server.log", f"{config.VLLM_BASE_ARGS} --max-num-seqs {NUM_SEQS}")
        wait_ready(url, server)
        for cap in caps:
            if cap["file"]:
                stats.append(judge_capability(s3, a, cap, work, url, workers))
                s3util.put_text(s3, f"{config.OUT_ROOT}{a.run}/final/report.json", json.dumps(stats, indent=2))
    finally:
        if server is not None:
            server.terminate()
    (work / "report.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    total = sum(s["judged"] for s in stats)
    print(f"[numerical] done: {total} pairs judged in {len(stats)} capabilities", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
