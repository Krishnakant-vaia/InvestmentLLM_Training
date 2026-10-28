"""Rewrite the not-preferred answers into hard negatives (same mistake, same form as the preferred answer), on a sample.

    python rewrite_run.py make-sample --pairs PATH --out sample.jsonl [--per-source 10]     # local, CPU
    python rewrite_run.py run --run rewrite-1 [--sample S3 key]                             # on the GPU box (daytona_run.py rewrite)
    python rewrite_run.py readable --results results.jsonl --out PATH                       # local: before/after, easy to read

`run` rewrites each pair with gpt-oss-120b (DPO_hard_negative_rewrite_prompt.txt), then checks the rewrite:
  * code checks: no admission words, length within 0.8x-1.25x of the preferred answer, same number of lines (numerical), a different final answer;
  * the judge (document prompt for document pairs, numerical prompt for numerical pairs) must still pass the preferred answer and FAIL the rewrite.
A pair is `accepted` only if every check passes. Nothing is dropped from the output: every pair keeps its original and its rewrite.
"""
import argparse
import json
import random
import re
import statistics as st
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import config
from schema import normalize_row
import judge
import s3util
from generate import LLM, read_jsonl

PROMPT = config.HERE / "prompts" / "DPO_hard_negative_rewrite_prompt.txt"
SAMPLE_KEY = f"{config.OUT_ROOT}rewrite-input/sample.jsonl"
SHEET_OF_PREFIX = {"DK": "domain_knowledge", "EK": "enterprise_knowledge", "FR": "financial_reasoning", "WU": "workflow_understanding",
                   "CG": "content_generation", "AR": "agentic_reasoning", "CL": "continuous_learning", "EA": "enterprise_ai_applications"}
ADMISSIONS = re.compile(r"\b(mistaken(ly)?|wrong(ly)?|incorrect(ly)?|errors?|misread|misappl\w*|inverted|ignored|omitted|skipped|forgot\w*|"
                        r"instead of|rather than|should have|off-by-one|in other words|mis-?\w+ed)\b", re.I)
RATIO = (0.8, 1.25)
ATTEMPTS = 3                                         # tries per pair; each retry is told what the checks rejected
TELL_CHECK = ("You check answers for preference training. Below are a question, a correct answer, and a WRONG answer written to be a hard "
              "negative. Does the wrong answer reveal that it is wrong by anything other than its content: for example it describes its own "
              "mistake as a deliberate step (\"I shifted each item one place\", \"I added one\"), admits or hints at an error, comments on its "
              "method, or reads as if written by someone who knows the conclusion is wrong? Answer true only for such a giveaway; a plain wrong "
              "claim stated confidently is false.\n\nQUESTION\n{q}\n\nCORRECT ANSWER\n{p}\n\nWRONG ANSWER\n{w}\n\n"
              "Return only JSON: {{\"reveals_mistake\": true or false}}")
FORMAT_RULE = {
    "numerical": "FORMAT: this is a worked-solution answer with no <think> block. Return only its text.",
    "document": "FORMAT: this answer has a short reasoning part and a final answer part. Write the reasoning as confident reasoning from the document "
                "toward the wrong conclusion (no admission of any mistake), about as long as the preferred reasoning.",
}
OUTPUT_SHAPE = {"numerical": '{"answer": "<the new wrong answer, same layout as the preferred answer>"}',
                "document": '{"think": "<the reasoning, confident, no admission>", "answer": "<the final answer>"}'}


# ---- the sample ------------------------------------------------------------------------------------------

def _split_user(text: str, kind: str) -> tuple[str, str]:
    """(document, question) of a pair's user message. Numerical pairs have no document."""
    if kind == "numerical":
        return "", text.strip()
    doc, _, question = text.rpartition("\n\n")
    return doc.strip(), question.strip()


def make_sample(pairs_path: Path, out_path: Path, per_source: int = 10, seed: int = 5) -> int:
    rows = [normalize_row(json.loads(l)) for l in open(pairs_path, encoding="utf-8") if l.strip()]
    rng, pools = random.Random(seed), {}
    for r in rows:
        if r.get("defect_type") == "padded" or r["not_preferred_score"] >= 7:        # padded pairs are correct on purpose: no mistake to rewrite
            continue
        kind = "document" if r["origin"].startswith("full-") else "numerical"
        m = re.match(r"^([A-Z]{2})-\d", str(r.get("chunk_id")))
        domain = r["source"] if kind == "document" else SHEET_OF_PREFIX.get(m.group(1) if m else "", r["source"])
        pools.setdefault((kind, domain), []).append(r)
    out = []
    for (kind, domain), pool in sorted(pools.items()):
        for r in rng.sample(pool, min(per_source, len(pool))):
            user = next(m["content"] for m in reversed(r["prompt"]) if m["role"] == "user")
            doc, question = _split_user(user, kind)
            out.append({"id": len(out), "kind": kind, "domain": domain, "pair_id": r["pair_id"], "document": doc, "question": question,
                        "preferred": r["preferred"].strip(), "original": r["not_preferred"].strip(),
                        "preferred_score": r["preferred_score"], "original_score": r["not_preferred_score"]})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out), encoding="utf-8")
    return len(out)


# ---- one rewrite -----------------------------------------------------------------------------------------

def _answer_part(text: str) -> str:
    return text.split("</think>")[-1].strip()


def _final_line(text: str) -> str:
    lines = [l.strip() for l in _answer_part(text).splitlines() if l.strip()]
    hit = [l for l in lines if l.startswith("Answer:")]
    return (hit[-1] if hit else (lines[-1] if lines else "")).lower()


def target_length(rec: dict) -> str:
    """The word budget the rewrite must hit, taken from the preferred answer."""
    words = lambda t: len(t.split())  # noqa: E731
    pref = rec["preferred"]
    if rec["kind"] == "document" and "</think>" in pref:
        think, answer = pref.split("</think>", 1)
        t, a = words(think.replace("<think>", "")), words(answer)
        return (f"The reasoning should be about {t} words (between {int(t * 0.9)} and {int(t * 1.15)}) and the final answer about {a} words "
                f"(between {int(a * 0.9)} and {int(a * 1.15)}).")
    n = words(pref)
    return f"The whole answer should be about {n} words (between {int(n * 0.9)} and {int(n * 1.15)})."


def build_prompt(template: str, rec: dict, feedback: str = "") -> str:
    prompt = (template.replace("{{format_rule}}", FORMAT_RULE[rec["kind"]]).replace("{{question}}", rec["question"])
              .replace("{{target_length}}", target_length(rec))
              .replace("{{preferred}}", rec["preferred"]).replace("{{original}}", rec["original"])
              .replace("{{output_shape}}", OUTPUT_SHAPE[rec["kind"]]))
    return prompt + (f"\n\nYOUR PREVIOUS ATTEMPT WAS REJECTED: {feedback}\nWrite a new answer that fixes this and keeps every other rule." if feedback else "")


def feedback_for(rec: dict, new: str, checks: dict) -> str:
    """What was wrong with an attempt, in words the rewriter can act on (empty if the code checks pass)."""
    why = []
    if not checks["length_ok"]:
        why.append(f"its length was {checks['length_ratio']}x the preferred answer's; keep it between {RATIO[0]}x and {RATIO[1]}x")
    if not checks["no_admission"]:
        why.append("it contained a word that admits or describes a mistake (" + ", ".join(sorted({m.group(0) for m in ADMISSIONS.finditer(new)})) + ")")
    if not checks["same_lines"]:
        why.append("it did not have the same number of lines as the preferred answer")
    if not checks["different_final"]:
        why.append("its final answer was the same as the preferred answer's: it must reach the original wrong result")
    return "; ".join(why)


def parse_rewrite(text: str, kind: str) -> str | None:
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        obj = json.loads(text[text.index("{"):text.rindex("}") + 1])
    except ValueError:
        return None
    answer = obj.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return None
    if kind == "document":
        think = obj.get("think")
        return f"<think>{think.strip()}</think>\n\n{answer.strip()}" if isinstance(think, str) and think.strip() else None
    return answer.strip()


def code_checks(rec: dict, new: str) -> dict:
    words = lambda t: len(t.split())  # noqa: E731
    ratio = words(new) / max(1, words(rec["preferred"]))
    lines = lambda t: len([l for l in _answer_part(t).splitlines() if l.strip()])  # noqa: E731
    return {"no_admission": not ADMISSIONS.search(new), "length_ratio": round(ratio, 2), "length_ok": RATIO[0] <= ratio <= RATIO[1],
            "same_lines": rec["kind"] == "document" or abs(lines(new) - lines(rec["preferred"])) <= 1,
            "different_final": _final_line(new) != _final_line(rec["preferred"])}


def rewrite_all(records: list[dict], llm: LLM, workers: int) -> list[dict]:
    template = PROMPT.read_text(encoding="utf-8")
    out = [None] * len(records)

    def work(i):
        rec = records[i]
        new, feedback, tries = None, "", 0
        for tries in range(1, ATTEMPTS + 1):
            attempt = None
            try:
                raw = llm.chat(build_prompt(template, rec, feedback), system="Respond with only the JSON object specified. No other text.",
                               temperature=0.4, max_tokens=4000, effort="medium")["content"]
                attempt = parse_rewrite(raw, rec["kind"])
            except Exception as e:  # noqa: BLE001 - a failed call is recorded, not fatal
                feedback = f"the reply could not be used ({type(e).__name__}); return only the JSON object"
            if attempt:
                new = attempt                                     # the latest usable attempt is kept, even if a check fails
                feedback = feedback_for(rec, new, code_checks(rec, new))
                if not feedback:
                    break
            elif not feedback:
                feedback = "the reply was not valid JSON in the required shape"
        out[i] = {**rec, "rewrite": new, "checks": code_checks(rec, new) if new else None, "attempts": tries}
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(work, range(len(records))))
    return out


def tell_check_all(results: list[dict], llm: LLM, workers: int = 32) -> None:
    """Ask the model whether each rewrite gives itself away (adds `reveals_mistake`: True / False / None if unanswered)."""
    def one(r):
        r["reveals_mistake"] = None
        if not r["rewrite"]:
            return
        try:
            text = llm.chat(TELL_CHECK.format(q=r["question"], p=r["preferred"], w=r["rewrite"]), system="Respond with only the JSON object.",
                            temperature=0.0, max_tokens=800, effort="medium")["content"]
            r["reveals_mistake"] = bool(json.loads(text[text.index("{"):text.rindex("}") + 1]).get("reveals_mistake"))
        except Exception:  # noqa: BLE001 - unanswered stays None and is reported
            pass
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, results))


def judge_all(results: list[dict], llm: LLM) -> None:
    """Add `judge_preferred` and `judge_rewrite` to every result that has a rewrite (preferred and rewrite are judged blind, sides random)."""
    rng = random.Random(11)
    for kind, name in (("numerical", "numerical"), ("document", "document")):
        template = judge.load_template(config.JUDGE_PROMPT_NUMERICAL if kind == "numerical" else config.JUDGE_PROMPT)
        todo = [r for r in results if r["kind"] == kind and r["rewrite"]]
        for start in range(0, len(todo), config.JUDGE_BATCH):
            chunk, items, flips = todo[start:start + config.JUDGE_BATCH], [], []
            for pos, r in enumerate(chunk):
                flip = rng.random() < 0.5
                a, b = (_answer_part(r["rewrite"]), _answer_part(r["preferred"])) if flip else (_answer_part(r["preferred"]), _answer_part(r["rewrite"]))
                item = {"id": pos, "question": r["question"], "answer_a": a, "answer_b": b}
                item.update({"reference_solution": _answer_part(r["preferred"])} if kind == "numerical" else {"document": r["document"]})
                items.append(item)
                flips.append(flip)
            verdicts = {v["id"]: v for v in judge.judge_batch(items, llm, template, config.JUDGE_REASONING_EFFORT)}
            for pos, r in enumerate(chunk):
                v = verdicts.get(pos)
                if v:
                    pref, new = (v["b"], v["a"]) if flips[pos] else (v["a"], v["b"])
                    r["judge_preferred"], r["judge_rewrite"] = pref, new


def decide(r: dict) -> None:
    c, jp, jr = r["checks"], r.get("judge_preferred"), r.get("judge_rewrite")
    why = []
    if not r["rewrite"]:
        why.append("no valid rewrite")
    else:
        why += [k for k in ("no_admission", "length_ok", "same_lines", "different_final") if not c[k]]
        if r.get("reveals_mistake"):
            why.append("reveals the mistake (describes it or hints at it)")
        if not jp or not jr:
            why.append("no judge verdict")
        else:
            if jp["score"] < 7:
                why.append("judge fails the preferred answer")
            if jr["score"] >= 7:
                why.append("judge passes the rewrite (it is no longer wrong)")
    r["accepted"], r["rejected_because"] = not why, why


def summarize(results: list[dict]) -> dict:
    words = lambda t: len(t.split())  # noqa: E731
    ok = [r for r in results if r["rewrite"]]
    before = [words(r["original"]) / max(1, words(r["preferred"])) for r in results]
    after = [words(r["rewrite"]) / max(1, words(r["preferred"])) for r in ok]
    by = {}
    for r in results:
        d = by.setdefault(r["domain"], {"n": 0, "accepted": 0})
        d["n"] += 1
        d["accepted"] += r["accepted"]
    reasons = {}
    for r in results:
        for w in r["rejected_because"]:
            reasons[w] = reasons.get(w, 0) + 1
    return {"pairs": len(results), "rewritten": len(ok), "accepted": sum(r["accepted"] for r in results),
            "rejected_because": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
            "length_ratio_to_preferred": {"original_median": round(st.median(before), 2), "rewrite_median": round(st.median(after), 2) if after else None},
            "attempts_used": {k: sum(r.get("attempts") == k for r in results) for k in (1, 2, 3)},
            "reveals_mistake": {"yes": sum(r.get("reveals_mistake") is True for r in ok), "unanswered": sum(r.get("reveals_mistake") is None for r in ok)},
            "admission_words": {"original": sum(bool(ADMISSIONS.search(r["original"])) for r in results), "rewrite": sum(bool(ADMISSIONS.search(r["rewrite"])) for r in ok)},
            "by_domain": by}


# ---- commands --------------------------------------------------------------------------------------------

def cmd_run(a) -> int:
    from pipeline import server_is_up, start_server, wait_ready
    work = config.HERE / "work" / a.run
    work.mkdir(parents=True, exist_ok=True)
    s3 = s3util.client()
    s3util.download(s3, a.sample or SAMPLE_KEY, work / "sample.jsonl")
    records = read_jsonl(work / "sample.jsonl")
    url, server = config.SERVER_URL, None
    try:
        if not server_is_up(url):
            server = start_server(work / "server.log", f"{config.VLLM_BASE_ARGS} --max-num-seqs 64")
        wait_ready(url, server)
        llm = LLM(url, config.SERVED_NAME, 32)
        results = rewrite_all(records, llm, 32)
        judge_all(results, llm)
        tell_check_all(results, llm)
    finally:
        if server is not None:
            server.terminate()
    for r in results:
        decide(r)
    report = summarize(results)
    (work / "results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8")
    (work / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    s3util.upload(s3, work / "results.jsonl", f"{config.OUT_ROOT}{a.run}/final/results.jsonl")
    s3util.upload(s3, work / "report.json", f"{config.OUT_ROOT}{a.run}/final/report.json")
    print(json.dumps(report, indent=2), flush=True)
    print(f"[rewrite] done: {report['accepted']} of {report['pairs']} rewrites accepted", flush=True)
    return 0


def cmd_readable(a) -> int:
    rows = read_jsonl(Path(a.results))
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"n": r["id"] + 1, "set": r["kind"], "domain": r["domain"], "accepted": r["accepted"], "rejected_because": r["rejected_because"],
                                "question": r["question"], "preferred_answer": r["preferred"], "original_not_preferred": r["original"],
                                "rewritten_not_preferred": r["rewrite"], "length_ratio_original": round(len(r["original"].split()) / max(1, len(r["preferred"].split())), 2),
                                "length_ratio_rewrite": (r["checks"] or {}).get("length_ratio"),
                                "judge_preferred": (r.get("judge_preferred") or {}).get("score"), "judge_rewrite": (r.get("judge_rewrite") or {}).get("score")},
                               ensure_ascii=False) + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-sample")
    m.add_argument("--pairs", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--per-source", type=int, default=10)
    r = sub.add_parser("run")
    r.add_argument("--run", required=True)
    r.add_argument("--sample", help="S3 key of the sample (default: rewrite-input/sample.jsonl)")
    r.add_argument("--resume", action="store_true", help="accepted for the launcher")
    d = sub.add_parser("readable")
    d.add_argument("--results", required=True)
    d.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "make-sample":
        print(make_sample(Path(a.pairs), Path(a.out), a.per_source), "pairs in the sample")
        return 0
    return cmd_run(a) if a.cmd == "run" else cmd_readable(a)


if __name__ == "__main__":
    sys.exit(main())
