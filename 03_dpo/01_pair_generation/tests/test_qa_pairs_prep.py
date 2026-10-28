"""Tests for the DPO QA-pair pipeline: corpus filters and chunker, assignment balance, reply parsing, judge validation,
trainer compatibility (real Qwen3.5 tokenizer), and the whole generate -> judge -> select -> build path against a fake
OpenAI-compatible server (no GPU, no S3).

    cd DPO_Training/QA_pairs_prep && python -m pytest tests -q
"""
import json
import re
import sys
import threading
from argparse import Namespace
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
SFT = HERE.parent.parent / "SFT_Script"

import assign  # noqa: E402
import build_dpo_pairs  # noqa: E402
import agreement  # noqa: E402
import capacity  # noqa: E402
import config  # noqa: E402
import generate  # noqa: E402
import judge  # noqa: E402
import pipeline  # noqa: E402
import prep_corpus as prep  # noqa: E402
import score_pairs  # noqa: E402

WORDS = lambda n, w="word": " ".join([w] * n)  # noqa: E731


# ---- corpus ------------------------------------------------------------------------------------------

def test_front_matter_is_split_off_and_unescaped():
    text = '---\nurl: https://x.com/a\ntitle: "Credit \\u2013 Basics: Part 1"\nfinance_density: 91.3\nwords: 931\n---\n\n# Body\ntext'
    meta, body = prep.parse_front_matter(text)
    assert meta["title"] == "Credit – Basics: Part 1" and meta["finance_density"] == "91.3" and body.strip().startswith("# Body")
    assert prep.parse_front_matter("no front matter") == ({}, "no front matter")


@pytest.mark.parametrize("url,title,keep", [
    ("https://www.cleartax.in/s/credit-score", "What is Credit Score?", True),               # "tax" is in the domain only
    ("https://www.cibil.com/faq-loan-rejections-disputes", "FAQs – Loan Rejections and Disputes | CIBIL", True),
    ("https://www.cleartax.in/s/capital-gains-income", "What is Capital Gains Tax In India", True),    # kept on purpose
    ("https://www.cleartax.in/mutual-funds/sbi-elss", "SBI ELSS Tax Saver Fund", True),
    ("https://www.cibil.com/choose-subscription", "Choose Your CIBIL Subscription | CIBIL", False),
])
def test_cibil_rule_uses_title_and_path_not_the_domain(url, title, keep):
    ok, _ = prep.keep_document(config.CIBIL_FOLDER, {"url": url, "title": title}, WORDS(400))
    assert ok is keep


def test_a_long_page_with_a_navigation_word_is_kept_but_a_short_one_is_not():
    meta = {"url": "https://www.cibil.com/contact-us-faq", "title": "Contact Us FAQ's | CIBIL"}
    assert prep.keep_document(config.CIBIL_FOLDER, meta, WORDS(6000))[0]
    assert prep.keep_document(config.CIBIL_FOLDER, meta, WORDS(900)) == (False, "cibil: short navigation or marketing page")
    tax = {"url": "https://www.cleartax.in/s/capital-gains-income", "title": "Capital Gains Tax"}
    assert prep.keep_document(config.CIBIL_FOLDER, tax, WORDS(6000))[0] is True        # tax pages are kept on purpose


def test_other_folders_use_finance_density_and_length():
    f = "Private Markets Terminology"
    assert prep.keep_document(f, {"finance_density": "80"}, WORDS(400))[0]
    assert prep.keep_document(f, {"finance_density": "20"}, WORDS(400)) == (False, "low finance_density")
    assert prep.keep_document(f, {"finance_density": "80"}, WORDS(100)) == (False, "too short")
    assert prep.keep_document(f, {}, WORDS(400))[0]                                  # no score: kept, not dropped


def test_chunks_cover_the_text_overlap_and_respect_the_size():
    paras = [f"Paragraph {i}. " + "alpha beta gamma delta. " * 20 for i in range(40)]
    chunks = prep.chunk_text("\n\n".join(paras), size=3000, overlap=300, min_tail=800)
    assert len(chunks) > 3 and all(len(c) <= 3000 + 800 for c in chunks)
    for p in paras:
        assert any(p[:40] in c for c in chunks), "a paragraph is missing from every chunk"
    for a, b in zip(chunks, chunks[1:]):
        assert b[:60].strip() and b[:60].split()[1] in a[-400:], "no overlap between neighbouring chunks"


def test_one_huge_paragraph_is_split_without_loss():
    text = ("Sentence number one is here. " * 1000).strip()
    chunks = prep.chunk_text(text, size=2000, overlap=0, min_tail=500)
    assert all(len(c) <= 2000 + 500 for c in chunks) and sum(len(c) for c in chunks) >= len(text) * 0.98


def test_near_duplicates_keep_the_longest_copy():
    a = {"key": "a", "words": 600, "body": "the quick brown fox jumps over the lazy dog near the river bank " * 60}
    b = {"key": "b", "words": 590, "body": a["body"][:-30]}
    c = {"key": "c", "words": 600, "body": "completely different text about waterfalls and carried interest " * 60}
    kept, dups = prep.dedupe([a, b, c])
    assert {d["key"] for d in kept} == {"a", "c"} and dups[0]["key"] == "b" and dups[0]["duplicate_of"] == "a"


# ---- assignment --------------------------------------------------------------------------------------

def make_chunks(n, quantities=12):
    return [{"chunk_id": f"c{i:04d}", "folder": "F", "n_quantities": quantities} for i in range(n)]


def test_assignment_hits_target_shares_and_rules():
    calls = assign.assign(make_chunks(400))
    pairs = [(t, d) for c in calls for t, d in zip(c["types"], c["defects"])]
    share = Counter(t for t, _ in pairs)
    total_w = sum(config.TYPE_MIX.values())
    for t, w in config.TYPE_MIX.items():
        assert abs(share[t] / len(pairs) - w / total_w) < 0.02, t
    assert all(d in score_pairs.COMPATIBLE[t] for t, d in pairs)
    defect_share = Counter(d for _, d in pairs)
    assert max(defect_share.values()) / len(pairs) <= config.MAX_DEFECT_SHARE, defect_share
    assert abs(defect_share["padded"] / len(pairs) - config.DEFECT_TARGETS["padded"]) < 0.015
    for c in calls:
        assert len(c["types"]) <= score_pairs.MAX_PAIRS_PER_CALL and len(set(c["types"])) == len(c["types"])
        if set(c["types"]) & score_pairs.ONE_PAIR_PER_CALL:
            assert len(c["types"]) == 1
    per_chunk = Counter()
    for c in calls:
        per_chunk.update((c["chunk_id"], t) for t in c["types"])
    assert max(per_chunk.values()) == 1, "a type was repeated for the same chunk"


def test_an_unreachable_mix_is_refused():
    with pytest.raises(ValueError, match="cannot exceed"):
        assign.assign(make_chunks(10), type_mix={"Terminology": 0.5, "Short Answer": 0.3, "Comparative Ranking": 0.2})


def test_a_defect_target_above_the_cap_is_refused(monkeypatch):
    monkeypatch.setitem(config.DEFECT_TARGETS, "unsupported_claim", 0.30)
    with pytest.raises(ValueError, match="MAX_DEFECT_SHARE"):
        assign.assign(make_chunks(10))


def test_numerical_types_only_go_to_chunks_with_figures():
    calls = assign.assign(make_chunks(100, quantities=0))
    assert not {t for c in calls for t in c["types"]} & config.NUMERIC_TYPES_NEED_FIGURES


def test_sample_is_seeded_and_stable():
    chunks = make_chunks(500)
    assert assign.sample_chunks(chunks, 20) == assign.sample_chunks(list(reversed(chunks)), 20)


def test_smoke_sample_covers_every_folder_and_includes_figure_chunks():
    folders = ["A", "B", "C"]
    chunks = [{"chunk_id": f"c{i:04d}", "folder": folders[i % 3], "n_quantities": 12 if i % 7 == 0 else 0} for i in range(300)]
    sample = assign.sample_chunks(chunks, 20)
    assert len(sample) == len({c["chunk_id"] for c in sample}) == 20
    assert {c["folder"] for c in sample} == set(folders)
    assert sum(c["n_quantities"] >= config.MIN_QUANTITIES_FOR_NUMERIC for c in sample) >= 2


# ---- replies -----------------------------------------------------------------------------------------

def good(t="Terminology", d="unsupported_claim"):
    return {"question_type": t, "defect_type": d, "question": "Q?", "preferred_thinking": "<think>a</think>", "preferred": "A",
            "not_preferred_thinking": "<think>b</think>", "not_preferred": "B", "numeric_check": {}}


def test_reply_parsing_handles_fences_empty_arrays_and_truncation():
    objs = [good(), good("Short Answer", "padded")]
    assert list(generate.iter_json_objects(json.dumps(objs))) == objs
    assert list(generate.iter_json_objects("```json\n" + json.dumps(objs) + "\n```")) == objs
    assert list(generate.iter_json_objects("[]")) == [] and list(generate.iter_json_objects("sorry")) == []
    cut = json.dumps(objs)[:-40]                                                      # second object cut off mid-way
    assert list(generate.iter_json_objects(cut)) == objs[:1]


def test_pair_validation():
    assigned = {"Terminology": "unsupported_claim"}
    assert generate.validate_pair(good(), assigned, set()) is None
    assert "not assigned" in generate.validate_pair(good("Short Answer", "padded"), assigned, set())
    assert "differs" in generate.validate_pair(good(d="padded"), assigned, set())
    assert "repeated" in generate.validate_pair(good(), assigned, {"Terminology"})
    assert "missing" in generate.validate_pair({"question": "x"}, assigned, set())


def test_missing_think_tags_are_repaired():
    assert generate.wrap_think("plain reasoning") == "<think>plain reasoning</think>"
    assert generate.wrap_think("<think>a</think>") == "<think>a</think>"
    assert generate.wrap_think("<think>a") == "<think>a</think>" and generate.wrap_think("a</think>") == "<think>a</think>"


def test_prompt_loads_and_fills():
    p = generate.load_prompt().format(chunk_text="{braces} in the document", question_types="Terminology", defect_types="padded")
    assert "{braces} in the document" in p and "Question types, in order: Terminology" in p and "{{" not in p


def test_judge_verdict_validation():
    ok = {"score": 9, "reason": "PASS_ALL_GATES", "failed_gates": []}
    bad = {"score": 2, "reason": "F06_NUMERICAL_ERROR", "failed_gates": ["G3", "G8"]}
    assert judge.valid_verdicts(2, [{"id": 0, "a": ok, "b": bad}, {"id": 1, "a": bad, "b": ok}])
    assert not judge.valid_verdicts(2, [{"id": 0, "a": ok, "b": bad}])                                # one missing
    assert not judge.valid_verdicts(1, [{"id": 0, "a": ok, "b": {**bad, "failed_gates": []}}])        # score and gates disagree
    assert not judge.valid_verdicts(1, [{"id": 0, "a": {**ok, "reason": "F06_NUMERICAL_ERROR"}, "b": bad}])
    assert not judge.valid_verdicts(1, [{"id": 0, "a": ok, "b": {**bad, "failed_gates": ["G9"]}}])


# ---- trainer compatibility ---------------------------------------------------------------------------

def test_pair_is_accepted_by_the_dpo_trainers_own_encoder():
    pytest.importorskip("torch")
    tdir = SFT / "models" / "pretuned"
    if not (tdir / "tokenizer.json").exists():
        pytest.skip("pre-tuned tokenizer files not downloaded")
    sys.path.insert(0, str(SFT))
    from transformers import AutoTokenizer
    from sft_common.chat_format import ChatFormat
    from sft_common.dpo_data import encode_pair, side_tokens
    tok = AutoTokenizer.from_pretrained(tdir)
    sentence = "The senior facility carries a margin of 3.25% over the benchmark and amortises in equal instalments. "
    row = {"document": ("Source: Senior debt\n\n" + sentence * 60)[:config.CHUNK_CHARS], "question": "What is the margin?",
           "preferred_thinking": "<think>" + WORDS(150, "think") + "</think>", "preferred": "The margin is 3.25%. " + WORDS(150),
           "not_preferred_thinking": "<think>" + WORDS(150, "idea") + "</think>", "not_preferred": "The margin is 2.25%. " + WORDS(150),
           "folder": "Private Markets Terminology", "question_type": "Short Answer", "defect_type": "calculation_slip",
           "chunk_id": "c1", "call_id": "c1-p0-0", "score": 0.7}
    pair = build_dpo_pairs.to_dpo_pair(row, (HERE / "prompts" / "system_prompt.txt").read_text(encoding="utf-8"))
    pair.update(source=pair["source"], line=0)
    enc = encode_pair(ChatFormat(tok), pair)
    assert isinstance(enc, dict), enc
    assert tok.decode(enc["chosen_ids"]).endswith("<|im_end|>")
    prompt_text = tok.decode(enc["prompt_ids"])
    assert "VaiaGPT" in prompt_text and "What is the margin?" in prompt_text and prompt_text.endswith("<|im_start|>assistant\n")
    assert side_tokens(enc) < 6000, "a typical pair must fit dpo.max_pair_tokens (6000)"
    print(f"\n[budget] system {len(tok(pair['prompt'][0]['content'])['input_ids'])} tokens, whole side {side_tokens(enc)} tokens")


# ---- the whole path against a fake server --------------------------------------------------------------

def fake_generate(prompt: str) -> str:
    types = re.search(r"Question types, in order: (.*)", prompt).group(1).split(", ")
    defects = re.search(r"Defect for each not-preferred answer, in the same order: (.*)", prompt).group(1).split(", ")
    out = []
    for t, d in zip(types, defects):
        n_rej = {"no_workings": 60, "omitted_element": 60, "single_driver": 60, "padded": 150}.get(d, 100)
        numeric = {"expression": "1+1", "final_value": 2, "unit": "m", "not_preferred_value": 3} if t in score_pairs.NUMERIC_TYPES else {}
        out.append({"question_type": t, "defect_type": d, "question": f"What does the document say about {t}?",
                    "preferred_thinking": "<think>" + WORDS(100, "think") + "</think>", "preferred": "CHOSEN_ANSWER " + WORDS(99, "c"),
                    "not_preferred_thinking": "<think>" + WORDS(100, "idea") + "</think>",
                    "not_preferred": f"REJECTED_ANSWER defect={d} " + WORDS(n_rej - 2, "r"), "numeric_check": numeric})
    text = json.dumps(out)
    return f"```json\n{text}\n```" if hash(prompt) % 5 == 0 else text


def fake_judge(prompt: str) -> str:
    def verdict(answer: str) -> dict:
        if answer.startswith("CHOSEN_ANSWER"):
            return {"score": 9, "reason": "PASS_ALL_GATES", "failed_gates": []}
        defect = re.search(r"defect=(\w+)", answer).group(1)
        return {"score": 2, "reason": "F06_NUMERICAL_ERROR", "failed_gates": sorted(score_pairs.EXPECTED_GATES[defect]) + ["G8"]}
    rows = []
    for line in prompt.splitlines():
        try:                                       # the prompt's own contract lines look like records but are not JSON
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and "answer_a" in row:
            rows.append(row)
    return json.dumps({"scores": [{"id": r["id"], "a": verdict(r["answer_a"]), "b": verdict(r["answer_b"])} for r in rows]})


class FakeHandler(BaseHTTPRequestHandler):
    def do_GET(self):                                     # the readiness probe: vLLM answers 200 on /v1/models once loaded
        data = json.dumps({"data": [{"id": config.SERVED_NAME}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["messages"][-1]["content"]
        assert body["reasoning_effort"] in ("low", "medium", "high")
        try:
            content = fake_judge(prompt) if "VAIA DPO PAIR JUDGE" in prompt else fake_generate(prompt)
        except Exception as e:  # noqa: BLE001 - answer with a 500 so a broken fake fails the test fast instead of hanging it
            self.send_error(500, str(e)[:100])
            return
        data = json.dumps({"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                           "usage": {"prompt_tokens": 10, "completion_tokens": 20}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture()
def fake_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_end_to_end_with_resume(tmp_path, fake_server):
    chunks = [{"chunk_id": f"c{i:03d}", "source_key": f"k{i}", "folder": "Credit Rating Replication", "title": "T", "url": "",
               "chunk_index": 0, "n_chunks": 1, "n_quantities": 12, "text": f"Source: T\n\nDocument {i}. " + "text " * 80}
              for i in range(12)]
    chunks_path, assignments = tmp_path / "chunks.jsonl", tmp_path / "assignments.jsonl"
    chunks_path.write_text("\n".join(map(json.dumps, chunks)))
    calls = assign.assign(chunks)
    assignments.write_text("\n".join(map(json.dumps, calls)))

    stats = generate.generate(chunks_path, assignments, tmp_path, workers=4, url=fake_server)
    asked = sum(len(c["types"]) for c in calls)
    assert stats["calls"] == len(calls) and stats["pairs"] == asked and stats["errors"] == 0
    raw = generate.read_jsonl(tmp_path / "pairs_raw.jsonl")
    assert len(raw) == asked and all(r["document"].startswith("Source: T") for r in raw)
    assert (tmp_path / "first_response.json").exists()
    assert generate.generate(chunks_path, assignments, tmp_path, workers=4, url=fake_server)["calls"] == 0     # resume: nothing left

    score_pairs.cmd_build(Namespace(pairs=str(tmp_path / "pairs_raw.jsonl"), out=str(tmp_path / "judge_in.jsonl"), key=str(tmp_path / "key.json")))
    n_judged = judge.run(tmp_path / "judge_in.jsonl", tmp_path / "judge_out.jsonl", url=fake_server, workers=3)
    assert n_judged == len(raw)
    score_pairs.cmd_select(Namespace(pairs=str(tmp_path / "pairs_raw.jsonl"), judge=str(tmp_path / "judge_out.jsonl"),
                                     key=str(tmp_path / "key.json"), out=str(tmp_path / "scored_pairs.jsonl")))
    report = build_dpo_pairs.build(tmp_path / "scored_pairs.jsonl", tmp_path / "dpo_pairs.jsonl", tmp_path / "report.json")
    final = generate.read_jsonl(tmp_path / "dpo_pairs.jsonl")
    assert len(final) == report["pairs"] == len(raw), "every fake pair is valid, so every pair must survive"
    p = final[0]
    assert [m["role"] for m in p["prompt"]] == ["system", "user"] and p["prompt"][1]["content"].startswith("Source: T")
    assert p["preferred"].startswith("<think>") and p["not_preferred"].startswith("<think>") and p["source"] == "credit_rating"


def test_nothing_is_dropped_bad_pairs_are_kept_with_flags(tmp_path):
    ok = dict(document="DOC", question="Q?", question_type="Terminology", defect_type="unsupported_claim", preferred_thinking="<think>" + WORDS(40) + "</think>",
              not_preferred_thinking="<think>" + WORDS(40, "r") + "</think>", preferred="CHOSEN_ANSWER " + WORDS(30, "c"), not_preferred="REJECTED_ANSWER defect=unsupported_claim " + WORDS(28, "r"),
              numeric_check={}, chunk_id="c", call_id="c-p0-0", folder="Credit Rating Replication", source_key="k")
    bad_think = {**ok, "preferred_thinking": "no tags at all", "preferred": "x"}                       # several code flags at once
    incomplete = {k: v for k, v in ok.items() if k != "not_preferred"}                              # cannot even be judged
    assert len(score_pairs.code_flags(bad_think)) >= 3 and score_pairs.code_flags(incomplete) == ["missing field"]
    pairs = tmp_path / "p.jsonl"
    pairs.write_text("\n".join(map(json.dumps, [ok, bad_think, incomplete])))
    score_pairs.cmd_build(Namespace(pairs=str(pairs), out=str(tmp_path / "in.jsonl"), key=str(tmp_path / "k.json")))
    lines = generate.read_jsonl(tmp_path / "in.jsonl")
    assert len(lines) == 2                                                                      # the incomplete pair is not sent to the judge...
    good, weak = {"score": 9, "reason": "PASS_ALL_GATES", "failed_gates": []}, {"score": 2, "reason": "F04_UNSUPPORTED_CLAIM", "failed_gates": ["G2", "G8"]}
    key = json.loads((tmp_path / "k.json").read_text())["key"]
    verdicts = [{"id": 0, "a": good, "b": weak}, {"id": 1, "a": weak, "b": weak}]
    for v, k in zip(verdicts, key):                                                             # make pair 0 clean, pair 1 a bad chosen answer
        if k["preferred_side"] == "b" and v["id"] == 0:
            v["a"], v["b"] = weak, good
    (tmp_path / "j.jsonl").write_text("\n".join(map(json.dumps, verdicts)))
    score_pairs.cmd_select(Namespace(pairs=str(pairs), judge=str(tmp_path / "j.jsonl"), key=str(tmp_path / "k.json"), out=str(tmp_path / "s.jsonl")))
    out = generate.read_jsonl(tmp_path / "s.jsonl")
    assert len(out) == 3                                                                        # ...but it is kept, and so is everything else
    assert any(r["passes_all_gates"] for r in out) and sum(not r["passes_all_gates"] for r in out) == 2
    assert not any(k in r for r in out for k in ("score", "score_parts", "rank"))              # only the judge's own verdicts are stored
    assert any(r["flags"] == ["missing field", "no judge verdict"] for r in out)


# ---- capacity ----------------------------------------------------------------------------------------

def test_model_numbers_match_the_published_ones():
    assert abs(capacity.WEIGHT_BYTES / 1e9 - 65) < 3                      # ~65 GB checkpoint
    assert abs(capacity.ACTIVE_PARAMS / 1e9 - 5.1) < 0.2                  # "5.1B active parameters"
    assert abs((capacity.EXPERT_PARAMS + capacity.ATTN_PARAMS + 2 * capacity.EMBED_PARAMS) / 1e9 - 117) < 2


def test_h100_fits_few_sequences_and_h200_is_much_faster():
    assert 20 <= capacity.max_seqs("H100", 7000) <= 40                    # vLLM reported 203,612 KV tokens on an 80 GB H100
    assert capacity.max_seqs("H200", 7000) > 150
    slow = capacity.estimate("H100", 2900, 9.1e6, 6000, 650, 14e6)["total_h"]
    fast = capacity.estimate("H200", 2900, 9.1e6, 6000, 650, 14e6)["total_h"]
    assert slow > 2 * fast and fast < 4


def test_decode_is_memory_bound_so_mfu_is_small_and_batching_helps():
    tps = {b: capacity.decode_tps("H200", b, 7000) for b in (8, 64, 192)}
    assert tps[8] < tps[64] < tps[192]
    assert capacity.mfu("H200", tps[192]) < 0.10
    assert abs(capacity.implied_mbu("H200", 192, 7000, tps[192]) - 0.5) < 0.01       # the inverse recovers the assumed MBU


def test_plan_without_a_gpu_changes_nothing(monkeypatch):
    monkeypatch.setattr(capacity, "detect_gpu", lambda: None)
    out = pipeline.plan([{"chunk_id": "c", "types": ["Terminology"]}], {"c": {"text": "x" * 100}})
    assert out == {"gpu": None, "seqs": None, "workers": None, "judge_workers": None}


# ---- second-opinion judge ------------------------------------------------------------------------------

def test_agreement_counts_pass_fail_and_usable_pairs():
    ok = {"score": 9, "reason": "PASS_ALL_GATES", "failed_gates": []}
    bad = {"score": 2, "reason": "F06_NUMERICAL_ERROR", "failed_gates": ["G3", "G8"]}
    first = {0: {"id": 0, "a": ok, "b": bad}, 1: {"id": 1, "a": ok, "b": bad}, 2: {"id": 2, "a": bad, "b": ok}}
    second = {0: {"id": 0, "a": ok, "b": bad}, 1: {"id": 1, "a": ok, "b": ok}, 2: {"id": 2, "a": bad, "b": ok}}
    r = agreement.compare(first, second)
    assert r["pass_fail_agreement"] == round(5 / 6, 3) and r["usable_pair_agreement"] == round(2 / 3, 3) and r["disagreeing_pair_ids"] == [1]


# ---- the entry point ----------------------------------------------------------------------------------

def test_pipeline_main_runs_every_stage_against_a_server(tmp_path, fake_server, monkeypatch):
    chunks = [{"chunk_id": f"c{i:03d}", "source_key": f"k{i}", "folder": "Private Markets Terminology", "title": "T", "url": "",
               "chunk_index": 0, "n_chunks": 1, "n_quantities": 12, "text": f"Source: T\n\nDocument {i}. " + "text " * 80} for i in range(8)]
    corpus = tmp_path / "work" / "corpus"
    corpus.mkdir(parents=True)
    (corpus / "chunks.jsonl").write_text("\n".join(map(json.dumps, chunks)))
    monkeypatch.setattr(config, "HERE", tmp_path)
    monkeypatch.setattr(sys, "argv", ["pipeline.py", "--run", "t", "--stage", "smoke", "--limit-chunks", "5", "--local",
                                      "--server-url", fake_server])
    assert pipeline.main() == 0
    out = tmp_path / "work" / "t"
    final = generate.read_jsonl(out / "dpo_pairs.jsonl")
    assert final and json.loads((out / "report.json").read_text())["pairs"] == len(final)
    assert all("flags" in p and "passes_all_gates" in p and p["judge_preferred"] for p in final)
