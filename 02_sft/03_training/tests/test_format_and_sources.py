"""Data preparation and chat format: pool rows used as provided, rendered with the real template."""
import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sft_common import chat_format as cf
from sft_common import sources as src

# A tool-calling row as it appears in the S3 pool (Glaive-style text, kept verbatim).
POOL_TOOL_ROW = {"messages": [
    {"role": "system", "content": 'You are a helpful assistant with access to the following functions. {"name": "f"}'},
    {"role": "user", "content": "Track 2 apples"},
    {"role": "assistant", "content": '<functioncall> {"name": "f", "arguments": \'{"n": 2}\'}'},
    {"role": "user", "content": '{"ok": true}'},
    {"role": "assistant", "content": "Done."},
]}


@pytest.fixture(scope="module")
def fmt():
    """Real Qwen3.5 template from the pre-tuned model's tokenizer (skip if not fetched)."""
    d = ROOT / "models" / "pretuned"
    if not (d / "tokenizer.json").exists():
        pytest.skip("run: python 02_training/fetch_pretuned.py --meta")
    from transformers import AutoTokenizer
    return cf.ChatFormat(AutoTokenizer.from_pretrained(d))


def _qa(i):
    return {"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]}


# ---- sources -------------------------------------------------------------------

def test_pool_row_keeps_text_verbatim():
    msgs = src.pool_row(POOL_TOOL_ROW)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant"]
    assert msgs[2]["content"] == POOL_TOOL_ROW["messages"][2]["content"]  # <functioncall> text untouched


def test_pool_row_moves_inline_think_to_reasoning():
    msgs = src.pool_row({"messages": [{"role": "user", "content": "Q?"},
                                      {"role": "assistant", "content": "<think>\nreason\n</think>\nfinal"}]})
    assert msgs[-1]["reasoning"] == "reason" and msgs[-1]["content"] == "final"


def test_pool_row_trims_trailing_user_and_rejects_unusable():
    msgs = src.pool_row({"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                                      {"role": "user", "content": "c"}]})
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert src.pool_row({"messages": [{"role": "user", "content": "a"}]}) is None  # no answer at all
    assert src.pool_row({"messages": [{"role": "tool", "content": "x"}]}) is None  # unknown role
    assert src.pool_row({"question": "old q/a shape"}) is None


# ---- chat format (real template) -----------------------------------------------

def test_render_pool_row_as_provided(fmt):
    text = fmt.render(src.pool_row(POOL_TOOL_ROW))
    assert "<functioncall>" in text and "<tool_call>" not in text  # trained exactly as written
    assert text.endswith("Done.<|im_end|>\n")


def test_render_keeps_reasoning_on_last_turn(fmt):
    msgs = src.pool_row({"messages": [{"role": "user", "content": "Q?"},
                                      {"role": "assistant", "content": "<think>\nstep\n</think>\nfinal"}]})
    assert "<think>\nstep\n</think>\n\nfinal" in fmt.render(msgs)


def test_domain_messages_with_inline_think_render_once(fmt):
    """Real pool shape: assistant content is '<think>..</think>answer' inside a messages row."""
    row = {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "Q"},
                        {"role": "assistant", "content": "<think>I locate the figures.</think>\nThe answer is 30%."}]}
    text = fmt.render(src.pool_row(row))
    assert text.count("<think>") == 1
    assert text.endswith("<think>\nI locate the figures.\n</think>\n\nThe answer is 30%.<|im_end|>\n")


def test_prompt_thinking_off_and_on(fmt):
    assert fmt.prompt([{"role": "user", "content": "hi"}]).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert fmt.prompt([{"role": "user", "content": "hi"}], thinking=True).endswith("<think>\n")


def test_final_turn_is_trained_exactly_as_it_is_served(fmt):
    """The LAST assistant turn is rendered in training exactly as the model sees it when serving."""
    msgs = src.pool_row(POOL_TOOL_ROW)
    assert fmt.render(msgs).startswith(fmt.prompt(msgs[:-1]))


def test_known_gap_middle_turns_have_no_think_block(fmt):
    """Documents a template property (not a bug in our code): assistant turns BEFORE the last user
    message are rendered without the empty <think></think> block. In the tool data the call is
    such a middle turn, while at serving time the call answers the last user message and does get
    the block. Eval measures calls the serving way, so its tool metrics show whether this matters.
    """
    msgs = src.pool_row(POOL_TOOL_ROW)
    text = fmt.render(msgs)
    assert "<|im_start|>assistant\n<functioncall>" in text                          # trained: no think block
    assert fmt.prompt(msgs[:2]).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")  # served: block


def test_loss_mask_markers_present(fmt):
    text = fmt.render(src.pool_row(_qa(0)))
    assert cf.USER_PART in text and cf.ASSISTANT_PART in text


def test_pretuned_tokens_are_native(fmt):
    """<think> and chat markers are single tokens already: no vocab resize needed."""
    for tokstr in ("<think>", "</think>", "<|im_start|>", "<|im_end|>"):
        assert len(fmt.tok.encode(tokstr, add_special_tokens=False)) == 1


def test_split_think_handles_both_output_shapes():
    assert cf.split_think("<think>\nabc\n</think>\n\nanswer") == ("abc", "answer")
    assert cf.split_think("abc\n</think>\n\nanswer") == ("abc", "answer")   # after a thinking prompt
    assert cf.split_think("plain") == ("", "plain")


def test_generation_config_stops_at_im_end(fmt):
    gen = json.loads((ROOT / "models" / "pretuned" / "generation_config.json").read_text())
    assert fmt.tok.convert_tokens_to_ids("<|im_end|>") in gen["eos_token_id"]


# ---- build_mixture ---------------------------------------------------------------

def _build_mixture():
    spec = importlib.util.spec_from_file_location("bm", ROOT / "01_data_prep" / "build_mixture.py")
    bm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bm)
    return bm


def test_build_mixture_labels_blocks_by_position(fmt):
    bm = _build_mixture()
    blocks = [{"source": "domain", "rows": 5}, {"source": "general", "rows": 3}, {"source": "tool_calling", "rows": None}]
    rows = [_qa(i) for i in range(5)] + [_qa(i) for i in range(5, 7)] + [None] + [_qa(i) for i in range(8, 12)]
    stats = {}
    out = bm.collect(iter(rows), blocks, fmt, stats)
    assert [e["messages"][0]["content"] for e in out["domain"]] == ["q0", "q1", "q2", "q3", "q4"]
    assert stats["general"] == {"kept": 2, "skipped": 1}  # the bad (None) row is counted in its own block
    assert len(out["tool_calling"]) == 4 and "<|im_start|>assistant" in out["tool_calling"][0]["text"]
    train, val = bm.split(out["domain"], 0.2, random.Random(0))
    assert len(val) == 1 and len(train) == 4


def test_build_mixture_block_config_errors(fmt):
    bm = _build_mixture()
    with pytest.raises(ValueError):
        bm.block_ends([{"source": "a", "rows": None}, {"source": "b", "rows": 1}])  # null must be last
    with pytest.raises(ValueError):
        bm.collect(iter([_qa(0), _qa(1)]), [{"source": "a", "rows": 1}], fmt, {})  # file longer than blocks


def test_read_jsonl_keeps_positions_for_bad_lines(tmp_path):
    bm = _build_mixture()
    f = tmp_path / "p.jsonl"
    f.write_text('{"a": 1}\nnot json\n\n{"b": 2}\n', encoding="utf-8")
    assert list(bm.read_jsonl(f)) == [{"a": 1}, None, {"b": 2}]  # blank skipped, bad kept as None


# ---- misc shared helpers ---------------------------------------------------------

def test_s3_classify_full_vs_adapter():
    from sft_common.s3_models import classify
    P = "x/merged_model/"
    objs = [(P + "config.json", 1), (P + "model-00001.safetensors", 9_000_000_000),
            (P + "model-00002.safetensors", 9_000_000_000), (P + "tokenizer.json", 1),
            ("x/ckpt/adapter_config.json", 1), ("x/ckpt/adapter_model.safetensors", 300_000_000)]
    recs = {r["folder"]: r for r in classify(objs)}
    assert recs["x/merged_model"]["kind"] == "full_weights" and recs["x/merged_model"]["has_tokenizer"]
    assert recs["x/ckpt"]["kind"] == "adapter"


def test_env_blank_values_do_not_override(tmp_path, monkeypatch):
    import os
    from sft_common.config import load_env
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "old")
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    f = tmp_path / ".env"
    f.write_text("AWS_ACCESS_KEY_ID=\nAWS_SECRET_ACCESS_KEY=new\n")
    load_env(f)
    assert os.environ["AWS_ACCESS_KEY_ID"] == "old" and os.environ["AWS_SECRET_ACCESS_KEY"] == "new"


def test_adapter_repoint_is_portable_and_idempotent(tmp_path):
    from sft_common.adapter import point_adapter_at_base
    (tmp_path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "models/pretuned", "r": 64}))
    base = tmp_path / "elsewhere" / "pretuned"
    point_adapter_at_base(tmp_path, base)
    point_adapter_at_base(tmp_path, base)
    cfg = json.loads((tmp_path / "adapter_config.json").read_text())
    assert cfg["base_model_name_or_path"] == str(base.resolve())
    assert cfg["original_base_model_name_or_path"] == "models/pretuned" and cfg["r"] == 64


# ---- wandb tracking (pure logic only; never imports wandb) ---------------------

def test_resolve_report_to_needs_both_enabled_and_key(monkeypatch):
    from sft_common.tracking import resolve_report_to
    cfg = {"training": {"wandb": {"enabled": True, "project": "p"}}}
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    assert resolve_report_to(cfg) == "none"
    monkeypatch.setenv("WANDB_API_KEY", "x")
    assert resolve_report_to(cfg) == "wandb"
    cfg["training"]["wandb"]["enabled"] = False
    assert resolve_report_to(cfg) == "none"
    assert resolve_report_to({"training": {}}) == "none"


def test_make_run_name_is_timestamped_and_prefixed():
    from datetime import datetime
    from sft_common.tracking import make_run_name
    assert make_run_name("sft", datetime(2026, 9, 27, 14, 30)) == "sft-2026-09-27_14-30"


def test_init_wandb_noop_when_not_configured(monkeypatch):
    from sft_common.tracking import init_wandb
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    assert init_wandb({"training": {"wandb": {"enabled": True, "project": "p"}}}, "run-1") is False


def test_tracking_log_is_a_safe_noop_without_an_active_run():
    from sft_common.tracking import log
    log(0, {"x": 1.0})
