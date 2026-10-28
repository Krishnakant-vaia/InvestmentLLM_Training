"""The DPO trainer end to end on CPU: a tiny random Qwen3 + LoRA, the real tokenizer, real Trainer machinery.

Proves the parts that break silently on a GPU run: the reference really is the adapter-free model, the first eval
(policy == reference) gives loss ln 2, training moves only the adapter, per-source evals and the averaged metrics reach
the log, and the batch plan covers every pair. Skipped without peft or the pre-tuned tokenizer files.
"""
import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest

peft = pytest.importorskip("peft")
torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sft_common.config import load_config  # noqa: E402
from sft_common.dpo_data import prepare  # noqa: E402

if not (ROOT / "models" / "pretuned" / "tokenizer.json").exists():
    pytest.skip("pre-tuned tokenizer files not downloaded", allow_module_level=True)


@pytest.fixture(scope="module")
def dpo_train():
    """02_training/dpo_train.py imported with Unsloth stubbed out (it needs a GPU stack; the trainer itself does not)."""
    stub = types.ModuleType("unsloth")
    stub.FastLanguageModel = object
    sys.modules["unsloth"] = stub
    try:
        spec = importlib.util.spec_from_file_location("dpo_train", ROOT / "02_training" / "dpo_train.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.modules.pop("unsloth", None)


def make_trainer(mod, tmp_path, monkeypatch):
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    tok = AutoTokenizer.from_pretrained(ROOT / "models" / "pretuned")
    cfg = load_config()
    d = dict(cfg["dpo"], tokens_per_micro_batch=1200, max_pair_tokens=600, max_rows_per_micro_batch=4, grad_accum_steps=2,
             eval_steps=3, save_steps=3, learning_rate=2e-3, warmup_ratio=0.0, logging_steps=1, optimizer="adamw_torch")
    cfg["dpo"] = d
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tok), hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                         num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=2048))
    model = peft.get_peft_model(model, peft.LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["q_proj", "v_proj"]))
    rows = [{"prompt": f"Question {i // 2}: what is a dividend?", "preferred": f"Good answer {i} about cash paid to owners.",
             "not_preferred": f"Bad answer {i}.", "source": "domain" if i % 3 else "general", "line": i} for i in range(90)]   # 2 pairs per prompt
    prepared = prepare(rows, tok, d["data"] | {"max_pair_tokens": d["max_pair_tokens"], "val_fraction": 0.25}, seed=1)
    assert prepared.train and set(prepared.val) == {"domain", "general"}

    real_args = mod.TrainingArguments
    monkeypatch.setattr(mod, "TrainingArguments", lambda **kw: real_args(**{**kw, "bf16": False, "use_cpu": True, "report_to": "none",
                                                                            "dataloader_num_workers": 0, "dataloader_pin_memory": False}))
    run = types.SimpleNamespace(train_dir=tmp_path / "run", name="dpo-test")
    args = mod.build_args(cfg, run, None, True)
    trainer = mod.DPOBudgetTrainer(model=model, args=args, train_dataset=mod.to_dataset(prepared.train),
                                   eval_dataset={s: mod.to_dataset(v) for s, v in prepared.val.items()},
                                   processing_class=tok, callbacks=[], d=d, pad_id=tok.pad_token_id or tok.eos_token_id)
    return trainer, model, d, tmp_path


def test_reference_is_the_frozen_model_and_training_moves_only_the_adapter(dpo_train, tmp_path, monkeypatch):
    trainer, model, d, tmp = make_trainer(dpo_train, tmp_path, monkeypatch)
    assert trainer.model_accepts_loss_kwargs is False                      # the Trainer divides by the accumulation steps itself
    base_before = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" not in n}

    lines = dpo_train.plan_and_check(trainer, tmp / "logs" / "batch_plan.jsonl", None)
    assert "each in exactly one micro-batch" in lines[0] and (tmp / "logs" / "batch_plan.jsonl").exists()
    assert dpo_train.reference_check(trainer)

    trainer.train()
    hist = trainer.state.log_history
    first = next(h for h in hist if "eval_domain_loss" in h)
    assert first["eval_domain_loss"] == pytest.approx(math.log(2), abs=1e-3)       # step 0: policy == reference
    assert first["eval_domain_rewards/margins"] == pytest.approx(0.0, abs=1e-4)
    assert {"eval_general_loss", "eval_general_ref/prefers_chosen"} <= {k for h in hist for k in h}
    train_logs = [h for h in hist if "loss" in h and "rewards/accuracies" in h]
    assert train_logs and all(0.0 <= h["rewards/accuracies"] <= 1.0 for h in train_logs)
    assert any((p.detach() != 0).any() for n, p in model.named_parameters() if "lora_B" in n)    # the adapter learned
    for n, p in model.named_parameters():
        if "lora_" not in n:
            assert torch.equal(p, base_before[n]), f"base weight {n} changed"                  # the SFT model stayed frozen
    assert (tmp / "run" / "checkpoint-3").exists()                                              # saved on schedule


def test_resume_with_a_different_grouping_is_refused(dpo_train, tmp_path, monkeypatch):
    trainer, _, d, tmp = make_trainer(dpo_train, tmp_path, monkeypatch)
    plan = tmp / "plan.jsonl"
    dpo_train.plan_and_check(trainer, plan, None)
    import hashlib
    original = hashlib.sha256(plan.read_bytes()).hexdigest()
    dpo_train.plan_and_check(trainer, plan, original)                                           # same data and config: accepted
    trainer.d = dict(d, max_rows_per_micro_batch=1)                                            # config changed since the run started
    with pytest.raises(SystemExit, match="grouping differs"):
        dpo_train.plan_and_check(trainer, plan, original)
