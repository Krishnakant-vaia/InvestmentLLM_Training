import math

import pytest

from train_pretune import (
    build_lora_config,
    build_training_config,
    build_wandb_run_config,
    perplexity_from_loss,
    log_epoch_perplexity,
    upload_checkpoint_to_s3,
    score_mcq_question,
    mcq_accuracy,
    load_mcq_eval_set,
    sample_mcq_eval_set,
    build_s3_client,
    LORA_R,
    LORA_ALPHA,
    LEARNING_RATE,
    MAX_SEQ_LENGTH,
    BASE_MODEL,
)


class FakeWandbModule:
    """Records wandb.log(...) calls — enough of the module-level API that
    log_epoch_perplexity uses, no real wandb network calls."""
    def __init__(self):
        self.logged = []

    def log(self, data, step=None):
        self.logged.append((data, step))


def test_build_lora_config_includes_defaults():
    config = build_lora_config()
    assert config["r"] == LORA_R
    assert config["lora_alpha"] == LORA_ALPHA
    assert config["lora_dropout"] == 0
    assert config["bias"] == "none"
    # "all-linear", not an enumerated list: Qwen3.5's hybrid architecture has
    # two structurally different layer types with different projection names
    # (full-attention: q_proj/k_proj/v_proj/o_proj; Gated DeltaNet linear
    # attention, 75% of layers: in_proj_qkv/in_proj_z/in_proj_b/in_proj_a/
    # out_proj) -- an enumerated list silently under-covers whichever type
    # isn't listed, with no error, just badly undertrained layers.
    assert config["target_modules"] == "all-linear"

def test_build_lora_config_allows_overrides():
    config = build_lora_config(r=32)
    assert config["r"] == 32
    assert config["lora_alpha"] == LORA_ALPHA  # untouched defaults still present

def test_build_training_config_includes_defaults_and_output_dir():
    config = build_training_config("./outputs/run1")
    assert config["output_dir"] == "./outputs/run1"
    assert config["learning_rate"] == LEARNING_RATE
    assert config["max_length"] == MAX_SEQ_LENGTH
    assert config["dataset_text_field"] == "text"
    assert config["packing"] is False  # data is already packed by pretune_dump.py
    assert config["report_to"] == "wandb"
    # 8, not "as many CPUs as possible" -- each dataloader worker is a full
    # forked process, not a thread, so it doesn't scale like S3_FETCH_WORKERS.
    assert config["dataloader_num_workers"] == 8
    # prepare_model_for_kbit_training's use_gradient_checkpointing=True alone
    # isn't enough -- the Trainer needs this config flag too, or checkpointing
    # never actually engages during training.
    assert config["gradient_checkpointing"] is True
    # No warmup at all previously meant the fresh LoRA adapter got hit with
    # the full learning rate from step 1. warmup_steps, not warmup_ratio --
    # confirmed on the real pod that the installed trl/transformers no
    # longer accept warmup_ratio at all (TypeError), genuine version skew.
    assert config["lr_scheduler_type"] == "cosine"
    assert config["warmup_steps"] == 59
    # epoch-based checkpointing/eval with EPOCHS=1 means exactly one save and
    # one eval, at the very end of the whole run -- a pod interruption at step
    # 1,900 (of ~1,964 total) would lose everything already trained. Step-based
    # at 500 gives ~4 checkpoints, each with a matching eval data point.
    assert config["save_strategy"] == "steps"
    assert config["save_steps"] == 500
    assert config["eval_strategy"] == "steps"
    assert config["eval_steps"] == 500

def test_build_training_config_allows_overrides():
    config = build_training_config("./outputs/run1", num_train_epochs=3)
    assert config["num_train_epochs"] == 3
    assert config["output_dir"] == "./outputs/run1"  # untouched defaults still present

def test_build_wandb_run_config_includes_dataset_and_paths():
    config = build_wandb_run_config("pretune-v1", "s3-copy/train.jsonl", "s3-copy/val.jsonl")
    assert config["base_model"] == BASE_MODEL
    assert config["dataset_name"] == "pretune-v1"
    assert config["train_path"] == "s3-copy/train.jsonl"
    assert config["val_path"] == "s3-copy/val.jsonl"
    assert config["max_seq_length"] == MAX_SEQ_LENGTH

def test_perplexity_from_loss_zero_loss_is_one():
    assert perplexity_from_loss(0.0) == 1.0

def test_perplexity_from_loss_matches_exp():
    assert perplexity_from_loss(math.log(2)) == pytest.approx(2.0)

def test_log_epoch_perplexity_logs_converted_value_tagged_with_step():
    fake_wandb = FakeWandbModule()
    log_epoch_perplexity(fake_wandb, eval_loss=math.log(2), step=42)
    data, step = fake_wandb.logged[0]
    assert data["eval_perplexity"] == pytest.approx(2.0)
    # step is a plain data field, NOT wandb.log's own step= kwarg -- passing
    # step= there was confirmed on the real pod to silently drop the whole
    # log call whenever wandb's internal counter had already advanced past
    # it from HF Trainer's own concurrent logging.
    assert data["trainer_step"] == 42
    assert step is None

def test_log_epoch_perplexity_does_nothing_when_eval_loss_is_none():
    # on_evaluate can fire with metrics that lack eval_loss (e.g. a
    # train-only evaluation call) — nothing to convert, nothing to log.
    fake_wandb = FakeWandbModule()
    log_epoch_perplexity(fake_wandb, eval_loss=None, step=10)
    assert fake_wandb.logged == []


class FakeS3Client:
    def __init__(self):
        self.calls = []
    def put_object(self, Bucket, Key, Body):
        self.calls.append((Bucket, Key, Body))


def test_upload_checkpoint_to_s3_uploads_every_file_preserving_structure(tmp_path):
    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")
    (tmp_path / "adapter_config.json").write_text('{"r": 16}', encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "tokenizer.json").write_text("{}", encoding="utf-8")

    client = FakeS3Client()
    uploaded = upload_checkpoint_to_s3(str(tmp_path), "s3://bucket/prefix", client)

    keys = {call[1] for call in client.calls}
    assert keys == {"prefix/adapter_model.safetensors", "prefix/adapter_config.json", "prefix/sub/tokenizer.json"}
    assert set(uploaded) == {
        "s3://bucket/prefix/adapter_model.safetensors",
        "s3://bucket/prefix/adapter_config.json",
        "s3://bucket/prefix/sub/tokenizer.json",
    }

def test_upload_checkpoint_to_s3_normalizes_missing_trailing_slash(tmp_path):
    (tmp_path / "f.txt").write_text("x", encoding="utf-8")
    client = FakeS3Client()
    upload_checkpoint_to_s3(str(tmp_path), "s3://bucket/prefix", client)  # no trailing slash
    assert client.calls[0][1] == "prefix/f.txt"

def test_upload_checkpoint_to_s3_uploads_real_file_bytes(tmp_path):
    (tmp_path / "weights.bin").write_bytes(b"\x00\x01binary-content")
    client = FakeS3Client()
    upload_checkpoint_to_s3(str(tmp_path), "s3://bucket/prefix/", client)
    assert client.calls[0][2] == b"\x00\x01binary-content"


# token_accuracy and its tests were deleted 2026-09-03: the real pod smoke
# test showed trl==1.12.0/transformers==5.16.1 already compute and log
# mean_token_accuracy (train) and eval_mean_token_accuracy (validation)
# natively via report_to="wandb" -- no custom code needed. See
# evaluate_perplexity's docstring in train_pretune.py for the full story.


# --- score_mcq_question / mcq_accuracy: plain decision logic, no model needed ---

def test_score_mcq_question_picks_highest():
    assert score_mcq_question([-5.2, -1.1, -3.0]) == 1

def test_score_mcq_question_single_option():
    assert score_mcq_question([-0.5]) == 0

def test_mcq_accuracy_all_correct():
    assert mcq_accuracy([0, 1, 2], [0, 1, 2]) == pytest.approx(1.0)

def test_mcq_accuracy_partial():
    assert mcq_accuracy([0, 1, 2], [0, 0, 2]) == pytest.approx(2 / 3)

def test_mcq_accuracy_empty_returns_zero():
    assert mcq_accuracy([], []) == 0.0


# --- load_mcq_eval_set: schema-flexible field detection ---

def test_load_mcq_eval_set_matches_question_options_answer_with_int_answer():
    raw = [{"question": "Q1?", "options": ["A opt", "B opt"], "answer": 1}]
    out = load_mcq_eval_set(raw)
    assert out == [{"question": "Q1?", "options": ["A opt", "B opt"], "answer": 1}]

def test_load_mcq_eval_set_resolves_letter_answer_via_alphabet_position():
    raw = [{"query": "Q1?", "choices": ["opt one", "opt two", "opt three"], "gold": "C"}]
    out = load_mcq_eval_set(raw)
    assert out[0]["answer"] == 2  # C -> index 2

def test_load_mcq_eval_set_resolves_answer_found_literally_in_options():
    raw = [{"prompt": "Q1?", "answers": ["X", "Y", "Z"], "correct_answer": "Y"}]
    out = load_mcq_eval_set(raw)
    assert out[0]["answer"] == 1

def test_load_mcq_eval_set_raises_clear_error_on_unrecognized_schema():
    raw = [{"totally_unexpected_field": "nope"}]
    with pytest.raises(ValueError, match="totally_unexpected_field"):
        load_mcq_eval_set(raw)

def test_load_mcq_eval_set_empty_input_returns_empty():
    assert load_mcq_eval_set([]) == []

def test_load_mcq_eval_set_matches_real_bhashabench_schema():
    # confirmed against the real dataset 2026-09-03: separate option_a..d
    # columns (not a single list field), correct_answer as a letter
    raw = [{
        "id": "abc123", "question": "Which is the odd one out?", "correct_answer": "C",
        "option_a": "103", "option_b": "12", "option_c": "38", "option_d": "228",
        "language": "en", "question_type": "MCQ", "question_level": "Medium",
        "topic": "Quantitative Aptitude", "subject_domain": "Mathematics for Finance",
    }]
    out = load_mcq_eval_set(raw)
    assert out == [{
        "question": "Which is the odd one out?",
        "options": ["103", "12", "38", "228"],
        "answer": 2,  # C -> index 2 -> "38"
    }]

def test_load_mcq_eval_set_skips_non_mcq_question_types():
    raw = [
        {"question": "Fill this in", "correct_answer": "A", "option_a": "x", "option_b": "y",
         "question_type": "Fill in blanks"},
        {"question": "Real MCQ", "correct_answer": "B", "option_a": "x", "option_b": "y",
         "question_type": "MCQ"},
    ]
    out = load_mcq_eval_set(raw)
    assert len(out) == 1
    assert out[0]["question"] == "Real MCQ"

def test_load_mcq_eval_set_all_non_mcq_returns_empty():
    raw = [{"question": "Fill this in", "correct_answer": "A", "option_a": "x", "question_type": "Fill in blanks"}]
    assert load_mcq_eval_set(raw) == []

def test_load_mcq_eval_set_option_columns_skip_missing_values():
    # a row with fewer real options than the max candidate count -- only
    # present, non-empty option_X columns should end up in the options list
    raw = [{"question": "Q?", "correct_answer": "B", "option_a": "x", "option_b": "y", "option_c": None}]
    out = load_mcq_eval_set(raw)
    assert out[0]["options"] == ["x", "y"]
    assert out[0]["answer"] == 1


# --- sample_mcq_eval_set: deterministic subsampling ---

def test_sample_mcq_eval_set_returns_requested_size():
    dataset = [{"question": f"Q{i}"} for i in range(1000)]
    sample = sample_mcq_eval_set(dataset, 300, seed=42)
    assert len(sample) == 300

def test_sample_mcq_eval_set_same_seed_same_sample():
    dataset = [{"question": f"Q{i}"} for i in range(1000)]
    a = sample_mcq_eval_set(dataset, 300, seed=42)
    b = sample_mcq_eval_set(dataset, 300, seed=42)
    assert a == b

def test_sample_mcq_eval_set_smaller_than_sample_size_returns_unchanged():
    dataset = [{"question": f"Q{i}"} for i in range(10)]
    assert sample_mcq_eval_set(dataset, 300, seed=42) == dataset


# --- build_s3_client: real timeouts, so a stuck connection fails and retries
# instead of hanging the whole run indefinitely (confirmed twice on real pods) ---

class FakeBoto3Module:
    def __init__(self):
        self.calls = []
    def client(self, service_name, **kwargs):
        self.calls.append((service_name, kwargs))
        return "fake-client"

def test_build_s3_client_sets_real_timeouts():
    fake_boto3 = FakeBoto3Module()
    client = build_s3_client(fake_boto3)
    assert client == "fake-client"
    service_name, kwargs = fake_boto3.calls[0]
    assert service_name == "s3"
    config = kwargs["config"]
    assert config.connect_timeout == 30
    assert config.read_timeout == 120
    assert config.retries["max_attempts"] == 5
    assert config.retries["mode"] == "adaptive"
