import json

import pytest

from sft_generation_core import (
    extract_capability_code,
    assign_group,
    chunk_id,
    iter_chunks,
    list_s3_keys,
    load_progress,
    parse_examples,
    build_record,
    upload_output_to_s3,
    log_line,
    document_is_settled,
)

CAPABILITY_GROUPS = {
    "first_4_capabilities": {"DK", "EK", "FR", "WF"},
    "last_4_capabilities": {"CG", "AR", "CL", "EA"},
}
S3_PREFIX = "Corpus/Invest_dataset/082126_fliteredextractionrun2/"


# --- capability code extraction / grouping ---

def test_extract_capability_code_real_example():
    key = S3_PREFIX + "AR_AUTONOMOUS_CLOSE_THE_LOOP_EXECUTION - Autonomous close-the-loop execution/doc1.md"
    assert extract_capability_code(key, S3_PREFIX) == "AR"

def test_extract_capability_code_no_underscore_returns_whole_segment():
    key = S3_PREFIX + "WEIRDFOLDERNAME/doc1.md"
    assert extract_capability_code(key, S3_PREFIX) == "WEIRDFOLDERNAME"

def test_extract_capability_code_key_equal_to_prefix_returns_none():
    # nothing after the prefix at all -> empty first segment -> no code
    assert extract_capability_code(S3_PREFIX, S3_PREFIX) is None

def test_assign_group_known_code():
    key = S3_PREFIX + "DK_SOME_SLUG - Domain Knowledge thing/doc1.md"
    assert assign_group(key, S3_PREFIX, CAPABILITY_GROUPS) == "first_4_capabilities"

def test_assign_group_unknown_code_returns_none():
    key = S3_PREFIX + "ZZ_UNKNOWN_CAPABILITY - mystery/doc1.md"
    assert assign_group(key, S3_PREFIX, CAPABILITY_GROUPS) is None


# --- chunk_id / iter_chunks (unchanged behavior, tested here for the first time) ---

def test_chunk_id_stable_for_same_key_and_index():
    assert chunk_id("some/key.md", 0) == chunk_id("some/key.md", 0)

def test_chunk_id_differs_by_index():
    assert chunk_id("some/key.md", 0) != chunk_id("some/key.md", 1)

def test_iter_chunks_short_text_yields_one_chunk():
    assert list(iter_chunks("hello world", 100, 10)) == ["hello world"]

def test_iter_chunks_blank_text_yields_nothing():
    assert list(iter_chunks("   ", 100, 10)) == []

def test_iter_chunks_overlap_clamped_when_too_large():
    # overlap >= size would loop forever without the clamp
    chunks = list(iter_chunks("a" * 250, 100, 500))
    assert len(chunks) > 0
    assert len(chunks) < 1000  # sanity: terminates


# --- list_s3_keys: splits by group, tallies unmatched ---

class FakePaginator:
    def __init__(self, pages):
        self._pages = pages
    def paginate(self, **kwargs):
        return self._pages

class FakeS3ListClient:
    def __init__(self, keys):
        self._keys = keys
    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return FakePaginator([{"Contents": [{"Key": k} for k in self._keys]}])

def test_list_s3_keys_splits_by_group_and_tallies_unmatched():
    keys = [
        S3_PREFIX + "DK_FOO - x/a.md",
        S3_PREFIX + "AR_BAR - y/b.md",
        S3_PREFIX + "ZZ_UNKNOWN - z/c.md",
        S3_PREFIX + "WF_BAZ - w/d.txt",
        S3_PREFIX + "EK_QUX - v/e.pdf",  # wrong extension, excluded entirely
    ]
    s3 = FakeS3ListClient(keys)
    matched, unmatched = list_s3_keys(s3, "bucket", S3_PREFIX, {".md", ".txt"},
                                       "first_4_capabilities", CAPABILITY_GROUPS)
    assert matched == [S3_PREFIX + "DK_FOO - x/a.md", S3_PREFIX + "WF_BAZ - w/d.txt"]
    assert unmatched == {"ZZ": 1}


# --- load_progress ---

def test_load_progress_no_file_returns_empty(tmp_path):
    done, complete = load_progress(str(tmp_path / "missing.jsonl"))
    assert done == set()
    assert complete == set()

def test_load_progress_marks_source_complete_when_all_chunks_seen(tmp_path):
    path = tmp_path / "out.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"_chunk_id": "a", "_source": "doc1", "_n_chunks": 2}) + "\n")
        f.write(json.dumps({"_chunk_id": "b", "_source": "doc1", "_n_chunks": 2}) + "\n")
    done, complete = load_progress(str(path))
    assert done == {"a", "b"}
    assert complete == {"doc1"}

def test_load_progress_partial_source_not_complete(tmp_path):
    path = tmp_path / "out.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"_chunk_id": "a", "_source": "doc1", "_n_chunks": 2}) + "\n")
    done, complete = load_progress(str(path))
    assert done == {"a"}
    assert complete == set()


# --- parse_examples: new schema, empty-array-is-valid, malformed-is-not ---

def _valid_pair_json():
    return json.dumps({
        "question_type": "Summary",
        "difficulty": "Basic",
        "section_context": "Overview",
        "question": "What is the headline figure?",
        "thinking": "<think>reasoning here</think>",
        "answer": "The headline figure is 42.",
    })

def test_parse_examples_valid_pair():
    raw = f"[{_valid_pair_json()}]"
    out = parse_examples(raw)
    assert len(out) == 1
    assert out[0]["question"] == "What is the headline figure?"
    assert out[0]["thinking"] == "<think>reasoning here</think>"
    assert out[0]["question_type"] == "Summary"

def test_parse_examples_deliberate_empty_array_is_valid_not_an_error():
    assert parse_examples("[]") == []
    assert parse_examples("  [ ]  ") == []

def test_parse_examples_malformed_output_raises():
    with pytest.raises(ValueError):
        parse_examples("not json at all, just prose")

def test_parse_examples_object_missing_required_field_is_dropped_not_fatal():
    incomplete = json.dumps({"question_type": "Summary", "question": "Q?", "answer": "A."})  # no thinking
    raw = f"[{incomplete}]"
    # dropped silently -> empty list, no exception, since the reply DID contain a
    # complete JSON object (just not a usable one)
    assert parse_examples(raw) == []

def test_parse_examples_truncated_reply_salvages_complete_objects():
    # complete first object, then a second object cut off mid-string (never closes)
    raw = "[" + _valid_pair_json() + ', {"question": "cut off mid'
    out = parse_examples(raw)
    assert len(out) == 1


# --- build_record: new field names, no double-wrapping ---

def test_build_record_shape():
    ex = {
        "question_type": "Summary",
        "difficulty": "Basic",
        "section_context": "Overview",
        "question": "What happened?",
        "thinking": "<think>because X</think>",
        "answer": "X happened.",
    }
    rec = build_record("cid1", "doc1.md", "chunk text", 3, ex, "@cf/openai/gpt-oss-120b",
                        "You are a finance and investment assistant.")
    messages = rec["messages"]
    assert messages[0] == {"role": "system", "content": "You are a finance and investment assistant."}
    assert messages[1]["role"] == "user"
    assert "<document>\nchunk text\n</document>" in messages[1]["content"]
    assert "What happened?" in messages[1]["content"]
    assert messages[2] == {"role": "assistant", "content": "<think>because X</think>\n\nX happened."}
    # no double-wrapping: thinking appears exactly once
    assert messages[2]["content"].count("<think>") == 1
    assert rec["_question_type"] == "Summary"
    assert rec["_difficulty"] == "Basic"
    assert rec["_section_context"] == "Overview"
    assert rec["_chunk_id"] == "cid1"
    assert rec["_source"] == "doc1.md"
    assert rec["_n_chunks"] == 3


# --- upload_output_to_s3 ---

class FakeS3PutClient:
    def __init__(self):
        self.calls = []
    def put_object(self, Bucket, Key, Body):
        self.calls.append((Bucket, Key, Body))

def test_upload_output_to_s3_uploads_real_bytes(tmp_path):
    local = tmp_path / "sft_data_first_4_capabilities.jsonl"
    local.write_bytes(b'{"messages": []}\n')
    client = FakeS3PutClient()
    uri = upload_output_to_s3(str(local), "s3://ft-vaia/Corpus/Invest_dataset/09032026_SFTQ&Apairs/first_4_capabilities/", client)
    assert client.calls[0][0] == "ft-vaia"
    assert client.calls[0][1] == "Corpus/Invest_dataset/09032026_SFTQ&Apairs/first_4_capabilities/sft_data_first_4_capabilities.jsonl"
    assert client.calls[0][2] == b'{"messages": []}\n'
    assert uri == "s3://ft-vaia/Corpus/Invest_dataset/09032026_SFTQ&Apairs/first_4_capabilities/sft_data_first_4_capabilities.jsonl"

def test_upload_output_to_s3_normalizes_missing_trailing_slash(tmp_path):
    local = tmp_path / "f.jsonl"
    local.write_text("x", encoding="utf-8")
    client = FakeS3PutClient()
    upload_output_to_s3(str(local), "s3://bucket/prefix", client)  # no trailing slash
    assert client.calls[0][1] == "prefix/f.jsonl"


# --- log_line: prints and persists, never fatal on write failure ---

def test_log_line_appends_to_file(tmp_path, capsys):
    path = tmp_path / "run.log"
    log_line("hello", str(path))
    log_line("world", str(path))
    assert path.read_text(encoding="utf-8") == "hello\nworld\n"
    assert "hello" in capsys.readouterr().out

def test_log_line_survives_unwritable_path(capsys):
    # a directory that doesn't exist -> open() raises OSError -> must not propagate
    log_line("still prints", "Z:/definitely/does/not/exist/run.log")
    assert "still prints" in capsys.readouterr().out


# --- document_is_settled: the pause-boundary check ---

def test_document_is_settled_true_once_all_pending_chunks_attempted():
    attempted = {"doc1": {"a", "b"}}
    expected = {"doc1": 2}
    assert document_is_settled("doc1", attempted, expected) is True

def test_document_is_settled_false_when_partial():
    attempted = {"doc1": {"a"}}
    expected = {"doc1": 2}
    assert document_is_settled("doc1", attempted, expected) is False

def test_document_is_settled_false_when_source_unknown():
    assert document_is_settled("never-seen", {}, {}) is False
