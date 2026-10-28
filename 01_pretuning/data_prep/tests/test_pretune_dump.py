import pytest

from pretune_dump import detect_sections, split_fixed_size, chunk_document


class FakeTokenizer:
    """1 token per character — deterministic stand-in for the real tokenizer in tests."""
    def encode(self, text: str) -> list[str]:
        return list(text)

    def decode(self, ids: list[str]) -> str:
        return "".join(ids)


def test_detect_sections_splits_at_item_headings():
    text = "ITEM 1. BUSINESS\nWe do things.\nITEM 1A. RISK FACTORS\nThings could go wrong."
    sections = detect_sections(text)
    assert len(sections) == 2
    assert sections[0].startswith("ITEM 1. BUSINESS")
    assert sections[1].startswith("ITEM 1A. RISK FACTORS")

def test_detect_sections_returns_empty_when_no_headings():
    text = "just plain prose with no headings at all, going on and on about nothing in particular."
    assert detect_sections(text) == []

def test_detect_sections_matches_mixed_case_item_and_part_headings():
    # The original regex was ITEM/PART uppercase-only with no case handling,
    # so "Item 1A." (the actual casing in real filings) never matched and
    # detect_sections silently returned [] on nearly every real document.
    text = "Item 1. Business\nWe do things.\nPart II\nMore things."
    sections = detect_sections(text)
    assert len(sections) == 2
    assert sections[0].startswith("Item 1. Business")
    assert sections[1].startswith("Part II")

def test_detect_sections_does_not_match_lowercase_item_in_prose():
    # Staying narrow: casual lowercase "item" inside running prose (not at
    # the start of a heading-shaped line) must not trigger a false split.
    text = "ITEM 1. BUSINESS\nSee item 3 of the agreement for details.\nITEM 2. PROPERTIES\nMore."
    sections = detect_sections(text)
    assert len(sections) == 2  # only the two real ALL-CAPS headings split

def test_detect_sections_keeps_preamble_as_its_own_section():
    text = "Cover page text before any heading.\nITEM 1. BUSINESS\nBody text."
    sections = detect_sections(text)
    assert len(sections) == 2
    assert sections[0] == "Cover page text before any heading."
    assert sections[1].startswith("ITEM 1. BUSINESS")

def test_detect_sections_no_preamble_when_text_starts_with_heading():
    text = "ITEM 1. BUSINESS\nBody text."
    sections = detect_sections(text)
    assert len(sections) == 1  # no empty leading section

def test_split_fixed_size_creates_overlapping_windows():
    tokenizer = FakeTokenizer()
    text = "x" * 100
    pieces = split_fixed_size(text, target_tokens=40, overlap_tokens=10, tokenizer=tokenizer)
    assert len(pieces) > 1
    assert all(len(p) <= 40 for p in pieces)
    assert pieces[0][-10:] == pieces[1][:10]  # the overlap region matches

def test_split_fixed_size_short_text_returns_single_piece():
    tokenizer = FakeTokenizer()
    assert split_fixed_size("short", target_tokens=100, overlap_tokens=10, tokenizer=tokenizer) == ["short"]

def test_split_fixed_size_rejects_overlap_not_smaller_than_target():
    tokenizer = FakeTokenizer()
    with pytest.raises(ValueError, match="overlap_tokens"):
        split_fixed_size("x" * 100, target_tokens=20, overlap_tokens=20, tokenizer=tokenizer)

def test_chunk_document_uses_sections_when_available():
    tokenizer = FakeTokenizer()
    text = "ITEM 1. BUSINESS\n" + ("a" * 20) + "\nITEM 2. PROPERTIES\n" + ("b" * 20)
    chunks = chunk_document("doc1", text, tokenizer, target_tokens=1000, overlap_tokens=100, max_section_tokens=1000)
    assert len(chunks) == 2
    assert chunks[0]["doc_key"] == "doc1"
    assert chunks[0]["chunk_id"] == "doc1-c0"
    assert "ITEM 1. BUSINESS" in chunks[0]["chunk_text"]
    assert chunks[0]["n_tokens"] == len(tokenizer.encode(chunks[0]["chunk_text"]))

def test_chunk_document_falls_back_to_fixed_size_when_no_headings():
    tokenizer = FakeTokenizer()
    text = "x" * 250
    chunks = chunk_document("doc1", text, tokenizer, target_tokens=100, overlap_tokens=20, max_section_tokens=100)
    assert len(chunks) > 1  # fell back to fixed-size splitting of the whole document
    assert all("n_tokens" in c for c in chunks)

def test_chunk_document_sub_splits_oversized_section():
    tokenizer = FakeTokenizer()
    text = "ITEM 1. BUSINESS\n" + ("a" * 300)
    chunks = chunk_document("doc1", text, tokenizer, target_tokens=100, overlap_tokens=20, max_section_tokens=150)
    assert len(chunks) > 1  # the one oversized section got sub-split rather than staying whole

def test_chunk_document_drops_blank_chunks():
    tokenizer = FakeTokenizer()
    text = "   \n\n  "  # whitespace-only document
    chunks = chunk_document("doc1", text, tokenizer, target_tokens=100, overlap_tokens=20, max_section_tokens=100)
    assert chunks == []


from pretune_dump import dedupe_chunks

def test_dedupe_chunks_drops_exact_duplicate_text():
    chunks = [
        {"doc_key": "docA", "chunk_id": "docA-c0", "chunk_text": "Standard risk disclaimer.", "n_tokens": 3},
        {"doc_key": "docB", "chunk_id": "docB-c0", "chunk_text": "Standard risk disclaimer.", "n_tokens": 3},
        {"doc_key": "docC", "chunk_id": "docC-c0", "chunk_text": "Unique content here.", "n_tokens": 3},
    ]
    kept, n_dropped = dedupe_chunks(chunks)
    assert n_dropped == 1
    assert len(kept) == 2
    assert kept[0]["chunk_id"] == "docA-c0"  # first occurrence kept
    assert {c["chunk_id"] for c in kept} == {"docA-c0", "docC-c0"}

def test_dedupe_chunks_keeps_all_when_no_duplicates():
    chunks = [
        {"doc_key": "docA", "chunk_id": "docA-c0", "chunk_text": "aaa", "n_tokens": 1},
        {"doc_key": "docB", "chunk_id": "docB-c0", "chunk_text": "bbb", "n_tokens": 1},
    ]
    kept, n_dropped = dedupe_chunks(chunks)
    assert n_dropped == 0
    assert len(kept) == 2


from pretune_dump import split_by_document

def test_split_by_document_keeps_all_of_a_documents_chunks_together():
    chunks = (
        [{"doc_key": "docA", "chunk_id": f"docA-c{i}", "chunk_text": f"a{i}"} for i in range(5)]
        + [{"doc_key": "docB", "chunk_id": f"docB-c{i}", "chunk_text": f"b{i}"} for i in range(5)]
    )
    splits = split_by_document(chunks, val_fraction=0.5, seed=1)

    assert set(splits.keys()) == {"train", "val"}
    train_docs = {c["doc_key"] for c in splits["train"]}
    val_docs = {c["doc_key"] for c in splits["val"]}
    assert train_docs.isdisjoint(val_docs)  # no document split across both sides

def test_split_by_document_is_deterministic():
    chunks = [{"doc_key": f"doc{i}", "chunk_id": f"doc{i}-c0", "chunk_text": "x"} for i in range(20)]
    s1 = split_by_document(chunks, val_fraction=0.2, seed=7)
    s2 = split_by_document(chunks, val_fraction=0.2, seed=7)
    assert [c["chunk_id"] for c in s1["val"]] == [c["chunk_id"] for c in s2["val"]]

def test_split_by_document_respects_val_fraction_roughly():
    chunks = [{"doc_key": f"doc{i}", "chunk_id": f"doc{i}-c0", "chunk_text": "x"} for i in range(100)]
    splits = split_by_document(chunks, val_fraction=0.1, seed=3)
    val_doc_count = len({c["doc_key"] for c in splits["val"]})
    assert 5 <= val_doc_count <= 15  # loose tolerance around the 10-doc target


from pretune_dump import pack_chunks

def test_pack_chunks_fills_sequences_up_to_budget():
    tokenizer = FakeTokenizer()
    chunks = [{"chunk_text": "x" * 30, "n_tokens": 30} for _ in range(3)]  # 90 chars total + 2 separators
    packed = pack_chunks(chunks, tokenizer, max_seq_len=50, separator="|")
    # 30 + 1 + 30 = 61 > 50, so chunk 1 and 2 can't share a sequence
    assert len(packed) >= 2
    assert all(len(tokenizer.encode(p["text"])) <= 50 or "|" not in p["text"] for p in packed)

def test_pack_chunks_concatenates_when_they_fit():
    tokenizer = FakeTokenizer()
    chunks = [{"chunk_text": "aa", "n_tokens": 2}, {"chunk_text": "bb", "n_tokens": 2}, {"chunk_text": "cc", "n_tokens": 2}]
    packed = pack_chunks(chunks, tokenizer, max_seq_len=100, separator="|")
    assert len(packed) == 1
    assert packed[0]["text"] == "aa|bb|cc"

def test_pack_chunks_emits_oversized_single_chunk_alone():
    tokenizer = FakeTokenizer()
    chunks = [{"chunk_text": "x" * 20, "n_tokens": 20}, {"chunk_text": "y" * 5, "n_tokens": 5}]
    packed = pack_chunks(chunks, tokenizer, max_seq_len=10, separator="|")
    assert any(p["text"] == "x" * 20 for p in packed)  # oversized chunk kept whole, not dropped

def test_pack_chunks_records_n_tokens_alongside_text():
    tokenizer = FakeTokenizer()
    chunks = [{"chunk_text": "aa", "n_tokens": 2}, {"chunk_text": "bb", "n_tokens": 2}]
    packed = pack_chunks(chunks, tokenizer, max_seq_len=100, separator="|")
    assert packed[0]["n_tokens"] == 5  # 2 + 1 (separator) + 2

def test_pack_chunks_uses_carried_n_tokens_not_retokenized_text():
    # A poison n_tokens value that doesn't match the real text length proves
    # pack_chunks trusts the carried field instead of calling tokenizer.encode
    # on chunk_text itself.
    tokenizer = FakeTokenizer()
    chunks = [{"chunk_text": "aa", "n_tokens": 999}]
    packed = pack_chunks(chunks, tokenizer, max_seq_len=100, separator="|")
    assert packed[0]["n_tokens"] == 999


from pretune_dump import generate_stats_report, render_markdown, PACK_SEQ_LEN

def test_generate_stats_report_counts_docs_chunks_and_sequences():
    splits = {
        "train": [{"doc_key": "docA", "chunk_id": "docA-c0", "chunk_text": "x" * 10, "n_tokens": 10}] * 2,
        "val": [{"doc_key": "docB", "chunk_id": "docB-c0", "chunk_text": "y" * 5, "n_tokens": 5}],
    }
    packed_splits = {
        "train": [{"text": "x" * 10 + "|" + "x" * 10, "n_tokens": 21}],
        "val": [{"text": "y" * 5, "n_tokens": 5}],
    }
    report = generate_stats_report(splits, packed_splits, n_duplicates_dropped=3)

    assert report["duplicate_chunks_dropped"] == 3
    assert report["chunk_counts"] == {"train": 2, "val": 1}
    assert report["chunk_token_counts"] == {"train": 20, "val": 5}
    assert report["packed_sequence_counts"] == {"train": 1, "val": 1}
    assert report["packed_token_counts"] == {"train": 21, "val": 5}

def test_generate_stats_report_computes_mean_packed_fill():
    splits = {"train": [], "val": []}
    packed_splits = {
        "train": [{"text": "a", "n_tokens": PACK_SEQ_LEN}, {"text": "b", "n_tokens": PACK_SEQ_LEN // 2}],
        "val": [],
    }
    report = generate_stats_report(splits, packed_splits, n_duplicates_dropped=0)
    # (8192 + 4096) / (2 * 8192) = 0.75
    assert report["mean_packed_fill"]["train"] == 0.75
    assert report["mean_packed_fill"]["val"] == 0.0  # no sequences -> 0.0, not a division error

def test_render_markdown_produces_readable_sections():
    report = {
        "duplicate_chunks_dropped": 3,
        "chunk_counts": {"train": 2, "val": 1},
        "chunk_token_counts": {"train": 20, "val": 5},
        "packed_sequence_counts": {"train": 1, "val": 1},
        "packed_token_counts": {"train": 21, "val": 5},
        "mean_packed_fill": {"train": 0.75, "val": 0.0},
    }
    md = render_markdown(report)
    assert "Duplicate chunks dropped" in md
    assert "## Chunk Counts" in md
    assert "## Chunk Token Counts" in md
    assert "## Packed Sequences" in md
    assert "## Packed Token Counts" in md
    assert "Mean Fill" in md


from pretune_dump import build_output_files, upload_outputs_to_s3

def test_build_output_files_creates_jsonl_and_report():
    packed_splits = {"train": [{"text": "a"}, {"text": "b"}], "val": [{"text": "c"}]}
    files = build_output_files(packed_splits, "# report content")
    assert files["train.jsonl"] == '{"text": "a"}\n{"text": "b"}\n'
    assert files["val.jsonl"] == '{"text": "c"}\n'
    assert files["report.md"] == "# report content"

def test_build_output_files_writes_only_the_text_field():
    # pack_chunks now attaches an n_tokens field for stats purposes; the
    # written training data should stay {"text": ...} only, not leak internals.
    packed_splits = {"train": [{"text": "a", "n_tokens": 5}], "val": []}
    files = build_output_files(packed_splits, "# r")
    assert files["train.jsonl"] == '{"text": "a"}\n'

def test_build_output_files_handles_empty_split():
    packed_splits = {"train": [{"text": "a"}], "val": []}
    files = build_output_files(packed_splits, "# r")
    assert files["val.jsonl"] == ""

def test_upload_outputs_to_s3_puts_each_file_under_prefix():
    class FakeS3Client:
        def __init__(self):
            self.calls = []
        def put_object(self, Bucket, Key, Body):
            self.calls.append((Bucket, Key, Body))

    client = FakeS3Client()
    files = {"train.jsonl": "data1", "report.md": "data2"}
    uploaded = upload_outputs_to_s3(files, "s3://ft-vaia/Corpus/Invest_dataset/08272026_PreTuningResults", client)

    assert len(client.calls) == 2
    keys = {call[1] for call in client.calls}
    assert "Corpus/Invest_dataset/08272026_PreTuningResults/train.jsonl" in keys
    assert "Corpus/Invest_dataset/08272026_PreTuningResults/report.md" in keys
    assert set(uploaded) == {
        "s3://ft-vaia/Corpus/Invest_dataset/08272026_PreTuningResults/train.jsonl",
        "s3://ft-vaia/Corpus/Invest_dataset/08272026_PreTuningResults/report.md",
    }

def test_upload_outputs_to_s3_normalizes_missing_trailing_slash():
    class FakeS3Client:
        def __init__(self):
            self.calls = []
        def put_object(self, Bucket, Key, Body):
            self.calls.append((Bucket, Key, Body))

    client = FakeS3Client()
    upload_outputs_to_s3({"report.md": "x"}, "s3://bucket/prefix", client)  # no trailing slash
    assert client.calls[0][1] == "prefix/report.md"


from pretune_dump import check_existing_keys

def test_check_existing_keys_detects_existing_objects():
    class FakeS3Client:
        def __init__(self, existing_keys):
            self.existing_keys = existing_keys
        def head_object(self, Bucket, Key):
            if Key not in self.existing_keys:
                raise Exception("NotFound")

    client = FakeS3Client(existing_keys={"prefix/train.jsonl"})
    existing = check_existing_keys({"train.jsonl": "x", "val.jsonl": "y"}, "s3://bucket/prefix", client)
    assert existing == ["train.jsonl"]

def test_check_existing_keys_empty_when_nothing_exists():
    class FakeS3Client:
        def head_object(self, Bucket, Key):
            raise Exception("NotFound")

    client = FakeS3Client()
    existing = check_existing_keys({"train.jsonl": "x"}, "s3://bucket/prefix", client)
    assert existing == []


from pretune_dump import with_retries

def test_with_retries_retries_then_succeeds():
    calls = {"count": 0}
    def flaky():
        calls["count"] += 1
        if calls["count"] < 3:
            raise RuntimeError("transient")
        return "ok"
    result = with_retries(flaky, retries=3, backoff_seconds=0)
    assert result == "ok"
    assert calls["count"] == 3

def test_with_retries_raises_after_exhausting_retries():
    def always_fails():
        raise RuntimeError("permanent")
    with pytest.raises(RuntimeError):
        with_retries(always_fails, retries=2, backoff_seconds=0)

def test_with_retries_fails_fast_on_permanent_s3_error():
    class PermanentError(Exception):
        def __init__(self):
            self.response = {"Error": {"Code": "AccessDenied"}}

    calls = {"count": 0}
    def always_permanent():
        calls["count"] += 1
        raise PermanentError()

    with pytest.raises(Exception):
        with_retries(always_permanent, retries=3, backoff_seconds=0)
    assert calls["count"] == 1  # no retries burned on a permanent error

def test_with_retries_still_retries_transient_errors():
    calls = {"count": 0}
    def flaky():
        calls["count"] += 1
        if calls["count"] < 3:
            raise RuntimeError("transient, no .response attribute")
        return "ok"
    result = with_retries(flaky, retries=3, backoff_seconds=0)
    assert result == "ok"
    assert calls["count"] == 3


from pretune_dump import load_documents_concurrently

class _FakeBody:
    def __init__(self, data: bytes):
        self.data = data
    def read(self):
        return self.data

class _PermanentError(RuntimeError):
    """Carries the .response shape with_retries checks for a permanent S3
    error, so a failure test doesn't burn real retry/backoff sleep time."""
    def __init__(self, key):
        super().__init__(f"simulated permanent failure for {key}")
        self.response = {"Error": {"Code": "AccessDenied"}}

class FakeS3ClientForLoad:
    def __init__(self, contents: dict, failing_keys=frozenset()):
        self.contents = contents
        self.failing_keys = failing_keys
    def get_object(self, Bucket, Key):
        if Key in self.failing_keys:
            raise _PermanentError(Key)
        return {"Body": _FakeBody(self.contents[Key])}


def test_load_documents_concurrently_preserves_input_order():
    # Order must match doc_keys input order even though fetches run
    # concurrently — downstream splitting/packing behavior must stay
    # identical to the old sequential loop, per "don't alter the spec."
    doc_keys = [f"s3://bucket/doc{i}.md" for i in range(5)]
    contents = {f"doc{i}.md": f"text {i}".encode("utf-8") for i in range(5)}
    client = FakeS3ClientForLoad(contents)
    docs, failed = load_documents_concurrently(doc_keys, client, max_workers=4)
    assert [d[0] for d in docs] == doc_keys
    assert [d[1] for d in docs] == [f"text {i}" for i in range(5)]
    assert failed == []

def test_load_documents_concurrently_collects_failures_without_losing_successes():
    doc_keys = ["s3://bucket/a.md", "s3://bucket/b.md", "s3://bucket/c.md"]
    contents = {"a.md": b"A", "c.md": b"C"}
    client = FakeS3ClientForLoad(contents, failing_keys={"b.md"})
    docs, failed = load_documents_concurrently(doc_keys, client, max_workers=2)
    assert [d[0] for d in docs] == ["s3://bucket/a.md", "s3://bucket/c.md"]
    assert failed == ["s3://bucket/b.md"]

def test_load_documents_concurrently_handles_empty_input():
    docs, failed = load_documents_concurrently([], FakeS3ClientForLoad({}), max_workers=4)
    assert docs == []
    assert failed == []


from pretune_dump import load_resume_log, append_resume_log

def test_load_resume_log_returns_empty_set_when_missing(tmp_path):
    assert load_resume_log(str(tmp_path / "missing.txt")) == set()

def test_load_resume_log_and_append_round_trip(tmp_path):
    path = str(tmp_path / "resume.txt")
    append_resume_log(path, ["doc1", "doc2"])
    append_resume_log(path, ["doc3"])
    assert load_resume_log(path) == {"doc1", "doc2", "doc3"}


from pretune_dump import load_local_docs, write_output_files_locally

def test_load_local_docs_reads_all_txt_files(tmp_path):
    (tmp_path / "doc1.txt").write_text("Document one content.", encoding="utf-8")
    (tmp_path / "doc2.txt").write_text("Document two content.", encoding="utf-8")
    (tmp_path / "notes.md").write_text("should be ignored", encoding="utf-8")

    docs = load_local_docs(str(tmp_path))
    doc_keys = {d[0] for d in docs}
    assert doc_keys == {"doc1.txt", "doc2.txt"}
    texts = {d[0]: d[1] for d in docs}
    assert texts["doc1.txt"] == "Document one content."

def test_write_output_files_locally_writes_each_file(tmp_path):
    files = {"train.jsonl": "a\nb\n", "report.md": "# report"}
    out_dir = str(tmp_path / "out")
    written = write_output_files_locally(files, out_dir)

    with open(written[0], "r", encoding="utf-8") as f:
        pass  # just confirm the path is openable
    contents = {}
    import os
    for path in written:
        with open(path, "r", encoding="utf-8") as f:
            contents[os.path.basename(path)] = f.read()
    assert contents == files


from pretune_dump import (
    build_wandb_config, log_run_to_wandb,
    MODEL_NAME, CHUNK_TARGET_TOKENS, VAL_FRACTION, S3_FETCH_WORKERS,
)

class FakeWandbArtifact:
    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.references = []
    def add_reference(self, uri):
        self.references.append(uri)

class FakeWandbRun:
    def __init__(self):
        self.logged = []
        self.artifacts_logged = []
        self.finished = False
        self.exit_code = None
    def log(self, data):
        self.logged.append(data)
    def log_artifact(self, artifact):
        self.artifacts_logged.append(artifact)
    def finish(self, exit_code=0):
        self.finished = True
        self.exit_code = exit_code

class RaisingWandbRun(FakeWandbRun):
    def log(self, data):
        raise RuntimeError("simulated failure")

class FakeWandbModule:
    def __init__(self, run=None):
        self.run = run or FakeWandbRun()
        self.init_calls = []
    def init(self, project, config, job_type):
        self.init_calls.append({"project": project, "config": config, "job_type": job_type})
        return self.run
    def Artifact(self, name, type):
        return FakeWandbArtifact(name, type)


def test_build_wandb_config_includes_tunable_constants():
    config = build_wandb_config()
    assert config["model_name"] == MODEL_NAME
    assert config["chunk_target_tokens"] == CHUNK_TARGET_TOKENS
    assert config["pack_seq_len"] == PACK_SEQ_LEN
    assert config["val_fraction"] == VAL_FRACTION
    assert config["s3_fetch_workers"] == S3_FETCH_WORKERS

def test_build_wandb_config_allows_overrides():
    config = build_wandb_config(pack_seq_len=4096)
    assert config["pack_seq_len"] == 4096
    assert config["model_name"] == MODEL_NAME  # untouched defaults still present

def test_log_run_to_wandb_logs_config_and_report_without_artifact_when_no_uris():
    fake_wandb = FakeWandbModule()
    config = {"pack_seq_len": 8192}
    report = {"chunk_counts": {"train": 10, "val": 1}}

    log_run_to_wandb(fake_wandb, "test-project", config, report, artifact_uris=[])

    assert fake_wandb.init_calls[0]["project"] == "test-project"
    assert fake_wandb.init_calls[0]["config"] == config
    assert fake_wandb.run.logged == [report]
    assert fake_wandb.run.finished is True
    assert fake_wandb.run.exit_code == 0
    assert fake_wandb.run.artifacts_logged == []

def test_log_run_to_wandb_registers_dataset_artifact_when_uris_given():
    fake_wandb = FakeWandbModule()
    uris = ["s3://bucket/prefix/train.jsonl", "s3://bucket/prefix/val.jsonl"]

    log_run_to_wandb(fake_wandb, "test-project", {}, {}, artifact_uris=uris)

    assert len(fake_wandb.run.artifacts_logged) == 1
    artifact = fake_wandb.run.artifacts_logged[0]
    assert artifact.name == "pretuning-dataset"
    assert artifact.references == uris

def test_log_run_to_wandb_uses_custom_artifact_name():
    fake_wandb = FakeWandbModule()
    log_run_to_wandb(fake_wandb, "test-project", {}, {}, artifact_uris=["s3://b/k"], artifact_name="pretune-09022026")
    assert fake_wandb.run.artifacts_logged[0].name == "pretune-09022026"

def test_log_run_to_wandb_finishes_run_with_error_exit_code_on_exception():
    fake_wandb = FakeWandbModule(run=RaisingWandbRun())
    with pytest.raises(RuntimeError, match="simulated failure"):
        log_run_to_wandb(fake_wandb, "test-project", {}, {"a": 1}, artifact_uris=[])
    assert fake_wandb.run.finished is True
    assert fake_wandb.run.exit_code == 1
