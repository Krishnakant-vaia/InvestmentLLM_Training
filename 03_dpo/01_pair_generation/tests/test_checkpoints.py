"""Judge checkpoints on S3: structure, round trip, safety (fingerprint, failed uploads), and a simulated eviction and resume.
S3 is replaced by an in-memory fake; the judge model by the fake OpenAI-compatible server of test_qa_pairs_prep."""
import io
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import checkpoints  # noqa: E402
import config  # noqa: E402
import judge  # noqa: E402
import s3util  # noqa: E402
from test_qa_pairs_prep import fake_server  # noqa: E402,F401  (pytest fixture)


class FakeS3:
    """The few boto3 calls s3util and checkpoints make, on a dict."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.fail_puts = 0

    def put_object(self, Bucket, Key, Body):
        if self.fail_puts:
            self.fail_puts -= 1
            raise OSError("simulated upload failure")
        self.objects[Key] = Body

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[Key])}

    def list_objects_v2(self, Bucket, Prefix, MaxKeys=1000):
        keys = [k for k in sorted(self.objects) if k.startswith(Prefix)][:MaxKeys]
        return {"KeyCount": len(keys), "Contents": [{"Key": k} for k in keys]}

    def get_paginator(self, name):
        return self

    def paginate(self, Bucket, Prefix):
        yield self.list_objects_v2(Bucket, Prefix)

    def upload_file(self, filename, bucket, key):
        self.objects[key] = Path(filename).read_bytes()


def verdict(i):
    return json.dumps({"id": i, "a": {"score": 9, "reason": "PASS_ALL_GATES", "failed_gates": []},
                       "b": {"score": 2, "reason": "F06_NUMERICAL_ERROR", "failed_gates": ["G3", "G8"]}, "effort": "medium"})


def write_verdicts(path: Path, ids) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.writelines(verdict(i) + "\n" for i in ids)


META = {"run": "r1", "total": 8, "input_sha256": "abc"}


def keys(s3) -> list[str]:
    return sorted(k[len(config.OUT_ROOT):] for k in s3.objects)


def test_stage_layout():
    assert s3util.stage_key("full-1", "pairs_raw.jsonl").endswith("full-1/generation/pairs_raw.jsonl")
    assert s3util.stage_key("full-1", "scored_pairs.jsonl").endswith("full-1/final/scored_pairs.jsonl")
    assert s3util.stage_key("full-1", "server.log").endswith("full-1/logs/server.log")
    assert s3util.stage_key("full-1/logs", "console.log").endswith("full-1/logs/console.log")
    assert s3util.stage_key("corpus", "chunks.jsonl").endswith("corpus/chunks.jsonl")


def test_push_writes_each_part_once_and_a_manifest(tmp_path):
    s3, out = FakeS3(), tmp_path / "out.jsonl"
    saved, parts = set(), 0
    write_verdicts(out, range(0, 5))
    parts = checkpoints.push(s3, "r1", out, saved, parts, META)
    write_verdicts(out, range(5, 8))
    parts = checkpoints.push(s3, "r1", out, saved, parts, META, complete=True)
    assert parts == 2 and saved == set(range(8))
    assert keys(s3) == ["r1/judge/manifest.json", "r1/judge/verdicts/part-00001.jsonl", "r1/judge/verdicts/part-00002.jsonl"]
    part1 = s3.objects[config.OUT_ROOT + "r1/judge/verdicts/part-00001.jsonl"].decode().split("\n")
    assert [json.loads(l)["id"] for l in part1 if l] == [0, 1, 2, 3, 4]          # the second part holds only what was new
    manifest = json.loads(s3.objects[config.OUT_ROOT + "r1/judge/manifest.json"])
    assert manifest["judged"] == 8 and manifest["parts"] == 2 and manifest["complete"] is True and manifest["total"] == 8
    before = dict(s3.objects)
    checkpoints.push(s3, "r1", out, saved, parts, META, complete=True)           # nothing new: no new part, the old parts are untouched
    assert [k for k in s3.objects if "part-" in k] == [k for k in before if "part-" in k]
    assert all(s3.objects[k] == before[k] for k in before if "part-" in k)


def test_pull_restores_everything_on_a_fresh_machine_and_skips_a_half_written_line(tmp_path):
    s3, out = FakeS3(), tmp_path / "out.jsonl"
    write_verdicts(out, range(6))
    checkpoints.push(s3, "r1", out, set(), 0, META)
    fresh = tmp_path / "fresh" / "judge_out.jsonl"
    fresh.parent.mkdir()
    fresh.write_text(verdict(99) + "\n" + '{"id": 100, "a": ')                    # local file with one extra verdict and a torn last line
    saved, n_parts, manifest = checkpoints.pull(s3, "r1", fresh, "abc")
    ids = sorted(json.loads(l)["id"] for l in fresh.read_text().split("\n") if l.strip())
    assert saved == set(range(6)) and n_parts == 1 and manifest["judged"] == 6
    assert ids == [0, 1, 2, 3, 4, 5, 99]


def test_checkpoints_are_never_mixed_with_a_different_judge_input(tmp_path):
    s3, out = FakeS3(), tmp_path / "out.jsonl"
    write_verdicts(out, range(3))
    checkpoints.push(s3, "r1", out, set(), 0, META)
    with pytest.raises(RuntimeError, match="different judge input"):
        checkpoints.pull(s3, "r1", tmp_path / "new.jsonl", "a-different-fingerprint")


def test_a_failed_upload_is_retried_at_the_next_checkpoint(tmp_path):
    s3, out, saved = FakeS3(), tmp_path / "out.jsonl", set()
    write_verdicts(out, range(4))
    s3.fail_puts = 1
    parts = checkpoints.push(s3, "r1", out, saved, 0, META)
    assert parts == 0 and saved == set() and keys(s3) == []                      # nothing was recorded as saved
    parts = checkpoints.push(s3, "r1", out, saved, parts, META)
    assert parts == 1 and saved == set(range(4))


def test_judge_survives_an_eviction_and_resumes_from_s3(tmp_path, fake_server, monkeypatch):
    s3 = FakeS3()
    monkeypatch.setattr(s3util, "client", lambda: s3)
    lines = [{"id": i, "document": "DOC", "question": "Q?", "answer_a": "CHOSEN_ANSWER ok",
              "answer_b": "REJECTED_ANSWER defect=padded extra words"} for i in range(25)]
    inp = tmp_path / "judge_in.jsonl"
    inp.write_text("\n".join(json.dumps(x) for x in lines))
    first = tmp_path / "first" / "judge_out.jsonl"
    first.parent.mkdir()
    assert judge.run(inp, first, batch=5, workers=2, url=fake_server, s3_run="r1", upload_every=2) == 25
    manifest = json.loads(s3.objects[config.OUT_ROOT + "r1/judge/manifest.json"])
    assert manifest["complete"] and manifest["judged"] == 25 and manifest["parts"] >= 2
    assert any(k.endswith("r1/judge/judge_out.jsonl") for k in s3.objects)       # the merged copy
    # the sandbox is lost: a new machine has no local verdicts, and the model server is not even reachable
    second = tmp_path / "second" / "judge_out.jsonl"
    second.parent.mkdir()
    assert judge.run(inp, second, batch=5, workers=2, url="http://127.0.0.1:9", s3_run="r1") == 0
    assert sorted(json.loads(l)["id"] for l in second.read_text().split("\n") if l.strip()) == list(range(25))
