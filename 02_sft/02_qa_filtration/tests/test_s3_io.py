import json
from unittest.mock import MagicMock, patch

import boto3
from moto import mock_aws
from qa_filter.s3_io import stream_jsonl_lines, upload_jsonl, upload_json


@mock_aws
def test_stream_jsonl_lines_yields_each_line():
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="test-bucket")
    body = "\n".join([json.dumps({"i": i}) for i in range(5)]) + "\n"
    s3.put_object(Bucket="test-bucket", Key="data.jsonl", Body=body.encode("utf-8"))

    lines = list(stream_jsonl_lines("test-bucket", "data.jsonl"))

    assert len(lines) == 5
    assert json.loads(lines[0]) == {"i": 0}
    assert json.loads(lines[4]) == {"i": 4}


def _fake_body(chunks):
    body = MagicMock()
    body.iter_chunks.return_value = iter(chunks)
    return {"Body": body}


def test_stream_jsonl_lines_resumes_from_exact_offset_after_connection_drop():
    # Full content: b'{"i": 0}\n{"i": 1}\n{"i": 2}\n' (28 bytes). First attempt
    # delivers the first chunk (up to and including "{"i": 1}\n", 18 bytes) then
    # raises -- simulating a connection reset mid-stream. A correct resume must
    # re-request starting at byte 18 (Range: bytes=18-), not byte 0, and must not
    # skip or duplicate the {"i": 1} line that already made it through.
    full_content = b'{"i": 0}\n{"i": 1}\n{"i": 2}\n'
    first_chunk = full_content[:18]

    def failing_first_attempt():
        yield first_chunk
        raise ConnectionResetError("simulated network drop")

    fake_client = MagicMock()
    fake_client.head_object.return_value = {"ContentLength": len(full_content)}

    first_call_body = MagicMock()
    first_call_body.iter_chunks.return_value = failing_first_attempt()
    second_call_body = MagicMock()
    second_call_body.iter_chunks.return_value = iter([full_content[18:]])
    fake_client.get_object.side_effect = [
        {"Body": first_call_body},
        {"Body": second_call_body},
    ]

    with patch("qa_filter.s3_io._client", return_value=fake_client), \
         patch("qa_filter.s3_io.time.sleep"):
        lines = list(stream_jsonl_lines("test-bucket", "data.jsonl"))

    assert lines == ['{"i": 0}', '{"i": 1}', '{"i": 2}']
    calls = fake_client.get_object.call_args_list
    assert calls[0].kwargs["Range"] == "bytes=0-"
    assert calls[1].kwargs["Range"] == "bytes=18-"


def test_stream_jsonl_lines_raises_after_exhausting_retries():
    fake_client = MagicMock()
    fake_client.head_object.return_value = {"ContentLength": 100}
    fake_client.get_object.side_effect = ConnectionResetError("simulated network drop")

    with patch("qa_filter.s3_io._client", return_value=fake_client), \
         patch("qa_filter.s3_io.time.sleep"):
        try:
            list(stream_jsonl_lines("test-bucket", "data.jsonl", max_retries=2))
            assert False, "expected ConnectionResetError"
        except ConnectionResetError:
            pass

    assert fake_client.get_object.call_count == 3  # initial attempt + 2 retries


@mock_aws
def test_upload_jsonl_puts_file_contents(tmp_path):
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="test-bucket")
    local_file = tmp_path / "kept.jsonl"
    # write_bytes, not write_text: write_text on Windows translates '\n' to '\r\n'
    # on write (universal newlines), which would corrupt this byte-for-byte check.
    local_file.write_bytes(b'{"a": 1}\n{"a": 2}\n')

    upload_jsonl("test-bucket", "out/kept.jsonl", str(local_file))

    obj = s3.get_object(Bucket="test-bucket", Key="out/kept.jsonl")
    assert obj["Body"].read().decode("utf-8") == '{"a": 1}\n{"a": 2}\n'


@mock_aws
def test_upload_json_serializes_dict():
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="test-bucket")

    upload_json("test-bucket", "out/report.json", {"pass_rate": 0.9})

    obj = s3.get_object(Bucket="test-bucket", Key="out/report.json")
    assert json.loads(obj["Body"].read().decode("utf-8")) == {"pass_rate": 0.9}
