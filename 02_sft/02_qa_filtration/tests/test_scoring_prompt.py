import re

import pytest

from qa_scoring.prompt import CATEGORY_CODES, FAILURE_CODES, PLACEHOLDER, load_prompt


def test_default_prompt_loads_and_has_placeholder():
    text, version = load_prompt()
    assert PLACEHOLDER in text
    assert version.startswith("gepa_quality_run2-")
    assert len(version) == len("gepa_quality_run2-") + 8


def test_version_is_stable_across_line_endings(tmp_path):
    body = "line one\r\nline two {{batch_records_jsonl}}\r\n"
    crlf = tmp_path / "crlf.txt"
    crlf.write_bytes(body.encode("utf-8"))
    lf = tmp_path / "lf.txt"
    lf.write_bytes(body.replace("\r\n", "\n").encode("utf-8"))

    assert load_prompt(str(crlf))[1] == load_prompt(str(lf))[1]


def test_version_changes_when_prompt_text_changes(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("x {{batch_records_jsonl}}", encoding="utf-8")
    b = tmp_path / "b.txt"
    b.write_text("y {{batch_records_jsonl}}", encoding="utf-8")

    assert load_prompt(str(a))[1] != load_prompt(str(b))[1]


def test_prompt_without_placeholder_is_rejected(tmp_path):
    p = tmp_path / "p.txt"
    p.write_text("no placeholder here", encoding="utf-8")

    with pytest.raises(ValueError):
        load_prompt(str(p))


def test_code_constants_match_the_real_prompt():
    # Guards drift: if the prompt's taxonomy or failure codes are edited, the
    # validation constants must be updated with them or this fails.
    text, _ = load_prompt()

    assert len(CATEGORY_CODES) == 15
    assert len(FAILURE_CODES) == 16
    assert set(re.findall(r"CATEGORY_[A-Z_]+", text)) == set(CATEGORY_CODES)
    assert set(re.findall(r"F\d\d_[A-Z_]+", text)) == set(FAILURE_CODES)
