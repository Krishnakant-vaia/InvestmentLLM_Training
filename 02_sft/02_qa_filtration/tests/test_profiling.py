import json
from qa_filter.profiling import profile_dataset


def _record(question_type, difficulty, user_len=100):
    return json.dumps({
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "x" * user_len},
            {"role": "assistant", "content": "ans"},
        ],
        "_question_type": question_type,
        "_difficulty": difficulty,
    })


def test_profile_dataset_counts_and_averages():
    lines = [
        _record("Calculation/Numerical", "Advanced", user_len=100),
        _record("Calculation/Numerical", "Applied", user_len=200),
        _record("Error Identification", "Advanced", user_len=300),
    ]

    result = profile_dataset(iter(lines))

    assert result["total_records"] == 3
    assert result["avg_user_content_chars"] == 200.0
    assert result["question_type_counts"] == {"Calculation/Numerical": 2, "Error Identification": 1}
    assert result["difficulty_counts"] == {"Advanced": 2, "Applied": 1}
    assert result["malformed_json_count"] == 0


def test_profile_dataset_counts_malformed_lines():
    lines = [_record("Calculation/Numerical", "Advanced"), "{not valid json"]

    result = profile_dataset(iter(lines))

    assert result["total_records"] == 1
    assert result["malformed_json_count"] == 1
