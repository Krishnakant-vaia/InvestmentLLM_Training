import json
from collections import Counter
from typing import Iterator


def profile_dataset(lines: Iterator[str]) -> dict:
    total_records = 0
    total_user_chars = 0
    question_type_counts: Counter = Counter()
    difficulty_counts: Counter = Counter()
    malformed_json_count = 0

    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            malformed_json_count += 1
            continue

        total_records += 1
        user_content = record["messages"][1]["content"]
        total_user_chars += len(user_content)
        question_type_counts[record.get("_question_type", "UNKNOWN")] += 1
        difficulty_counts[record.get("_difficulty", "UNKNOWN")] += 1

    avg_user_content_chars = (total_user_chars / total_records) if total_records else 0.0

    return {
        "total_records": total_records,
        "avg_user_content_chars": avg_user_content_chars,
        "question_type_counts": dict(question_type_counts),
        "difficulty_counts": dict(difficulty_counts),
        "malformed_json_count": malformed_json_count,
    }
