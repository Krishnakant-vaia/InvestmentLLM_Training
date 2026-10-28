import json
import sys

sys.path.insert(0, ".")

from qa_filter.config import load_settings
from qa_filter.profiling import profile_dataset
from qa_filter.s3_io import stream_jsonl_lines


def main():
    settings = load_settings()
    lines = stream_jsonl_lines(settings.s3_bucket, settings.source_key)
    result = profile_dataset(lines)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
