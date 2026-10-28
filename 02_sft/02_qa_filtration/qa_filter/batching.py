def estimate_tokens(record: dict) -> int:
    total_chars = sum(len(m["content"]) for m in record["messages"])
    return total_chars // 4


def build_batches(records: list[dict], token_ceiling: int, max_batch_records: int) -> list[list[dict]]:
    batches: list[list[dict]] = []
    current_batch: list[dict] = []
    current_tokens = 0

    for record in records:
        record_tokens = estimate_tokens(record)

        if not current_batch:
            current_batch = [record]
            current_tokens = record_tokens
            continue

        over_token_ceiling = current_tokens + record_tokens > token_ceiling
        over_record_count_cap = len(current_batch) + 1 > max_batch_records

        if over_token_ceiling or over_record_count_cap:
            batches.append(current_batch)
            current_batch = [record]
            current_tokens = record_tokens
        else:
            current_batch.append(record)
            current_tokens += record_tokens

    if current_batch:
        batches.append(current_batch)

    return batches
