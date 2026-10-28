from qa_filter.batching import build_batches, estimate_tokens


def _record(char_len):
    return {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "x" * char_len},
            {"role": "assistant", "content": ""},
        ]
    }


def test_estimate_tokens_uses_chars_over_four():
    record = _record(400)
    assert estimate_tokens(record) == 100


def test_build_batches_packs_whole_records_under_token_ceiling():
    records = [_record(400) for _ in range(5)]  # 100 tokens each

    batches = build_batches(records, token_ceiling=250, max_batch_records=50)

    assert sum(len(b) for b in batches) == 5
    for batch in batches:
        batch_tokens = sum(estimate_tokens(r) for r in batch)
        assert batch_tokens <= 250
    assert [len(b) for b in batches] == [2, 2, 1]


def test_build_batches_gives_oversized_record_its_own_batch():
    small = _record(40)       # 10 tokens
    huge = _record(4000)      # 1000 tokens, exceeds ceiling alone

    # Packing is strictly sequential (never reorders records to group small ones
    # together), so huge -- appearing between the two smalls -- forces three
    # batches: the first small alone (huge can't join it), huge alone (too big to
    # join anything), then the trailing small alone (huge already closed its batch).
    batches = build_batches([small, huge, small], token_ceiling=250, max_batch_records=50)

    assert batches == [[small], [huge], [small]]


def test_build_batches_never_splits_a_record():
    records = [_record(100) for _ in range(3)]

    batches = build_batches(records, token_ceiling=1000000, max_batch_records=50)

    assert len(batches) == 1
    assert batches[0] == records
    for record in records:
        assert record in batches[0]


def test_build_batches_caps_at_max_batch_records_even_under_token_ceiling():
    # 60 tiny records, each 1 token -- token_ceiling alone would pack all 60 into
    # one batch, but max_batch_records=50 must split it into 50 + 10 regardless.
    records = [_record(4) for _ in range(60)]

    batches = build_batches(records, token_ceiling=1000000, max_batch_records=50)

    assert [len(b) for b in batches] == [50, 10]
    assert sum(len(b) for b in batches) == 60


def test_build_batches_record_count_cap_and_token_ceiling_combine_whichever_hits_first():
    # token_ceiling of 300 with 100-token records allows only 3 per batch, well
    # under the 50-record cap -- the tighter limit (tokens, here) always wins.
    records = [_record(400) for _ in range(7)]  # 100 tokens each

    batches = build_batches(records, token_ceiling=300, max_batch_records=50)

    assert [len(b) for b in batches] == [3, 3, 1]
