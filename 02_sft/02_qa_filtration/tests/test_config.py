from qa_filter.config import JUDGE_PROMPT_TEMPLATE, load_settings


def test_load_settings_defaults(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct123")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "token123")
    for var in ["S3_BUCKET", "SOURCE_KEY", "OUTPUT_PREFIX", "MODEL_ID",
                "REASONING_EFFORT", "MAX_OUTPUT_TOKENS", "REQUEST_TIMEOUT_SECONDS",
                "MAX_RETRIES", "RETRY_BASE_SECONDS", "BATCH_TOKEN_CEILING",
                "MAX_BATCH_RECORDS", "SHARD_SIZE", "MAX_WORKERS", "RUN_STATE_DIR",
                "JUDGE_PROMPT_VERSION"]:
        monkeypatch.delenv(var, raising=False)

    settings = load_settings()

    assert settings.s3_bucket == "ft-vaia"
    assert settings.source_key == "Corpus/Invest_dataset/09032026_SFTQ&Apairs/sft_data_all_capabilities.jsonl"
    assert settings.output_prefix == "Corpus/Invest_dataset/09032026_SFTQ&Apairs/"
    assert settings.cloudflare_account_id == "acct123"
    assert settings.cloudflare_api_token == "token123"
    assert settings.model_id == "@cf/openai/gpt-oss-120b"
    assert settings.reasoning_effort == "low"
    assert settings.max_output_tokens == 6000
    assert settings.request_timeout_seconds == 180
    assert settings.max_retries == 4
    assert settings.retry_base_seconds == 2
    assert settings.batch_token_ceiling == 24000
    assert settings.max_batch_records == 50
    assert settings.shard_size == 5000
    assert settings.max_workers == 10
    assert settings.run_state_dir == "./run_state"
    assert settings.judge_prompt_version == "gepa_strict_v2"


def test_load_settings_missing_credentials(monkeypatch):
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)

    try:
        load_settings()
        assert False, "expected ValueError"
    except ValueError as e:
        assert "CLOUDFLARE_ACCOUNT_ID" in str(e) or "CLOUDFLARE_API_TOKEN" in str(e)


def test_judge_prompt_template_has_batch_placeholder():
    assert "{{batch_records_json}}" in JUDGE_PROMPT_TEMPLATE


def test_judge_prompt_template_documents_output_contract():
    assert '"verdicts"' in JUDGE_PROMPT_TEMPLATE
    assert '"id"' in JUDGE_PROMPT_TEMPLATE
    assert '"verdict"' in JUDGE_PROMPT_TEMPLATE
    assert '"reason"' in JUDGE_PROMPT_TEMPLATE
