# 02.1 · SFT Q&A generation

Generates supervised fine-tuning examples from the filtered S3 corpus with gpt-oss-120b (Cloudflare Workers AI).
Each record is the source chunk plus a question (user turn) and `<think>` reasoning plus an answer (assistant turn).
One run covers all 8 capabilities (DK, EK, FR, WF, CG, AR, CL, EA).

```
run_sft_generation.py     entry point
sft_generation_core.py    chunking, parallel generation, JSONL checkpointing, S3 mirroring
sft_config.py             paths, S3 prefixes, model settings and the full generation prompt
```

## Run

```bash
pip install -r requirements.txt
cp .env.example .env                                  # Cloudflare account/token, AWS if needed
python run_sft_generation.py --limit 20               # smoke test
python run_sft_generation.py                          # full run (resumable: rerun to continue)
python run_sft_generation.py --manifest keys.jsonl    # only the S3 keys listed in the manifest
```

Output goes to `results/` (`sft_data_all_capabilities.jsonl`, an error log and a run log) and is mirrored to
`SFT_S3_OUTPUT_PREFIX`. Pause cleanly by creating `results/PAUSE_all_capabilities`; rerun the same command to resume.

## Test

```bash
python -m pytest -q tests
```
