"""Entry point for SFT Q&A generation across all 8 capabilities in one run.

Usage:
  python run_sft_generation.py                     # process everything
  python run_sft_generation.py --limit 20          # smoke test
  python run_sft_generation.py --manifest keys.jsonl  # restrict to listed S3 keys
  python run_sft_generation.py --prefix other/     # override S3_PREFIX
"""
import sft_config as config
import sft_generation_core as core

if __name__ == "__main__":
    core.main(config)
