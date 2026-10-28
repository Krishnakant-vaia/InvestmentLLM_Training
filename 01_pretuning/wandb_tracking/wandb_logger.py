"""Standalone W&B dataset/run tracking for the financial pre-tuning pipeline.

Deliberately independent of pretune_dump.py — no import of it, no shared
state. It knows nothing about how a dataset was produced; it only knows how
to log a run's config/stats to W&B and register output files as a versioned
Artifact referencing their S3 locations. Wire it into any pipeline by
calling log_run_to_wandb() with that pipeline's own config/report/URIs.

The defaults below mirror pretune_dump.py's current tunable constants as of
2026-08-31, duplicated on purpose for independence — if that script's
values change, update these to match by hand; there is no shared import to
keep them in sync automatically.
"""

import argparse
import json

MODEL_NAME = "Qwen/Qwen3.5-9B-Base"
CHUNK_TARGET_TOKENS = 8192
CHUNK_OVERLAP_TOKENS = 0
MAX_SECTION_TOKENS = 8192
PACK_SEQ_LEN = 8192
PACK_SEPARATOR = "<|endoftext|>"
VAL_FRACTION = 0.005
SPLIT_SEED = 42


def build_wandb_config(**overrides) -> dict:
    """Returns the default tunable-constant config, with any keyword
    overrides applied on top — lets a caller log the config it actually
    ran with, even if it differs from these defaults."""
    config = {
        "model_name": MODEL_NAME,
        "chunk_target_tokens": CHUNK_TARGET_TOKENS,
        "chunk_overlap_tokens": CHUNK_OVERLAP_TOKENS,
        "max_section_tokens": MAX_SECTION_TOKENS,
        "pack_seq_len": PACK_SEQ_LEN,
        "pack_separator": PACK_SEPARATOR,
        "val_fraction": VAL_FRACTION,
        "split_seed": SPLIT_SEED,
    }
    config.update(overrides)
    return config


def validate_report(report) -> dict:
    """Raises ValueError if report-json didn't contain a JSON object. A list,
    string, or number is valid JSON but not something run.log() can sensibly
    handle, and W&B's own error for that case is not obvious to debug."""
    if not isinstance(report, dict):
        raise ValueError(
            f"report-json must contain a JSON object (dict), got {type(report).__name__}"
        )
    return report


def validate_artifact_uris(uris: list[str]) -> None:
    """Raises ValueError on the first URI that isn't an s3:// URI, so a typo
    (local path, http://, missing prefix) fails fast instead of being baked
    silently into the Artifact and only discovered when someone tries to
    resolve it later."""
    for uri in uris:
        if not uri.startswith("s3://"):
            raise ValueError(f"--artifact-uri must be an s3:// URI, got: {uri!r}")


def log_run_to_wandb(
    wandb_module,
    project: str,
    config: dict,
    report: dict,
    artifact_uris: list[str],
    artifact_name: str = "pretuning-dataset",
) -> None:
    """Logs a data-prep run's config and stats to W&B, and — if artifact_uris
    is non-empty (a real S3 upload happened) — registers the output as a
    versioned dataset Artifact referencing those S3 locations rather than
    re-uploading the content into W&B's own storage.

    artifact_name lets unrelated corpus slices/configs get their own Artifact
    lineage instead of all piling up as versions of one fixed name — pass a
    distinguishing name for anything other than the default pipeline.

    The logging/artifact work is wrapped in try/finally so the run is always
    finished — on the exception path with exit_code=1 — rather than left
    "running" forever on the W&B dashboard if something fails partway
    through (bad report data, a network drop, a bad artifact reference)."""
    run = wandb_module.init(project=project, config=config, job_type="data-prep")
    try:
        run.log(report)
        if artifact_uris:
            artifact = wandb_module.Artifact(name=artifact_name, type="dataset")
            for uri in artifact_uris:
                artifact.add_reference(uri)
            run.log_artifact(artifact)
    except Exception:
        run.finish(exit_code=1)
        raise
    else:
        run.finish(exit_code=0)


def main():
    parser = argparse.ArgumentParser(
        description="Log a pre-tuning data-prep run's config/stats to W&B, "
                     "and optionally register output files as a dataset Artifact.",
    )
    parser.add_argument("--project", default="vaia-financial-pretuning", help="W&B project name")
    parser.add_argument(
        "--report-json",
        required=True,
        help="Path to a JSON file containing the stats report to log (e.g. produced by "
             "pretune_dump.py's generate_stats_report, saved separately as JSON)",
    )
    parser.add_argument(
        "--artifact-uri", action="append", default=[],
        help="An S3 URI to register as part of the dataset Artifact. Repeatable "
             "(e.g. --artifact-uri s3://bucket/train.jsonl --artifact-uri s3://bucket/val.jsonl). "
             "Omit entirely to log only config/stats with no Artifact.",
    )
    parser.add_argument(
        "--dataset-name", default="pretuning-dataset",
        help="Name for the W&B dataset Artifact (default: pretuning-dataset). Give unrelated "
             "corpus slices/configs distinct names so their Artifact versions don't pile up "
             "together under one lineage.",
    )
    args = parser.parse_args()

    with open(args.report_json, "r", encoding="utf-8") as f:
        report = json.load(f)
    validate_report(report)
    validate_artifact_uris(args.artifact_uri)

    import wandb
    log_run_to_wandb(
        wandb, args.project, build_wandb_config(), report, args.artifact_uri,
        artifact_name=args.dataset_name,
    )


if __name__ == "__main__":
    main()
