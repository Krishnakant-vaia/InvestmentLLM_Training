#!/usr/bin/env bash
# Pod-side setup for pretune_dump.py. Run this once after connecting to the pod.
set -euo pipefail

cd "$(dirname "$0")/.."

# Prefer deploy/.env (transferred via scp, never typed into this terminal) --
# falls back to whatever's already exported if no .env is present, e.g. from
# a pre-existing ~/.aws/credentials.
if [[ -f deploy/.env ]]; then
    echo "Loading credentials from deploy/.env..."
    set -a
    # shellcheck disable=SC1091
    source deploy/.env
    set +a
fi

echo "Installing dependencies..."
pip install -r requirements.txt

if [[ -z "${AWS_ACCESS_KEY_ID:-}" || -z "${AWS_SECRET_ACCESS_KEY:-}" ]]; then
    echo "ERROR: AWS_ACCESS_KEY_ID and/or AWS_SECRET_ACCESS_KEY are not set."
    echo "Copy deploy/.env.example to deploy/.env, fill in real values LOCALLY"
    echo "(not on the pod), then scp deploy/.env to the pod before running this."
    exit 1
fi

echo "AWS credentials found."
echo ""
echo "Setup complete. Next steps:"
echo ""
echo "  1) Generate the real doc-keys list:"
echo "     python data_prep/list_doc_keys.py --prefix Corpus/Invest_dataset/082126_fliteredextractionrun2/ --out doc_keys.txt"
echo ""
echo "  2) Run the real pipeline (add --resume-log for a rented, interruptible pod):"
echo "     python data_prep/pretune_dump.py --doc-keys-from doc_keys.txt --s3-prefix <your output prefix> --resume-log resume.log"
