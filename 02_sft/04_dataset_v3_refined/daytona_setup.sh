#!/usr/bin/env bash
# Run inside the Daytona sandbox (needs one GPU with >= 80 GB, e.g. H100 / A100-80G).
# Downloads gpt-oss-120b onto the sandbox, serves it with vLLM, and waits until it answers.
# Then run the pipeline in the same sandbox:
#   python run_question_selection.py extract  --env <aws .env>
#   python run_question_selection.py classify --limit-batches 20     # pilot
#   python run_question_selection.py classify
#   python run_question_selection.py select --upload s3://ft-vaia/Corpus/Invest_dataset/<prefix>/
set -euo pipefail

MODEL="${SEL_MODEL_ID:-openai/gpt-oss-120b}"
PORT="${SEL_PORT:-8000}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { echo "No GPU visible"; exit 1; }

pip install -q -r requirements.txt vllm "huggingface_hub[cli]"

# ~65 GB, downloaded once into the sandbox's HF cache; later runs reuse it.
huggingface-cli download "$MODEL"

nohup vllm serve "$MODEL" \
  --port "$PORT" \
  --max-model-len 49152 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.90 \
  > vllm.log 2>&1 &

echo "waiting for vLLM on port $PORT ..."
for _ in $(seq 1 120); do
  if curl -sf "http://localhost:$PORT/v1/models" > /dev/null; then
    echo "ready: http://localhost:$PORT/v1"
    exit 0
  fi
  sleep 10
done
echo "vLLM did not become ready; see vllm.log"; tail -20 vllm.log; exit 1
