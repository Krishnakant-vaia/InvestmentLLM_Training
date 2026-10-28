"""Offline batch inference on Modal (vLLM): JSONL of questions in, JSONL of answers out.

One-time setup (uploads the merged model to a Modal volume):
    modal volume put sft-models outputs/merged merged
Run:
    modal run 05_production_deployment/batch_inference.py --input questions.jsonl --output answers.jsonl
Input rows: {"question": "...", "system": "..."(optional)}  -> output adds "answer".
"""
import json

import modal

GPU = "A100-80GB"
VOLUME_NAME, MODEL_DIR = "sft-models", "/models/merged"  # keep in sync with config.yaml `deployment`
MAX_MODEL_LEN = 262144  # prompt + answer per request; the model's native context (see serve.py)

app = modal.App("sft-batch-inference")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install("vllm", "transformers")


@app.cls(image=image, gpu=GPU, volumes={"/models": volume}, timeout=3600, scaledown_window=300)
class Generator:
    @modal.enter()
    def load(self):
        from vllm import LLM
        self.llm = LLM(model=MODEL_DIR, dtype="bfloat16", max_model_len=MAX_MODEL_LEN, trust_remote_code=True)

    @modal.method()
    def generate(self, rows: list[dict], max_tokens: int = 1024, temperature: float = 0.0) -> list[str]:
        from vllm import SamplingParams
        # Chat prompts are built with the model's own template, thinking disabled.
        chats = [
            ([{"role": "system", "content": r["system"]}] if r.get("system") else [])
            + [{"role": "user", "content": r["question"]}]
            for r in rows
        ]
        params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        outs = self.llm.chat(chats, params, chat_template_kwargs={"enable_thinking": False})
        return [o.outputs[0].text.strip() for o in outs]


@app.local_entrypoint()
def main(input: str, output: str, batch_size: int = 256, max_tokens: int = 1024):
    with open(input, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    gen = Generator()
    with open(output, "w", encoding="utf-8") as out:
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            for row, ans in zip(chunk, gen.generate.remote(chunk, max_tokens)):
                out.write(json.dumps({**row, "answer": ans}, ensure_ascii=False) + "\n")
            print(f"{min(i + batch_size, len(rows))}/{len(rows)}")
