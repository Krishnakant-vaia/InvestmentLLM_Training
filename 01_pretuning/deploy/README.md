# Running pretune_dump.py on RunPod

## Why no GPU / Dockerfile

pretune_dump.py does S3 I/O and CPU-based tokenization only — it never touches a GPU.
Pick the cheapest available Pod type; a GPU pod is unnecessary cost for this specific
script (GPU pods are needed later, for training on train_pretune.py's output, not here).

There's no Dockerfile because this script has no unusual system dependencies — RunPod's
existing base Pod images (Python + pip) are enough.

## SSH key

Save the pod's PEM/private key to `deploy/keys/` (this folder, not just `~/.ssh/`) —
keeps everything for a given pod together in the project rather than scattered in the
system-wide SSH config. Reference it directly in your `ssh`/`scp` commands:

```
ssh -i deploy/keys/<pod-name>.pem -p <port> root@<host>
```

## AWS credentials — via deploy/.env, not the SSH terminal

Typing or pasting multi-line credential blocks directly into an SSH session is fragile —
terminal paste handling can silently corrupt the value (this happened during an earlier
pod's setup: a heredoc-pasted access key got mangled with stray characters, and boto3
failed with `InvalidAccessKeyId` / `ConfigParseError` until this was fixed). Avoid that
class of problem entirely:

1. Copy `deploy/.env.example` to `deploy/.env` and fill in the real values **locally**,
   in a normal text editor on your own machine — not by typing into the pod's terminal.
2. Transfer the completed file to the pod:
   ```
   scp -i deploy/keys/<pod-name>.pem -P <port> deploy/.env root@<host>:/workspace/pretuning/deploy/.env
   ```
3. `deploy/setup_and_run.sh` sources `deploy/.env` automatically if present.

`deploy/.env` is never committed, printed, or transcribed anywhere else — it's a plain
file transfer, not a value anyone has to read or retype.

## Setup

1. Connect to your pod:
   ```
   ssh -i deploy/keys/<pod-name>.pem -p <port> root@<host>
   ```
2. Make sure `deploy/.env` (see above) has already been copied onto the pod.
3. Run the setup script:
   ```
   bash deploy/setup_and_run.sh
   ```
   This loads `deploy/.env`, installs dependencies, and checks credentials are present.
   It prints the next two commands to run rather than running them for you, since the
   S3 prefix and resume-log path are per-run choices.

## Running the real pipeline

```
python data_prep/list_doc_keys.py --prefix Corpus/Invest_dataset/082126_fliteredextractionrun2/ --out doc_keys.txt
python data_prep/pretune_dump.py --doc-keys-from doc_keys.txt --s3-prefix <your output prefix> --resume-log resume.log
```

`--resume-log` is recommended for this run: a rented pod can be interrupted, and
`--resume-log` lets you re-run the same command afterward without reprocessing
documents from a prior fully-completed run. Note it does not resume *mid*-run
(see pretune_dump.py's own --resume-log help text) — only "start over" vs.
"skip a fully-completed prior run" are distinguished.
