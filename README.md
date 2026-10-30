# image-gen-klein3-server

A small on-demand HTTP server for FLUX 3 Image Klein, the distilled 16-step checkpoint (not Klein Base). Jobs run one
at a time in a FIFO queue on a single worker thread. The model loads the first time it is used and stays in memory
until the server stops, and so does the optional prompt upsampler. Clients submit a task, poll it and download the PNG.

It has the same API shape as `image_gen_server` (Qwen-Image-2.1), and listens on the next port so both can be installed.

## Setup

Accept the license on [the gated repo](https://huggingface.co/black-forest-labs/flux-3-image-klein) first, and log in
with `hf auth login`. Klein takes 20.3 GB here, and the optional upsampler 18.8 GB (not gated).

```sh
hf download black-forest-labs/flux-3-image-klein --local-dir ~/models/flux-3-image-klein \
  --include "transformer/*" --include "text_encoder/*" --include "vae/*" --include "LICENSE.md" --include "NOTICE.md"
hf download black-forest-labs/flux-3-image-klein-pup --local-dir ~/models/flux-3-image-klein-pup   # optional

uv sync
cp .env.example .env   # then edit the paths and set the key
uv run --env-file .env image-gen-klein3-server
```

| Variable | Default | Meaning |
|---|---|---|
| `IMAGEGEN_API_KEY` | required | 16+ characters. Every route but `GET /health` needs `Authorization: Bearer <key>` |
| `KLEIN_PATH` | required | FLUX 3 Image Klein model directory |
| `KLEIN_UPSAMPLER_PATH` | unset | Prompt upsampler directory, for requests that send `"upsample": true`. Unset, those are a 400 |
| `KLEIN_DEVICE` | `auto` | `auto`, `mps`, `cuda` or `cpu` (auto: MPS, else CUDA, else CPU) |
| `TRANSFORMER_PRECISION` | `bf16` | `bf16`, `int8`, `fp8` or `fp8_dynamic`: transformer weights, quantized while loading (no converted copy on disk). `fp8_dynamic` also quantizes activations and uses the FP8 matmul; it needs a CUDA GPU with FP8 support |
| `TEXT_ENCODER_PRECISION` | `bf16` | `bf16`, `int8` or `fp8`: text encoder weights, same as above (the VAE and upsampler stay `bf16`) |
| `GPU_MEMORY_FRACTION` | `0` (no cap) | CUDA only: the share of the card this server may hold, so an overrun fails one render instead of a process beside it |
| `HOST` | `127.0.0.1` | Bind address |
| `PORT` | `8741` | Port |
| `PRELOAD` | empty | `klein` loads the model at startup |
| `MAX_QUEUED` | `64` | Waiting (`CREATED`) tasks allowed before `POST /v1/tasks` answers 429 |
| `TASK_TTL` | `3600` | Seconds a finished task and its PNG are kept before polls return 404 |
| `DB_PATH` | `data/imagegen.sqlite3` | SQLite file holding tasks and PNGs (relative to the working directory; created if missing) |

## API

See [doc/API.md](doc/API.md). In short: `POST /v1/tasks` with `prompt` and optional `ratio`, `width`/`height` (multiples
of 16, 64..4096), `steps` (1..100, runs at most 16, the default), `seed`, `upsample` (default off) and up to 10 base64 `images`, then poll
`GET /v1/tasks/{task_id}` and download `GET /v1/tasks/{task_id}/image`. `GET /health` reports the device, models and queue.

```sh
auth="Authorization: Bearer $IMAGEGEN_API_KEY"
id=$(curl -s localhost:8741/v1/tasks -H "$auth" -d '{"prompt": "A neon shop sign that reads KLEIN", "ratio": "16:9"}' \
  | python3 -c 'import json, sys; print(json.load(sys.stdin)["data"]["task_id"])')
while :; do
  status=$(curl -s localhost:8741/v1/tasks/$id -H "$auth" | python3 -c 'import json, sys; print(json.load(sys.stdin)["data"]["status"])')
  echo "$status"
  case $status in COMPLETED|FAILED) break;; esac
  sleep 2
done
curl -sf localhost:8741/v1/tasks/$id/image -H "$auth" -o klein.png
```

On SIGTERM/Ctrl+C the server stops taking tasks (503), finishes and stores the one rendering, then exits; queued tasks
stay `CREATED` and run again on the next start. A task that was rendering when the process died is `FAILED` with
`server restarted during render`. A second server on the same `DB_PATH` refuses to start.

## Prompts

The model takes full scene captions, and a short prompt sent as typed gives generic output. Have an LLM write the
caption before submitting: use `t2i` from [doc/system_prompts.json](doc/system_prompts.json) as its system prompt, or
`i2i` with the reference images when editing. `prompt` goes to the model as sent.

With `KLEIN_UPSAMPLER_PATH` set, a request can send `"upsample": true` to have the server write the caption itself with
`flux-3-image-klein-pup`. The caption comes back in `caption` when polling the finished task.

## Memory

The model takes about 20 GB of weights in `bf16`, and the optional upsampler about 19 GB more. Neither is evicted.

## Tests

```sh
uv run python -m unittest discover -s tests -v
```
