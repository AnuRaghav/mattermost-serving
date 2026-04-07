# Mattermost moderation assist — serving (NYU MLOps)

Production-style, lightweight **HTTP serving** for an **adaptive, human-in-the-loop** toxicity assist feature for Mattermost. This repo is scoped to the **serving** role: a FastAPI app loads a **local sklearn/joblib pipeline** (no MLflow at runtime), scores message text, applies a fixed threshold policy, and **never blocks the chat path on model failure** (fallback: allow and log).

The **baseline artifact is a placeholder** trained on a handful of hand-written examples. For your final project, export your real `pipeline.joblib` (or compatible sklearn pipeline with `predict_proba`) and point **`MODEL_PATH`** at it — no code changes required beyond configuration.

## Intended serving comparison rows (final write-up)

| experiment_id | What it represents |
|---------------|-------------------|
| `baseline_http` | Default artifact + single uvicorn worker (this `Dockerfile`) |
| `smaller_artifact` | Model-level optimization: `toxicity_pipeline_small.joblib` via `MODEL_PATH` |
| `multiworker_http` | System-level: uvicorn `--workers N` (`Dockerfile.multiworker`) |
| `best_combined` | Smaller artifact + multi-worker (or your best tuning) |
| `larger_instance_cpu` | Same stack on a larger Chameleon flavor (optional) |
| `ray_serve_http` | Optional extra credit: Ray Serve ingress (`Dockerfile.ray`), same `/health` + `/predict` API |

Record numbers in [`results/SERVING_METRICS.md`](results/SERVING_METRICS.md) and/or CSV from the benchmark script.

## API

- **`GET /health`** — `status`, `model_loaded`
- **`POST /predict`** — body matches a Mattermost-style moderation request:

```json
{
  "message_id": "mm-001",
  "text": "Hello team",
  "channel_type": "public",
  "prior_violation_count": 0
}
```

Response includes `message_id`, `model_version`, `toxicity_probability`, `risk_bucket`, `action_recommendation`, `inference_status` (`success` or `fallback`). On inference errors or missing model: `inference_status=fallback`, `toxicity_probability=null`, `risk_bucket=unknown`, `action_recommendation=allow_and_log`.

**Threshold policy** (moderator remains final authority):

| Probability | `risk_bucket` | `action_recommendation` |
|-------------|---------------|-------------------------|
| &lt; 0.50 | `low` | `no_action` |
| 0.50 – &lt; 0.85 | `medium` | `low_priority_queue` |
| 0.85 – &lt; 0.95 | `high` | `high_priority_queue` |
| ≥ 0.95 | `critical` | `highest_priority_queue` |

## Repository layout

```
mattermost-serving/
  app/                 # FastAPI app, schemas, loader, policy, settings
  scripts/             # train placeholders, benchmark
  models/              # generated .joblib (gitignored)
  sample_data/         # JSONL payloads for load tests
  results/             # benchmark CSV + metrics template
  requirements.txt
  requirements-ray.txt # Ray Serve (optional extra-credit image)
  Dockerfile           # single worker
  Dockerfile.multiworker
  Dockerfile.ray       # Ray Serve
```

## Local sanity test

From the repo root (so `app` and `models/` resolve correctly):

```bash
cd mattermost-serving
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_placeholder_model.py
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

In another terminal:

```bash
curl -s http://127.0.0.1:8000/health | python3 -m json.tool
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"message_id":"t1","text":"Thanks for the help!","channel_type":"public","prior_violation_count":0}' | python3 -m json.tool
```

### Smaller artifact (model-level comparison)

```bash
python scripts/train_smaller_placeholder_model.py
MODEL_PATH=models/toxicity_pipeline_small.joblib uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Configuration (environment)

| Variable | Default | Purpose |
|----------|---------|---------|
| `MODEL_PATH` | `models/toxicity_pipeline.joblib` | Path to joblib pipeline |
| `MODEL_VERSION` | `placeholder-1.0.0` | Reported in API responses |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Used if you wrap uvicorn yourself |

## Docker — baseline (single worker)

Train artifacts **on the host** (or copy them into `models/` before build):

```bash
python scripts/train_placeholder_model.py
docker build -t mattermost-serving:baseline -f Dockerfile .
docker run --rm -p 8000:8000 mattermost-serving:baseline
```

With explicit model path inside the container (defaults match):

```bash
docker run --rm -p 8000:8000 -e MODEL_PATH=models/toxicity_pipeline.joblib mattermost-serving:baseline
```

Mount freshly trained models without rebuilding:

```bash
docker run --rm -p 8000:8000 -v "$(pwd)/models:/app/models" mattermost-serving:baseline
```

## Docker — multi-worker (system-level comparison)

```bash
docker build -t mattermost-serving:multi -f Dockerfile.multiworker .
docker run --rm -p 8000:8000 -e UVICORN_WORKERS=4 mattermost-serving:multi
```

Each worker loads its own copy of the sklearn pipeline (higher memory use; better CPU utilization under concurrent load).

## Extra credit: Ray Serve (not FastAPI / not Triton)

This path uses **Ray Serve** with a Starlette-style deployment class that handles **`GET /health`** and **`POST /predict`** the same way as the FastAPI app, so [`scripts/benchmark.py`](scripts/benchmark.py) works unchanged. Inference logic is shared via [`app/predict_service.py`](app/predict_service.py).

**Build and run (Linux x86_64 VM — e.g. Chameleon):** Ray publishes manylinux wheels; on some ARM laptops `pip install ray` may fail.

```bash
python scripts/train_placeholder_model.py
docker build -t mattermost-serving:ray -f Dockerfile.ray .
docker run --rm --shm-size=1g -p 8000:8000 mattermost-serving:ray
```

`--shm-size=1g` avoids Ray instability from Docker’s default small `/dev/shm`.

Optional environment variables (inside the container):

| Variable | Default | Purpose |
|----------|---------|---------|
| `RAY_SERVE_HOST` | `0.0.0.0` | HTTP bind host |
| `RAY_SERVE_PORT` | `8000` | HTTP port |
| `RAY_SERVE_NUM_REPLICAS` | `1` | Number of Serve replicas (scale-out comparison) |

Example benchmark row label:

```bash
python scripts/benchmark.py --url http://127.0.0.1:8000 --concurrency 5 --requests 1000 \
  --csv results/benchmark_runs.csv --label ray_serve_http
```

**PDF / write-up (meaningful improvement — adapt to what you measure):** Ray Serve gives a **deployment-centric** serving model (replicas, optional resource options, same process cluster as future Ray Data/Train if you grow the system) distinct from “a single FastAPI + uvicorn process.” A concrete example is turning **`RAY_SERVE_NUM_REPLICAS`** above 1 on a multi-core Chameleon VM and comparing tail latency and throughput under concurrent `/predict` load against the baseline `Dockerfile` row—then explain whether replica overhead or queueing dominated on your instance.

## Benchmark

Requires the same venv / `pip install -r requirements.txt` (uses `httpx`, `numpy`):

```bash
# Service running on :8000
python scripts/benchmark.py --url http://127.0.0.1:8000 --concurrency 5 --requests 500 --label baseline_http
```

Append to a shared CSV:

```bash
python scripts/benchmark.py --url http://127.0.0.1:8000 --concurrency 10 --requests 2000 \
  --csv results/benchmark_runs.csv --label multiworker_http
```

Output includes **p50 / p95 / p99** latency (ms), **throughput** (successful req/s), **error rate**, and **concurrency** tested.

## Chameleon Cloud (Ubuntu VM)

SSH into your instance, then:

```bash
sudo apt-get update
sudo apt-get install -y git ca-certificates curl
# Docker Engine (official convenience script — review https://docs.docker.com/engine/install/ubuntu/ for production)
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"
# Log out and back in so group membership applies, then:
git clone <YOUR_REPO_URL> mattermost-serving
cd mattermost-serving
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_placeholder_model.py

docker build -t mattermost-serving:baseline -f Dockerfile .
docker run -d --name mm-serve -p 8000:8000 mattermost-serving:baseline

python scripts/benchmark.py --url http://127.0.0.1:8000 --concurrency 5 --requests 1000 --label baseline_http
```

Open the security group / `ufw` for TCP **8000** if you benchmark from another machine (use the VM’s floating IP in `--url`).

### Course rule: experiments in containers

Run the **server** in Docker on Chameleon as above; run the **benchmark** from the same VM (host Python) or from a second VM hitting the containerized service — both satisfy “serving experiments on Chameleon” as long as the API under test is containerized.

## Swapping in the real model

1. Train/export your sklearn `Pipeline` elsewhere; save with `joblib.dump(..., "pipeline.joblib")`.
2. Copy the file into `models/` on the VM (or bake into the image).
3. Set `MODEL_PATH` (and `MODEL_VERSION`) to match.

The loader only requires `predict_proba` and uses **column 1** as the toxic (positive) class probability.

## License / course use

Built for an NYU MLOps course serving submission; adapt as needed for your team’s policies.
