# Submission checklist (NYU MLOps — serving role)

Use this file to pick **portal uploads** and to paste **“which files go with which experiment row”** into assignment text fields. All measured runs should be from **Chameleon** (server in Docker; benchmark from the VM or another client hitting the container).

## Course upload buckets (typical rubric)

### 1) Serving code / serving configuration

Upload these from this repo:

| File | Role |
|------|------|
| [Dockerfile](Dockerfile) | Baseline: single uvicorn worker |
| [Dockerfile.multiworker](Dockerfile.multiworker) | System comparison: uvicorn multi-worker |
| [Dockerfile.ray](Dockerfile.ray) | Optional extra credit: Ray Serve |
| [requirements.txt](requirements.txt) | Python deps (FastAPI path) |
| [requirements-ray.txt](requirements-ray.txt) | Ray Serve deps (only if submitting Ray) |
| [app/main.py](app/main.py) | FastAPI app: `GET /health`, `POST /predict` |
| [app/predict_service.py](app/predict_service.py) | Shared inference + graceful fallback |
| [app/model_loader.py](app/model_loader.py) | Loads `joblib` pipeline; `predict_proba` |
| [app/policy.py](app/policy.py) | Thresholds → risk bucket + action suggestion |
| [app/schemas.py](app/schemas.py) | Pydantic request/response models |
| [app/settings.py](app/settings.py) | `MODEL_PATH`, `MODEL_VERSION`, etc. |
| [app/ray_serve_app.py](app/ray_serve_app.py) | Ray Serve HTTP ingress (extra credit only) |

**Artifact generation (optional to include as “serving pipeline” scripts):**

| File | Role |
|------|------|
| [scripts/train_placeholder_model.py](scripts/train_placeholder_model.py) | Writes `models/toxicity_pipeline.joblib` |
| [scripts/train_smaller_placeholder_model.py](scripts/train_smaller_placeholder_model.py) | Writes `models/toxicity_pipeline_small.joblib` |

Note: `models/*.joblib` are **gitignored**; graders may not need binaries if you describe training + paths. If your course requires the artifact, zip it or use `git add -f` per instructor rules.

### 2) Scripts for evaluating a serving configuration

Upload:

| File | Role |
|------|------|
| [scripts/benchmark.py](scripts/benchmark.py) | Async load test; p50/p95/p99, throughput, errors |
| [sample_data/requests.jsonl](sample_data/requests.jsonl) | Request bodies for the benchmark |

**Evidence of runs:**

| File | Role |
|------|------|
| [results/benchmark_runs.csv](results/benchmark_runs.csv) | Append-only benchmark output (`--csv …`) |
| [results/SERVING_METRICS.md](results/SERVING_METRICS.md) | Human-readable comparison table template |

## Paste-ready text: “which table rows map to which setup”

Edit labels if your CSV uses different `--label` strings.

- **baseline_http** — Image: `Dockerfile` (tag e.g. `mattermost-serving:baseline`). Model: default `MODEL_PATH=models/toxicity_pipeline.joblib` (train via `scripts/train_placeholder_model.py` before `docker build`, or bind-mount `models/`). Process: single uvicorn worker.

- **smaller_artifact** — Same image as baseline (`Dockerfile`). Model: `MODEL_PATH=models/toxicity_pipeline_small.joblib` (from `scripts/train_smaller_placeholder_model.py`). Env overrides at `docker run`: `-e MODEL_PATH=…` and optionally `-e MODEL_VERSION=…`.

- **multiworker_http** — Image: `Dockerfile.multiworker`. Model: baseline artifact. Env: e.g. `-e UVICORN_WORKERS=4`.

- **best_combined** — Image: `Dockerfile.multiworker`. Model: small artifact via `-e MODEL_PATH=models/toxicity_pipeline_small.joblib` (and matching `UVICORN_WORKERS`).

- **larger_instance_cpu** (optional) — Same as one of the above, but on a larger Chameleon flavor; note **instance type** in the table.

- **ray_serve_http** (optional extra credit) — Image: `Dockerfile.ray`. Model: default or overridden with `-e MODEL_PATH`. Recommend `docker run --shm-size=1g …`. Replicas: optional `-e RAY_SERVE_NUM_REPLICAS=…`.

## How to benchmark (repeat per row)

1. **Train** (on the VM, repo root):

   ```bash
   python scripts/train_placeholder_model.py
   python scripts/train_smaller_placeholder_model.py   # if needed for that row
   ```

2. **Build** the image for that row (`Dockerfile`, `Dockerfile.multiworker`, or `Dockerfile.ray`).

3. **Run** container with `-p 8000:8000` and the **env vars** for that row. Free the port between runs (`docker ps`, `docker stop …`) or use `-p 8001:8000` and change `--url` below.

4. **Activate venv** and install deps if needed:

   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

5. **Run** [scripts/benchmark.py](scripts/benchmark.py):

   ```bash
   python scripts/benchmark.py \
     --url http://127.0.0.1:8000 \
     --concurrency 5 \
     --requests 1000 \
     --csv results/benchmark_runs.csv \
     --label baseline_http
   ```

   Match `--concurrency` / `--requests` to what you report. Use a distinct `--label` per configuration.

6. Copy numbers into [results/SERVING_METRICS.md](results/SERVING_METRICS.md) and/or attach [results/benchmark_runs.csv](results/benchmark_runs.csv).

## Quick Docker build tags (suggested)

```bash
docker build -t mattermost-serving:baseline -f Dockerfile .
docker build -t mattermost-serving:multi -f Dockerfile.multiworker .
docker build -t mattermost-serving:ray -f Dockerfile.ray .
```

## Optional: Ray Serve run reminder

```bash
docker run --rm --shm-size=1g -p 8000:8000 mattermost-serving:ray
```

See [README.md](README.md) for env vars (`RAY_SERVE_HOST`, `RAY_SERVE_PORT`, `RAY_SERVE_NUM_REPLICAS`, `MODEL_PATH`, `MODEL_VERSION`).
