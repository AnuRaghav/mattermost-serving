# Serving experiment log

Copy numbers from `scripts/benchmark.py` output or from CSV rows under `results/`. Fill in after each Chameleon run (same VM specs when comparing rows).

| experiment_id          | instance | concurrency | p50_ms | p95_ms | p99_ms | throughput_rps | error_rate | notes |
|------------------------|----------|------------:|-------:|-------:|-------:|---------------:|-----------:|-------|
| baseline_http          |          |             |        |        |        |                |            |       |
| smaller_artifact       |          |             |        |        |        |                |            |       |
| multiworker_http       |          |             |        |        |        |                |            |       |
| best_combined          |          |             |        |        |        |                |            |       |
| larger_instance_cpu    |          |             |        |        |        |                |            |       |

Optional: append machine-readable rows with `--csv results/benchmark_runs.csv --label <experiment_id>`.
