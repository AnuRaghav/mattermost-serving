#!/usr/bin/env python3
"""Load-test POST /predict: latency percentiles, throughput, error rate."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

import httpx
import numpy as np


@dataclass
class RunStats:
    concurrency: int
    total_requests: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    throughput_rps: float
    errors: int
    error_rate: float
    wall_seconds: float


def load_payloads(path: Path) -> List[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


async def run_benchmark(
    base_url: str,
    payloads: List[dict[str, Any]],
    total_requests: int,
    concurrency: int,
) -> tuple[list[float], int]:
    sem = asyncio.Semaphore(concurrency)
    url = base_url.rstrip("/") + "/predict"
    client_timeout = httpx.Timeout(60.0, connect=10.0)

    async with httpx.AsyncClient(timeout=client_timeout) as client:

        async def one_request(i: int) -> tuple[float, int]:
            body = payloads[i % len(payloads)]
            async with sem:
                t0 = time.perf_counter()
                try:
                    r = await client.post(url, json=body)
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    err = 1 if r.status_code >= 400 else 0
                    return dt_ms, err
                except Exception:
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    return dt_ms, 1

        rows = await asyncio.gather(*(one_request(i) for i in range(total_requests)))

    latencies_ms = [r[0] for r in rows]
    errors = sum(r[1] for r in rows)
    return latencies_ms, errors


def summarize(
    latencies_ms: list[float],
    errors: int,
    wall_seconds: float,
    concurrency: int,
    total_requests: int,
) -> RunStats:
    arr = np.array(latencies_ms, dtype=np.float64)
    ok = total_requests - errors
    throughput = ok / wall_seconds if wall_seconds > 0 else 0.0
    error_rate = errors / total_requests if total_requests else 0.0
    return RunStats(
        concurrency=concurrency,
        total_requests=total_requests,
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        throughput_rps=throughput,
        errors=errors,
        error_rate=error_rate,
        wall_seconds=wall_seconds,
    )


def write_csv(path: Path, stats: RunStats, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "concurrency": stats.concurrency,
        "total_requests": stats.total_requests,
        "p50_ms": round(stats.p50_ms, 3),
        "p95_ms": round(stats.p95_ms, 3),
        "p99_ms": round(stats.p99_ms, 3),
        "throughput_rps": round(stats.throughput_rps, 3),
        "errors": stats.errors,
        "error_rate": round(stats.error_rate, 6),
        "wall_seconds": round(stats.wall_seconds, 3),
    }
    write_header = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark POST /predict")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000",
        help="Base URL of the API (no trailing path)",
    )
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrent in-flight requests")
    parser.add_argument("--requests", type=int, default=200, help="Total requests to send")
    parser.add_argument(
        "--payloads",
        type=Path,
        default=None,
        help="Path to JSONL file (default: sample_data/requests.jsonl under repo root)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Append results row to this CSV (default: results/benchmark_<timestamp>.csv)",
    )
    parser.add_argument(
        "--label",
        default="run",
        help="Short label stored in CSV (e.g. baseline_http, multiworker_http)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    payload_path = args.payloads or (repo_root / "sample_data" / "requests.jsonl")
    payloads = load_payloads(payload_path)

    t_start = time.perf_counter()
    latencies, errors = asyncio.run(
        run_benchmark(args.url, payloads, args.requests, args.concurrency)
    )
    wall = time.perf_counter() - t_start
    stats = summarize(latencies, errors, wall, args.concurrency, args.requests)

    print(f"Concurrency:        {stats.concurrency}")
    print(f"Total requests:     {stats.total_requests}")
    print(f"Wall time (s):      {stats.wall_seconds:.3f}")
    print(f"p50 latency (ms):   {stats.p50_ms:.3f}")
    print(f"p95 latency (ms):   {stats.p95_ms:.3f}")
    print(f"p99 latency (ms):   {stats.p99_ms:.3f}")
    print(f"Throughput (req/s): {stats.throughput_rps:.3f}")
    print(f"Errors:             {stats.errors}")
    print(f"Error rate:         {stats.error_rate:.4%}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_csv = args.csv or (repo_root / "results" / f"benchmark_{ts}.csv")
    write_csv(out_csv, stats, args.label)
    print(f"Wrote CSV row to {out_csv}")


if __name__ == "__main__":
    main()
