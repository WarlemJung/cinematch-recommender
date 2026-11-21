"""Load testing, latency percentile profiling, and memory footprint benchmarking CLI."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time
import tracemalloc
from typing import Any
import numpy as np
import psutil

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from cinematch.config import AppConfig  # noqa: E402
from cinematch.main import create_app  # noqa: E402


def _summarize_latencies(latencies_ms: list[float], elapsed_s: float, errors: int) -> dict[str, Any]:
    arr = np.array(latencies_ms, dtype=float) if latencies_ms else np.array([0.0])
    total = len(latencies_ms) + errors
    return {
        "total_requests": total,
        "errors": errors,
        "error_rate_pct": round((errors / max(1, total)) * 100.0, 3),
        "elapsed_seconds": round(elapsed_s, 3),
        "throughput_rps": round(total / max(1e-6, elapsed_s), 2),
        "latency_ms": {
            "mean": round(float(np.mean(arr)), 2),
            "p50": round(float(np.percentile(arr, 50)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
            "p99": round(float(np.percentile(arr, 99)), 2),
            "max": round(float(np.max(arr)), 2),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CineMatch cold-start, concurrency, and memory profiling.")
    parser.add_argument("--seq", type=int, default=100, help="Sequential requests count.")
    parser.add_argument("--conc", type=int, default=200, help="Concurrent requests count.")
    parser.add_argument("--threads", type=int, default=8, help="Concurrent worker threads.")
    parser.add_argument("--sustained", type=int, default=500, help="Sustained mixed load requests count.")
    parser.add_argument("--output-json", type=str, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    proc = psutil.Process()
    rss_before_mb = proc.memory_info().rss / (1024 * 1024)

    tracemalloc.start()
    t_boot_start = time.perf_counter()
    config = AppConfig.from_env()
    app = create_app(config=config)
    boot_seconds = time.perf_counter() - t_boot_start
    _curr_mem, peak_trace_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    rss_after_boot_mb = proc.memory_info().rss / (1024 * 1024)

    # 1. First recommendation latency
    client = app.test_client()
    t_first = time.perf_counter()
    first_res = client.get("/api/recommendations/hybrid/862?top_n=12")
    first_rec_ms = (time.perf_counter() - t_first) * 1000.0

    endpoints = [
        "/api/health",
        "/api/trending?top_n=12&page=1",
        "/api/genres",
        "/api/movies/search?q=Toy+Story",
        "/api/movies/862",
        "/api/recommend/862?top_n=12",
        "/api/recommend?title=Heat&top_n=12",
        "/api/recommendations/hybrid/862?top_n=12",
        "/api/recommendations/hybrid?title=Pulp+Fiction&top_n=12",
        "/api/recommendations/hybrid?title=Toy+Story&genre=Animation&min_year=1990",
    ]

    # 2. Sequential workload
    seq_latencies: list[float] = []
    seq_errors = 0
    t_seq_start = time.perf_counter()
    for idx in range(args.seq):
        ep = endpoints[idx % len(endpoints)]
        t0 = time.perf_counter()
        resp = client.get(ep)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if resp.status_code == 200:
            seq_latencies.append(dt_ms)
        else:
            seq_errors += 1
    seq_elapsed = time.perf_counter() - t_seq_start

    # 3. Concurrent workload (each thread uses its own Flask test_client safely)
    def _worker_call(ep: str) -> tuple[bool, float]:
        c = app.test_client()
        t0 = time.perf_counter()
        resp = c.get(ep)
        return (resp.status_code == 200, (time.perf_counter() - t0) * 1000.0)

    conc_latencies: list[float] = []
    conc_errors = 0
    t_conc_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        conc_results = list(ex.map(_worker_call, [endpoints[i % len(endpoints)] for i in range(args.conc)]))
    conc_elapsed = time.perf_counter() - t_conc_start
    for ok, lat in conc_results:
        if ok:
            conc_latencies.append(lat)
        else:
            conc_errors += 1

    # 4. Sustained mixed load
    sust_latencies: list[float] = []
    sust_errors = 0
    t_sust_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        sust_results = list(ex.map(_worker_call, [endpoints[i % len(endpoints)] for i in range(args.sustained)]))
    sust_elapsed = time.perf_counter() - t_sust_start
    for ok, lat in sust_results:
        if ok:
            sust_latencies.append(lat)
        else:
            sust_errors += 1

    rss_after_load_mb = proc.memory_info().rss / (1024 * 1024)

    report = {
        "cold_start": {
            "boot_seconds": round(boot_seconds, 3),
            "rss_before_boot_mb": round(rss_before_mb, 2),
            "rss_after_boot_mb": round(rss_after_boot_mb, 2),
            "boot_rss_delta_mb": round(rss_after_boot_mb - rss_before_mb, 2),
            "tracemalloc_peak_mb": round(peak_trace_bytes / (1024 * 1024), 2),
            "first_hybrid_recommendation_ms": round(first_rec_ms, 2),
            "first_hybrid_status": first_res.status_code,
        },
        "sequential": _summarize_latencies(seq_latencies, seq_elapsed, seq_errors),
        "concurrent": {
            "threads": args.threads,
            **_summarize_latencies(conc_latencies, conc_elapsed, conc_errors),
        },
        "sustained": {
            "threads": args.threads,
            **_summarize_latencies(sust_latencies, sust_elapsed, sust_errors),
        },
        "memory_after_load": {
            "rss_after_load_mb": round(rss_after_load_mb, 2),
            "rss_growth_under_load_mb": round(rss_after_load_mb - rss_after_boot_mb, 2),
        },
    }

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
