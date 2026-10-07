"""Steps 3-4: measure system cost of both baselines and check deployment budgets.

Each model is measured in its own subprocess so that peak RSS of one model is
not polluted by allocations left over from the other.
"""
import argparse
import gc
import json
import subprocess
import sys
import threading
import time
import tracemalloc
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil

from common import RESULTS, load_split, make_models, set_seeds

TRAIN_RUNS = 5
INFER_RUNS = 100
MIB = 1024 * 1024

# Upper limits per target (None = no upper limit). Memory in MB, latency in ms, size in bytes.
BUDGETS = {
    "Cloud": {"mem_mb": None, "latency_ms": 100, "size_bytes": 500 * MIB},
    "Edge": {"mem_mb": 1024, "latency_ms": 50, "size_bytes": 50 * MIB},
    "Mobile": {"mem_mb": 256, "latency_ms": 20, "size_bytes": 10 * MIB},
    "TinyML": {"mem_mb": 256 / 1024, "latency_ms": 10, "size_bytes": 100 * 1024},
}


def rss_mib():
    return psutil.Process().memory_info().rss / MIB


class PeakMemory:
    """Peak process RSS (psutil, sampled every 1 ms in a thread) and peak
    Python/NumPy heap allocated inside the block (tracemalloc)."""

    def __enter__(self):
        gc.collect()
        self.base_rss = rss_mib()
        self.peak_rss = self.base_rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        tracemalloc.start()
        return self

    def _sample(self):
        while not self._stop.is_set():
            self.peak_rss = max(self.peak_rss, rss_mib())
            time.sleep(0.001)

    def __exit__(self, *exc):
        _, peak_heap = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self._stop.set()
        self._thread.join()
        self.peak_rss = max(self.peak_rss, rss_mib())
        self.peak_heap_mb = peak_heap / MIB


def measure_one(name):
    set_seeds()
    warnings.filterwarnings("ignore")
    X_train, X_test, y_train, y_test = load_split()
    factory = make_models()[name]

    # Training time: 1 warm-up, then median of 5 runs.
    factory().fit(X_train, y_train)
    train_times = []
    for _ in range(TRAIN_RUNS):
        t0 = time.perf_counter()
        factory().fit(X_train, y_train)
        train_times.append(time.perf_counter() - t0)

    # Peak memory during one more training run (the model we keep).
    with PeakMemory() as mem_train:
        model = factory().fit(X_train, y_train)

    # Single-sample inference: 1 warm-up, then median of 100 runs.
    sample = X_test[:1]
    model.predict(sample)
    infer_times = []
    for _ in range(INFER_RUNS):
        t0 = time.perf_counter()
        model.predict(sample)
        infer_times.append(time.perf_counter() - t0)

    with PeakMemory() as mem_infer:
        for _ in range(INFER_RUNS):
            model.predict(sample)

    # Model size on disk.
    path = RESULTS / f"model_{name}.joblib"
    joblib.dump(model, path)
    size = path.stat().st_size

    return {
        "model": name,
        "train_time_median_s": float(np.median(train_times)),
        "train_time_min_s": float(np.min(train_times)),
        "train_time_max_s": float(np.max(train_times)),
        "infer_latency_median_ms": float(np.median(infer_times)) * 1e3,
        "infer_latency_p95_ms": float(np.percentile(infer_times, 95)) * 1e3,
        "model_size_bytes": size,
        "model_size_kb": size / 1024,
        "rss_before_train_mb": mem_train.base_rss,
        "peak_rss_train_mb": mem_train.peak_rss,
        "peak_heap_train_mb": mem_train.peak_heap_mb,
        "rss_before_infer_mb": mem_infer.base_rss,
        "peak_rss_infer_mb": mem_infer.peak_rss,
        "peak_heap_infer_mb": mem_infer.peak_heap_mb,
        "model_file": path.name,
    }


def fits(budget, mem_mb, latency_ms, size_bytes):
    return (
        (budget["mem_mb"] is None or mem_mb <= budget["mem_mb"])
        and latency_ms <= budget["latency_ms"]
        and size_bytes <= budget["size_bytes"]
    )


def budget_table(df):
    rows = []
    for _, r in df.iterrows():
        for target, b in BUDGETS.items():
            rows.append({
                "model": r["model"],
                "target": target,
                "latency_ok": r["infer_latency_median_ms"] <= b["latency_ms"],
                "size_ok": r["model_size_bytes"] <= b["size_bytes"],
                # Full Python process (interpreter + numpy + sklearn + model).
                "process_mem_ok": b["mem_mb"] is None or r["peak_rss_infer_mb"] <= b["mem_mb"],
                "fits_python_runtime": fits(b, r["peak_rss_infer_mb"],
                                            r["infer_latency_median_ms"], r["model_size_bytes"]),
                # Model alone: serialized weights + memory added by inference.
                "fits_model_only": fits(b, r["model_size_bytes"] / MIB + r["peak_heap_infer_mb"],
                                        r["infer_latency_median_ms"], r["model_size_bytes"]),
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    args = parser.parse_args()

    if args.model:
        print(json.dumps(measure_one(args.model)))
        return

    rows = []
    for name in make_models():
        out = subprocess.run([sys.executable, str(Path(__file__)), "--model", name],
                             check=True, capture_output=True, text=True)
        rows.append(json.loads(out.stdout.strip().splitlines()[-1]))
    df = pd.DataFrame(rows)
    df.round(4).to_csv(RESULTS / "measurements.csv", index=False)
    print(df.round(4).T.to_string())

    fit = budget_table(df)
    fit.to_csv(RESULTS / "deployment_fit.csv", index=False)
    print()
    print(fit.to_string(index=False))


if __name__ == "__main__":
    main()
