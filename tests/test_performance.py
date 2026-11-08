"""PARTIE VIII - Contrôles de performance rapides (le protocole complet est dans
scripts/run_performance.py). Les seuils viennent de app/config.py."""

import time
import tracemalloc

import numpy as np
import pandas as pd

from app.config import MODEL_PATH, PERFORMANCE_CRITERIA
from app.prediction import predict_proba


def _p95_ms(func, repetitions=300, warmup=20):
    for _ in range(warmup):
        func()
    timings = []
    for _ in range(repetitions):
        start = time.perf_counter()
        func()
        timings.append((time.perf_counter() - start) * 1000)
    return float(np.percentile(timings, 95))


def test_single_prediction_latency_p95(sample_bundle, X_test):
    row = X_test.iloc[[0]]
    p95 = _p95_ms(lambda: predict_proba(sample_bundle, row))
    assert p95 < PERFORMANCE_CRITERIA["single_p95_ms"], f"P95 = {p95:.2f} ms"


def test_batch_throughput(sample_bundle, X_test):
    batch = pd.concat([X_test] * 20, ignore_index=True)  # environ 16 000 lignes
    start = time.perf_counter()
    predict_proba(sample_bundle, batch)
    throughput = len(batch) / (time.perf_counter() - start)
    assert throughput > PERFORMANCE_CRITERIA["batch_throughput_min"] * 10


def test_delivered_model_size():
    size_mb = MODEL_PATH.stat().st_size / 1024 / 1024
    assert size_mb < PERFORMANCE_CRITERIA["model_size_max_mb"]


def test_prediction_memory_peak(sample_bundle, X_test):
    batch = pd.concat([X_test] * 10, ignore_index=True)
    tracemalloc.start()
    predict_proba(sample_bundle, batch)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak / 1024 / 1024 < 100
