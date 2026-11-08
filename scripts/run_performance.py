"""PARTIE VIII - Protocole de mesure des performances.

Protocole :
* matériel et versions relevés automatiquement ;
* mesures avec time.perf_counter, après 50 appels de chauffe ;
* prédiction isolée : 2 000 répétitions sur une transaction réelle du jeu de test ;
* lots de 1 à 56 746 transactions : 20 répétitions (5 pour le lot complet) ;
* mémoire mesurée dans un processus séparé (RSS via psutil) : démarrage, chargement
  du modèle, scoring du jeu de test complet ;
* API : serveur uvicorn local (1 processus), client httpx avec connexion
  persistante ; 2 000 requêtes séquentielles puis 4 000 requêtes avec 16 clients
  concurrents.

    python scripts/run_performance.py
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
import pandas as pd
import psutil
from common import ROOT, load_context, md_table, write_json, write_text

from app.config import MODEL_PATH, PERFORMANCE_CRITERIA
from app.pipeline import library_versions
from app.prediction import load_model, predict_proba

WARMUP = 50


def stats(timings_ms) -> dict:
    t = np.asarray(timings_ms)
    return {
        "n": int(t.size),
        "mean_ms": float(t.mean()),
        "p50_ms": float(np.percentile(t, 50)),
        "p95_ms": float(np.percentile(t, 95)),
        "p99_ms": float(np.percentile(t, 99)),
        "max_ms": float(t.max()),
    }


def time_calls(func, repetitions: int) -> list[float]:
    for _ in range(WARMUP):
        func()
    out = []
    for _ in range(repetitions):
        start = time.perf_counter()
        func()
        out.append((time.perf_counter() - start) * 1000)
    return out


MEMORY_PROBE = r"""
import json, os, sys, psutil
proc = psutil.Process()
r0 = proc.memory_info().rss
sys.path.insert(0, sys.argv[1])
import pandas as pd
from app.prediction import load_model, predict_proba
r1 = proc.memory_info().rss
bundle = load_model(sys.argv[2])
r2 = proc.memory_info().rss
X = pd.read_pickle(sys.argv[3])
predict_proba(bundle, X)
r3 = proc.memory_info().rss
print(json.dumps({"python_mb": r0/2**20, "after_imports_mb": r1/2**20,
                  "after_model_load_mb": r2/2**20, "after_full_test_scoring_mb": r3/2**20}))
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def tree_rss_mb(pid: int) -> float:
    """Mémoire de l'arbre de processus (sous Windows, l'exécutable du venv est un
    lanceur : le serveur réel et ses workers sont des processus enfants)."""
    root = psutil.Process(pid)
    procs = [root, *root.children(recursive=True)]
    return sum(p.memory_info().rss for p in procs if p.is_running()) / 2**20


def stop_tree(server: subprocess.Popen) -> None:
    try:
        children = psutil.Process(server.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    for proc in children:
        proc.kill()
    server.kill()
    server.wait(timeout=10)


def paced_load(base: str, payloads: list, rate: float, duration: float = 20.0) -> dict:
    """Charge à débit imposé (boucle ouverte) : la requête i est due à t0 + i / rate.

    La latence est mesurée depuis l'instant prévu d'envoi : un serveur qui prend du
    retard voit donc ses latences augmenter (pas d'omission coordonnée).
    """
    import threading

    n_threads = 32
    total = int(rate * duration)
    lock, state = threading.Lock(), {"next": 0, "t0": None}
    latencies, errors = [], []
    # Les connexions sont ouvertes AVANT le début du cadencement (pool persistant,
    # comme un client de production) : l'établissement TCP n'est pas mesuré.
    ready = threading.Barrier(n_threads, action=lambda: state.update(t0=time.perf_counter() + 0.2))

    def worker():
        with httpx.Client(base_url=base, timeout=10) as client:
            client.get("/health")
            ready.wait()
            t0 = state["t0"]
            while True:
                with lock:
                    i = state["next"]
                    state["next"] += 1
                if i >= total:
                    return
                due = t0 + i / rate
                delay = due - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                try:
                    r = client.post("/predict", json=payloads[i % len(payloads)])
                    if r.status_code != 200:
                        errors.append(r.status_code)
                except httpx.HTTPError as exc:
                    errors.append(type(exc).__name__)
                latencies.append((time.perf_counter() - due) * 1000)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - state["t0"]
    return {
        **stats(latencies),
        "target_rps": rate,
        "achieved_rps": len(latencies) / elapsed,
        "duration_s": duration,
        "error_rate": len(errors) / max(len(latencies), 1),
    }


def api_measurements(X: pd.DataFrame, workers: int = 1) -> dict:
    port = free_port()
    env = {**os.environ, "MODEL_PATH": str(MODEL_PATH)}
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.api:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
            "--no-access-log",
            "--workers",
            str(workers),
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(150):
            try:
                if httpx.get(f"{base}/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
        time.sleep(1.0 if workers > 1 else 0)  # laisse démarrer tous les workers
        payloads = [
            {k: float(v) for k, v in row.items()} for row in X.head(500).to_dict(orient="records")
        ]
        with httpx.Client(base_url=base, timeout=10) as client:
            i = {"n": 0}

            def one():
                response = client.post("/predict", json=payloads[i["n"] % len(payloads)])
                i["n"] += 1
                assert response.status_code == 200

            sequential = stats(time_calls(one, 2000))
            batch_body = {"transactions": payloads[:100]}
            batch = stats(time_calls(lambda: client.post("/predict/batch", json=batch_body), 200))

        errors, latencies = [], []

        def worker(k: int):
            with httpx.Client(base_url=base, timeout=10) as c:
                for j in range(250):
                    start = time.perf_counter()
                    try:
                        r = c.post("/predict", json=payloads[(k * 250 + j) % len(payloads)])
                        if r.status_code != 200:
                            errors.append(r.status_code)
                    except httpx.HTTPError as exc:
                        errors.append(type(exc).__name__)
                    latencies.append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(worker, range(16)))
        elapsed = time.perf_counter() - start
        concurrent = {
            **stats(latencies),
            "clients": 16,
            "requests": len(latencies),
            "throughput_rps": len(latencies) / elapsed,
            "error_rate": len(errors) / len(latencies),
        }
        paced = paced_load(base, payloads, rate=PERFORMANCE_CRITERIA["batch_throughput_min"])
        server_rss = tree_rss_mb(server.pid)
    finally:
        stop_tree(server)
    return {
        "workers": workers,
        "paced_target_load": paced,
        "sequential_predict": sequential,
        "batch_100_predict": batch,
        "concurrent_predict": concurrent,
        "server_rss_mb": server_rss,
    }


def main() -> int:
    ctx = load_context()
    X = ctx["splits"]["test"]["X"]
    environment = {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "processor": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        **library_versions(),
    }
    start = time.perf_counter()
    bundle = load_model(MODEL_PATH)
    load_ms = (time.perf_counter() - start) * 1000

    row = X.iloc[[100]]
    pipeline = bundle["pipeline"]
    single_raw = stats(time_calls(lambda: pipeline.predict_proba(row), 2000))
    single_full = stats(time_calls(lambda: predict_proba(bundle, row), 2000))

    batches = []
    for size in (1, 10, 100, 1000, 10_000, len(X)):
        part = X.iloc[:size]
        timings = time_calls(
            lambda part=part: predict_proba(bundle, part), 5 if size == len(X) else 20
        )
        s = stats(timings)
        batches.append({"size": size, **s, "throughput_per_s": size / (s["p50_ms"] / 1000)})

    probe_data = ROOT / "reports" / "_perf_probe.pkl"
    X.to_pickle(probe_data)
    try:
        out = subprocess.run(
            [sys.executable, "-c", MEMORY_PROBE, str(ROOT), str(MODEL_PATH), str(probe_data)],
            capture_output=True,
            text=True,
            check=True,
        )
        memory = json.loads(out.stdout.strip().splitlines()[-1])
    finally:
        probe_data.unlink(missing_ok=True)

    api = api_measurements(X, workers=1)
    api_multi = api_measurements(X, workers=4)
    model_mb = MODEL_PATH.stat().st_size / 2**20
    best_batch = max(b["throughput_per_s"] for b in batches)
    limit_rps = PERFORMANCE_CRITERIA["batch_throughput_min"]
    checks = [
        (
            "Latence P95 d'une prédiction (modèle + validation)",
            single_full["p95_ms"],
            "<=",
            PERFORMANCE_CRITERIA["single_p95_ms"],
            "ms",
        ),
        (
            "Latence P95 d'une requête API /predict",
            api["sequential_predict"]["p95_ms"],
            "<=",
            PERFORMANCE_CRITERIA["api_p95_ms"],
            "ms",
        ),
        (
            "Débit max API, 1 processus (saturation)",
            api["concurrent_predict"]["throughput_rps"],
            ">=",
            limit_rps,
            "req/s",
        ),
        (
            "Débit max API, 4 processus (saturation)",
            api_multi["concurrent_predict"]["throughput_rps"],
            ">=",
            limit_rps,
            "req/s",
        ),
        (
            f"P95 API à {limit_rps:g} req/s imposées, 1 processus",
            api["paced_target_load"]["p95_ms"],
            "<=",
            PERFORMANCE_CRITERIA["api_p95_ms"],
            "ms",
        ),
        (
            f"P95 API à {limit_rps:g} req/s imposées, 4 processus",
            api_multi["paced_target_load"]["p95_ms"],
            "<=",
            PERFORMANCE_CRITERIA["api_p95_ms"],
            "ms",
        ),
        ("Débit par lot", best_batch, ">=", limit_rps, "préd./s"),
        ("Taille du modèle", model_mb, "<=", PERFORMANCE_CRITERIA["model_size_max_mb"], "Mo"),
        (
            "Mémoire du service, 1 processus (RSS)",
            api["server_rss_mb"],
            "<=",
            PERFORMANCE_CRITERIA["memory_max_mb"],
            "Mo",
        ),
        (
            "Mémoire du service, 4 processus (RSS)",
            api_multi["server_rss_mb"],
            "<=",
            PERFORMANCE_CRITERIA["memory_max_mb"] * 4,
            "Mo",
        ),
    ]
    results = {
        "environment": environment,
        "model_bytes": MODEL_PATH.stat().st_size,
        "model_load_ms": load_ms,
        "single_prediction_model_only": single_raw,
        "single_prediction_with_validation": single_full,
        "batches": batches,
        "memory_separate_process": memory,
        "api": api,
        "api_4_workers": api_multi,
        "criteria": [
            {
                "check": c[0],
                "value": c[1],
                "operator": c[2],
                "limit": c[3],
                "unit": c[4],
                "passed": c[1] <= c[3] if c[2] == "<=" else c[1] >= c[3],
            }
            for c in checks
        ],
    }
    write_json("performance.json", results)
    write_text("PERFORMANCE.md", render(results))
    for c in results["criteria"]:
        print(f"{'OK ' if c['passed'] else 'KO '} {c['check']} : {c['value']:.2f} {c['unit']}")
    return 0


def render(r: dict) -> str:
    env = r["environment"]
    api = r["api"]
    multi = r["api_4_workers"]
    lines = [
        "# Rapport de performance",
        "",
        "## Protocole",
        "",
        f"- Machine : {env['processor']}, {env['physical_cores']} cœurs physiques / "
        f"{env['logical_cores']} logiques, {env['ram_gb']} Go de RAM, {env['os']}.",
        f"- Logiciels : Python {env['python']}, scikit-learn {env['scikit-learn']}, "
        f"numpy {env['numpy']}, pandas {env['pandas']}.",
        "- Chronométrage `time.perf_counter`, 50 appels de chauffe avant chaque série.",
        "- Prédiction isolée : 2 000 répétitions ; lots : 20 répétitions (5 pour le lot complet).",
        "- Mémoire : processus séparé, RSS mesurée par psutil.",
        "- API : uvicorn sur 127.0.0.1 (1 puis 4 processus), client httpx à connexion "
        "persistante, 2 000 requêtes séquentielles puis 16 clients concurrents x 250 requêtes. "
        "Client et serveur partagent la même machine (le client consomme aussi du CPU).",
        "",
        "## Prédiction isolée",
        "",
        md_table(
            ["Mesure", "Moyenne (ms)", "P50 (ms)", "P95 (ms)", "P99 (ms)"],
            [
                [
                    "Modèle seul (predict_proba)",
                    *[
                        r["single_prediction_model_only"][k]
                        for k in ("mean_ms", "p50_ms", "p95_ms", "p99_ms")
                    ],
                ],
                [
                    "Modèle + validation du contrat",
                    *[
                        r["single_prediction_with_validation"][k]
                        for k in ("mean_ms", "p50_ms", "p95_ms", "p99_ms")
                    ],
                ],
                [
                    "Requête API /predict (HTTP + validation JSON)",
                    *[
                        api["sequential_predict"][k]
                        for k in ("mean_ms", "p50_ms", "p95_ms", "p99_ms")
                    ],
                ],
                [
                    "Requête API, 16 clients concurrents, 1 processus",
                    *[
                        api["concurrent_predict"][k]
                        for k in ("mean_ms", "p50_ms", "p95_ms", "p99_ms")
                    ],
                ],
                [
                    "Requête API, 16 clients concurrents, 4 processus",
                    *[
                        multi["concurrent_predict"][k]
                        for k in ("mean_ms", "p50_ms", "p95_ms", "p99_ms")
                    ],
                ],
                [
                    "Requête API /predict/batch (100 transactions)",
                    *[
                        api["batch_100_predict"][k]
                        for k in ("mean_ms", "p50_ms", "p95_ms", "p99_ms")
                    ],
                ],
                [
                    f"Charge imposée {api['paced_target_load']['target_rps']:g} req/s, 1 processus",
                    *[
                        api["paced_target_load"][k]
                        for k in ("mean_ms", "p50_ms", "p95_ms", "p99_ms")
                    ],
                ],
                [
                    f"Charge imposée {multi['paced_target_load']['target_rps']:g} req/s, 4 processus",
                    *[
                        multi["paced_target_load"][k]
                        for k in ("mean_ms", "p50_ms", "p95_ms", "p99_ms")
                    ],
                ],
            ],
        ),
        "",
        "## Débit par taille de lot",
        "",
        md_table(
            ["Taille du lot", "P50 (ms)", "P95 (ms)", "Débit (prédictions/s)"],
            [
                [b["size"], b["p50_ms"], b["p95_ms"], round(b["throughput_per_s"])]
                for b in r["batches"]
            ],
        ),
        "",
        f"API, 16 clients concurrents : {api['concurrent_predict']['throughput_rps']:.0f} requêtes/s "
        f"avec 1 processus (erreurs {api['concurrent_predict']['error_rate']:.2%}), "
        f"{multi['concurrent_predict']['throughput_rps']:.0f} requêtes/s avec 4 processus "
        f"(erreurs {multi['concurrent_predict']['error_rate']:.2%}).",
        "",
        "## Ressources",
        "",
        md_table(
            ["Mesure", "Valeur"],
            [
                ["Taille du modèle (octets)", r["model_bytes"]],
                ["Temps de chargement du modèle (ms)", r["model_load_ms"]],
                *[[k, v] for k, v in r["memory_separate_process"].items()],
                ["RSS du serveur API, 1 processus (Mo)", api["server_rss_mb"]],
                ["RSS du serveur API, 4 processus + superviseur (Mo)", multi["server_rss_mb"]],
            ],
        ),
        "",
        "## Critères d'acceptation",
        "",
        "Volume du jeu : 284 807 transactions en 48 h, soit 1,65 transaction/s en moyenne. "
        "Exigence de dimensionnement retenue : 100 fois ce débit (165/s). Budget de latence "
        "d'une autorisation de paiement : quelques centaines de millisecondes, dont 10 ms "
        "allouées au modèle et 50 ms à l'appel du service de scoring.",
        "",
        md_table(
            ["Contrôle", "Mesure", "Exigence", "Résultat"],
            [
                [
                    c["check"],
                    f"{c['value']:.2f} {c['unit']}",
                    f"{c['operator']} {c['limit']} {c['unit']}",
                    "conforme" if c["passed"] else "NON CONFORME",
                ]
                for c in r["criteria"]
            ],
        ),
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
