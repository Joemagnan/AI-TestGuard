"""Pipeline de bout en bout : chargement, contrôle, découpage, sélection sur
validation, choix du seuil, sauvegarde, évaluation finale unique sur le test.

Utilisation : ``python -m app.pipeline --data data/creditcard.csv``
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from app import __version__
from app.config import (
    ACCEPTANCE_CRITERIA,
    COST_FN,
    COST_FP,
    FULL_DATA_PATH,
    MAX_FPR,
    MODELS_DIR,
    REPORTS_DIR,
    ROOT,
    SEED,
    TARGET,
)
from app.evaluation import bootstrap_ci, check_acceptance, evaluate_model
from app.monitoring import build_reference_profile
from app.prediction import file_sha256, save_model
from app.preprocessing import (
    DataValidationError,
    canonical_order,
    data_fingerprint,
    deduplicate,
    preprocess_data,
    validate_data,
)
from app.training import PARAM_GRIDS, select_threshold, split_data, train_model

FAMILY_LABELS = {
    "logistic_regression": "Régression logistique",
    "decision_tree": "Arbre de décision",
}


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Lit le CSV ; toute erreur de lecture est convertie en erreur explicite."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}. Télécharger creditcard.csv (voir data/README.md)."
        )
    try:
        return pd.read_csv(path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise DataValidationError(f"Lecture du CSV impossible : {exc}") from exc


def prepare_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Contrôle du contrat, suppression des doublons exacts, ordre canonique."""
    data = validate_data(df, require_target=True)
    data, dedup = deduplicate(data)
    data = canonical_order(data)
    info = {
        **dedup,
        "rows": int(len(data)),
        "frauds": int(data[TARGET].sum()),
        "fraud_rate": float(data[TARGET].mean()),
        "fingerprint": data_fingerprint(data),
    }
    return data, info


def library_versions() -> dict:
    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
    }


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        return out.stdout.strip() + ("-modifie" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "inconnu"


def make_splits(data: pd.DataFrame, seed: int = SEED) -> dict:
    train, val, test = split_data(data, seed=seed)
    splits = {}
    for name, part in (("train", train), ("val", val), ("test", test)):
        X, y = preprocess_data(part)
        splits[name] = {"X": X, "y": y.to_numpy(), "frame": part}
    splits["validation_fingerprint"] = data_fingerprint(val)
    return splits


def fit_and_calibrate(
    splits: dict,
    model_name: str,
    params: dict | None,
    time_encoding: str = "raw",
    seed: int = SEED,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
    max_fpr: float = MAX_FPR,
) -> dict:
    """Entraîne sur train, choisit le seuil sur validation, mesure sur validation."""
    pipeline = train_model(
        splits["train"]["X"], splits["train"]["y"], model_name, params, time_encoding, seed
    )
    y_val = splits["val"]["y"]
    proba = pipeline.predict_proba(splits["val"]["X"])[:, 1]
    selection = select_threshold(y_val, proba, cost_fn=cost_fn, cost_fp=cost_fp, max_fpr=max_fpr)
    metrics = evaluate_model(y_val, (proba >= selection["threshold"]).astype(int), proba)
    metrics["expected_cost"] = selection["expected_cost"]
    return {
        "pipeline": pipeline,
        "model_name": model_name,
        "params": params or {},
        "time_encoding": time_encoding,
        "selection": selection,
        "val_proba": proba,
        "validation_metrics": metrics,
    }


def grid_search(
    splits: dict, families, grids=None, time_encoding: str = "raw", seed: int = SEED
) -> tuple[dict, list[dict]]:
    """Sélection des hyperparamètres sur la validation (PR-AUC), par famille."""
    grids = grids or PARAM_GRIDS
    best, rows = {}, []
    for family in families:
        for params in grids[family]:
            pipeline = train_model(
                splits["train"]["X"], splits["train"]["y"], family, params, time_encoding, seed
            )
            proba = pipeline.predict_proba(splits["val"]["X"])[:, 1]
            scores = evaluate_model(splits["val"]["y"], (proba >= 0.5).astype(int), proba)
            row = {
                "model": family,
                "params": params,
                "val_pr_auc": scores["pr_auc"],
                "val_roc_auc": scores["roc_auc"],
            }
            rows.append(row)
            if family not in best or row["val_pr_auc"] > best[family]["val_pr_auc"]:
                best[family] = row
    return best, rows


def run_training(
    data_path: str | Path = FULL_DATA_PATH,
    models_dir: str | Path = MODELS_DIR,
    reports_dir: str | Path = REPORTS_DIR,
    families=("logistic_regression", "decision_tree"),
    grids=None,
    time_encoding: str = "raw",
    seed: int = SEED,
    version: str = __version__,
    n_boot: int = 1000,
    make_plots: bool = True,
) -> dict:
    models_dir, reports_dir = Path(models_dir), Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    raw = load_dataset(data_path)
    data, data_info = prepare_dataset(raw)
    data_info["source_file"] = Path(data_path).name
    data_info["source_sha256"] = file_sha256(data_path)
    splits = make_splits(data, seed)

    # 1. Hyperparamètres choisis sur la validation
    best_params, grid_rows = grid_search(splits, families, grids, time_encoding, seed)
    # 2. Seuil choisi sur la validation pour le meilleur réglage de chaque famille
    candidates = {
        family: fit_and_calibrate(
            splits, family, best_params[family]["params"], time_encoding, seed
        )
        for family in families
    }
    # 3. Modèle retenu : coût attendu minimal sur validation (puis PR-AUC)
    selected_name = min(
        candidates,
        key=lambda f: (
            candidates[f]["validation_metrics"]["expected_cost"],
            -candidates[f]["validation_metrics"]["pr_auc"],
        ),
    )
    selected = candidates[selected_name]
    threshold = selected["selection"]["threshold"]

    # 4. Évaluation finale unique sur le test (aucune décision prise après)
    X_test, y_test = splits["test"]["X"], splits["test"]["y"]
    test_results = {}
    for family, cand in candidates.items():
        proba = cand["pipeline"].predict_proba(X_test)[:, 1]
        pred = (proba >= cand["selection"]["threshold"]).astype(int)
        test_results[family] = {"proba": proba, "metrics": evaluate_model(y_test, pred, proba)}
    sel_test = test_results[selected_name]
    test_metrics = sel_test["metrics"]
    ci = bootstrap_ci(
        y_test,
        (sel_test["proba"] >= threshold).astype(int),
        sel_test["proba"],
        n_boot=n_boot,
        seed=seed,
    )
    acceptance = check_acceptance(test_metrics, ACCEPTANCE_CRITERIA)
    train_proba = selected["pipeline"].predict_proba(splits["train"]["X"])[:, 1]
    train_metrics = evaluate_model(
        splits["train"]["y"], (train_proba >= threshold).astype(int), train_proba
    )

    metadata = {
        "version": version,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "code_version": git_commit(),
        "model_name": selected_name,
        "params": selected["params"],
        "time_encoding": time_encoding,
        "threshold_rule": selected["selection"]["rule"],
        "seed": seed,
        "features": list(splits["train"]["X"].columns),
        "split_sizes": {k: int(len(splits[k]["y"])) for k in ("train", "val", "test")},
        "split_frauds": {k: int(splits[k]["y"].sum()) for k in ("train", "val", "test")},
        "data": data_info,
        "validation_fingerprint": splits["validation_fingerprint"],
        "validation_metrics": selected["validation_metrics"],
        "validation_flag_rate": float(np.mean(selected["val_proba"] >= threshold)),
        "test_metrics": test_metrics,
        "libraries": library_versions(),
        "reference_profile": build_reference_profile(splits["train"]["X"]),
    }
    full_metadata = save_model(
        selected["pipeline"], threshold, metadata, models_dir / "fraud_model.joblib"
    )

    report = {
        "version": version,
        "selected_model": selected_name,
        "threshold": threshold,
        "threshold_selection": selected["selection"],
        "data": data_info,
        "split_sizes": metadata["split_sizes"],
        "split_frauds": metadata["split_frauds"],
        "grid_search": grid_rows,
        "candidates": {
            family: {
                "params": cand["params"],
                "threshold": cand["selection"]["threshold"],
                "validation": cand["validation_metrics"],
                "test": test_results[family]["metrics"],
            }
            for family, cand in candidates.items()
        },
        "train_metrics_selected": train_metrics,
        "test_metrics_selected": test_metrics,
        "test_confidence_intervals_95": ci,
        "acceptance": acceptance,
        "naive_all_normal_accuracy": float(1 - y_test.mean()),
        "libraries": metadata["libraries"],
        "artifact_sha256": full_metadata["artifact_sha256"],
        "artifact_bytes": full_metadata["artifact_bytes"],
    }
    (reports_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if make_plots:
        from app.plots import plot_confusion_matrix, plot_roc_pr, plot_threshold_analysis

        figures = reports_dir / "figures"
        plot_confusion_matrix(
            selected["validation_metrics"],
            f"{FAMILY_LABELS[selected_name]} - validation",
            figures / "confusion_validation.png",
        )
        for family in families:
            plot_confusion_matrix(
                test_results[family]["metrics"],
                f"{FAMILY_LABELS[family]} - test",
                figures / f"confusion_test_{family}.png",
            )
        plot_roc_pr(
            {
                FAMILY_LABELS[f]: {
                    "y_true": y_test,
                    "y_proba": test_results[f]["proba"],
                    "threshold": candidates[f]["selection"]["threshold"],
                }
                for f in families
            },
            figures / "roc_pr_test.png",
            " (test)",
        )
        plot_threshold_analysis(
            splits["val"]["y"],
            selected["val_proba"],
            selected["selection"],
            figures / "threshold_validation.png",
        )
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Entraînement et évaluation AI-TestGuard")
    parser.add_argument("--data", default=str(FULL_DATA_PATH))
    parser.add_argument("--models-dir", default=str(MODELS_DIR))
    parser.add_argument("--reports-dir", default=str(REPORTS_DIR))
    parser.add_argument("--time-encoding", choices=["raw", "cyclic"], default="raw")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    report = run_training(
        args.data,
        args.models_dir,
        args.reports_dir,
        time_encoding=args.time_encoding,
        make_plots=not args.no_plots,
    )
    m = report["test_metrics_selected"]
    print(
        f"Modèle retenu : {report['selected_model']} | seuil {report['threshold']:.4f}\n"
        f"Test : rappel {m['recall']:.3f} | précision {m['precision']:.3f} | FPR {m['fpr']:.4f} "
        f"| PR-AUC {m['pr_auc']:.3f}\n"
        f"Critères d'acceptation : {'OK' if report['acceptance']['accepted'] else 'NON ATTEINTS'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
