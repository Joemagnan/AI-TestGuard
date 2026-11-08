"""Métriques adaptées à une classe rare, intervalles de confiance, critères
d'acceptation et comparaison à la référence (non-régression)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from app.config import (
    ACCEPTANCE_CRITERIA,
    HIGHER_IS_BETTER,
    REGRESSION_TOLERANCES,
)


def _as_binary(values, name: str) -> np.ndarray:
    array = np.asarray(values).ravel()
    if array.size == 0:
        raise ValueError(f"« {name} » est vide.")
    if not np.isin(array, [0, 1]).all():
        raise ValueError(f"« {name} » doit contenir uniquement 0 et 1.")
    return array.astype(int)


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def evaluate_model(y_true, y_pred, y_proba=None) -> dict:
    """Calcule la matrice de confusion et les métriques de détection de fraude.

    Convention : classe positive = fraude (1). Une métrique non définie (par
    exemple le rappel sans aucune fraude réelle) vaut ``None`` au lieu d'une
    valeur arbitraire.
    """
    y_true = _as_binary(y_true, "y_true")
    y_pred = _as_binary(y_pred, "y_pred")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true et y_pred n'ont pas la même longueur.")

    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel())
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    if precision is None or recall is None or precision + recall == 0:
        f1 = 0.0 if (precision is not None or recall is not None) else None
    else:
        f1 = 2 * precision * recall / (precision + recall)

    metrics = {
        "n": int(len(y_true)),
        "n_frauds": int(y_true.sum()),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float((tp + tn) / len(y_true)),
        "precision": precision,
        "recall": recall,
        "f1": None if f1 is None else float(f1),
        "fpr": _ratio(fp, fp + tn),
        "fnr": _ratio(fn, fn + tp),
        "roc_auc": None,
        "pr_auc": None,
    }

    if y_proba is not None:
        proba = np.asarray(y_proba, dtype="float64").ravel()
        if len(proba) != len(y_true):
            raise ValueError("y_proba n'a pas la même longueur que y_true.")
        if not np.isfinite(proba).all() or (proba < 0).any() or (proba > 1).any():
            raise ValueError("y_proba doit contenir des probabilités finies entre 0 et 1.")
        if len(np.unique(y_true)) == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, proba))
            metrics["pr_auc"] = float(average_precision_score(y_true, proba))
    return metrics


def bootstrap_ci(
    y_true,
    y_pred,
    y_proba=None,
    metrics=("recall", "precision", "fpr", "pr_auc"),
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Intervalles de confiance par rééchantillonnage stratifié (bootstrap percentile).

    Le rééchantillonnage se fait séparément dans chaque classe pour conserver le
    nombre de fraudes, très faible, d'un tirage à l'autre.
    """
    y_true = _as_binary(y_true, "y_true")
    y_pred = _as_binary(y_pred, "y_pred")
    proba = None if y_proba is None else np.asarray(y_proba, dtype="float64")
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y_true == 1), np.flatnonzero(y_true == 0)
    samples = {name: [] for name in metrics}
    for _ in range(n_boot):
        idx = np.concatenate(
            [rng.choice(pos, size=len(pos), replace=True), rng.choice(neg, size=len(neg))]
        )
        result = evaluate_model(y_true[idx], y_pred[idx], None if proba is None else proba[idx])
        for name in metrics:
            if result.get(name) is not None:
                samples[name].append(result[name])
    alpha = (1 - confidence) / 2
    return {
        name: (
            [float(np.quantile(values, alpha)), float(np.quantile(values, 1 - alpha))]
            if values
            else None
        )
        for name, values in samples.items()
    }


def check_acceptance(metrics: dict, criteria: dict | None = None) -> dict:
    """Vérifie les critères d'acceptation définis à partir des exigences."""
    criteria = criteria or ACCEPTANCE_CRITERIA
    checks = []
    for name, bounds in criteria.items():
        value = metrics.get(name)
        ok = value is not None
        if ok and "min" in bounds:
            ok = value >= bounds["min"]
        if ok and "max" in bounds:
            ok = value <= bounds["max"]
        checks.append({"metric": name, "value": value, **bounds, "passed": bool(ok)})
    return {"accepted": all(c["passed"] for c in checks), "checks": checks}


def compare_with_baseline(
    candidate: dict,
    baseline: dict,
    tolerances: dict | None = None,
) -> dict:
    """Compare une nouvelle version à la référence figée.

    ``candidate`` et ``baseline`` contiennent ``validation_metrics`` et
    ``validation_fingerprint``. Règle de décision :

    * dégradation > tolérance sur au moins une métrique : **rejeter** ;
    * dégradation > moitié de la tolérance, ou jeu de validation différent :
      **investiguer** ;
    * sinon : **accepter**.
    """
    tolerances = tolerances or REGRESSION_TOLERANCES
    reasons, checks = [], []

    same_validation = candidate.get("validation_fingerprint") == baseline.get(
        "validation_fingerprint"
    )
    if not same_validation:
        reasons.append(
            "Jeu de validation différent de celui de la référence : comparaison invalide."
        )

    cand_m, base_m = candidate["validation_metrics"], baseline["validation_metrics"]
    for name, tolerance in tolerances.items():
        before, after = base_m.get(name), cand_m.get(name)
        if before is None or after is None:
            checks.append({"metric": name, "status": "non_calculable"})
            reasons.append(f"Métrique « {name} » indisponible.")
            continue
        degradation = (before - after) if HIGHER_IS_BETTER[name] else (after - before)
        if degradation > tolerance + 1e-12:
            status = "regression"
        elif degradation > tolerance / 2 + 1e-12:
            status = "attention"
        else:
            status = "ok"
        checks.append(
            {
                "metric": name,
                "baseline": before,
                "candidate": after,
                "delta": after - before,
                "degradation": degradation,
                "tolerance": tolerance,
                "status": status,
            }
        )
        if status != "ok":
            reasons.append(f"{name} : dégradation de {degradation:.4f} (tolérance {tolerance}).")

    statuses = {c["status"] for c in checks}
    if "regression" in statuses:
        decision = "rejeter"
    elif not same_validation or statuses & {"attention", "non_calculable"}:
        decision = "investiguer"
    else:
        decision = "accepter"

    notes = []
    base_libs, cand_libs = baseline.get("libraries", {}), candidate.get("libraries", {})
    for lib in sorted(set(base_libs) | set(cand_libs)):
        if base_libs.get(lib) != cand_libs.get(lib):
            notes.append(f"Version de {lib} : {base_libs.get(lib)} -> {cand_libs.get(lib)}")
    return {"decision": decision, "checks": checks, "reasons": reasons, "notes": notes}
