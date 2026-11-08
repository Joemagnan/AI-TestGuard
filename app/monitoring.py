"""Profil de référence, alertes par transaction et détection de dérive (PSI).

Ces contrôles ne nécessitent pas les labels : ils sont disponibles immédiatement
en production, contrairement au rappel ou au taux de faux négatifs qui exigent de
connaître la vraie classe (souvent plusieurs semaines plus tard).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import FEATURES, MISSING_RATE_ALERT, PSI_ALERT, PSI_WARNING
from app.preprocessing import clean_features

EPSILON = 1e-6


def build_reference_profile(X_train: pd.DataFrame, n_bins: int = 10) -> dict:
    """Résumé des données d'entraînement : bornes, médiane et déciles par variable."""
    X = clean_features(X_train)
    profile = {}
    for col in FEATURES:
        values = X[col].dropna().to_numpy()
        edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1]))
        counts = np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)
        profile[col] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "median": float(np.median(values)),
            "edges": edges.tolist(),
            "proportions": (counts / counts.sum()).tolist(),
        }
    return profile


def population_stability_index(values, reference: dict) -> float:
    """PSI entre la distribution observée et la distribution de référence.

    Repères usuels : < 0,10 stable ; 0,10-0,25 à surveiller ; > 0,25 dérive forte.
    """
    values = np.asarray(values, dtype="float64")
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    edges = np.asarray(reference["edges"])
    expected = np.clip(np.asarray(reference["proportions"]), EPSILON, None)
    counts = np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)
    actual = np.clip(counts / counts.sum(), EPSILON, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def drift_report(X_new: pd.DataFrame, profile: dict) -> dict:
    """PSI par variable et statut global (stable / surveillance / alerte)."""
    X = clean_features(X_new)
    features = {}
    for col in FEATURES:
        psi = population_stability_index(X[col], profile[col])
        if psi > PSI_ALERT:
            status = "alerte"
        elif psi > PSI_WARNING:
            status = "surveillance"
        else:
            status = "stable"
        features[col] = {"psi": round(psi, 6), "status": status}
    statuses = {f["status"] for f in features.values()}
    overall = (
        "alerte"
        if "alerte" in statuses
        else "surveillance"
        if "surveillance" in statuses
        else "stable"
    )
    ranked = sorted(features.items(), key=lambda kv: kv[1]["psi"], reverse=True)
    return {"status": overall, "top": [name for name, _ in ranked[:5]], "features": features}


def critical_features(pipeline, k: int = 3) -> list[str]:
    """Les ``k`` variables d'entrée dont dépend le plus le modèle.

    Importances pour un arbre, |coefficient| pour un modèle linéaire (variables
    standardisées). Sert à repérer les transactions dont une variable critique a dû
    être imputée (anomalie A02 : l'imputation masque alors la fraude).
    """
    model = pipeline.named_steps["model"]
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        weights = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        weights = np.abs(np.ravel(model.coef_))
    else:
        return []
    ranked = [str(names[i]) for i in np.argsort(-weights)]
    inputs = []
    for name in ranked:
        source = "Time" if name.startswith("Time") else name
        if source in FEATURES and source not in inputs:
            inputs.append(source)
        if len(inputs) == k:
            break
    return inputs


def transaction_alerts(
    X: pd.DataFrame, profile: dict, critical: list[str] | None = None
) -> list[list[str]]:
    """Alertes par transaction (sans bloquer la prédiction).

    * ``variable_critique_manquante`` : une variable de ``critical`` a été imputée ;
      la décision doit être confiée à un analyste (revue manuelle) ;
    * ``valeurs_manquantes`` : au moins une variable absente ou infinie (imputée) ;
    * ``montant_hors_domaine`` : montant supérieur au maximum vu à l'entraînement ;
    * ``temps_hors_domaine`` : ``Time`` au-delà de la période d'entraînement
      (attendu en production : la variable n'est pas transposable telle quelle) ;
    * ``valeur_extreme`` : une composante V au-delà des bornes d'entraînement.
    """
    clean = clean_features(X)
    missing = clean.isna().any(axis=1).to_numpy()
    amount_out = (clean["Amount"] > profile["Amount"]["max"]).to_numpy()
    time_out = (clean["Time"] > profile["Time"]["max"]).to_numpy()
    pca_cols = [c for c in FEATURES if c.startswith("V")]
    lows = np.array([profile[c]["min"] for c in pca_cols])
    highs = np.array([profile[c]["max"] for c in pca_cols])
    pca = clean[pca_cols].to_numpy()
    extreme = ((pca < lows) | (pca > highs)).any(axis=1)
    critical_missing = (
        clean[critical].isna().any(axis=1).to_numpy() if critical else np.zeros(len(clean), bool)
    )

    alerts = []
    for i in range(len(clean)):
        row = []
        if critical_missing[i]:
            row.append("variable_critique_manquante")
        if missing[i]:
            row.append("valeurs_manquantes")
        if amount_out[i]:
            row.append("montant_hors_domaine")
        if time_out[i]:
            row.append("temps_hors_domaine")
        if extreme[i]:
            row.append("valeur_extreme")
        alerts.append(row)
    return alerts


def batch_quality_report(X: pd.DataFrame, profile: dict, critical: list[str] | None = None) -> dict:
    """Synthèse de qualité d'un lot : taux de lignes incomplètes et alertes."""
    clean = clean_features(X)
    alerts = transaction_alerts(clean, profile, critical)
    missing_rate = float(clean.isna().any(axis=1).mean())
    counts = {}
    for row in alerts:
        for name in row:
            counts[name] = counts.get(name, 0) + 1
    return {
        "rows": int(len(clean)),
        "missing_rate": missing_rate,
        "missing_alert": missing_rate > MISSING_RATE_ALERT,
        "alert_counts": counts,
    }


def flag_rate_status(
    n_flagged: int, n_total: int, expected_rate: float, factor: float = 2.0
) -> dict:
    """Compare la part de transactions signalées à celle observée en validation.

    Un écart d'un facteur ``factor`` (à la hausse ou à la baisse) déclenche une alerte :
    hausse = surcharge de l'équipe de revue, baisse = fraudes possiblement manquées.
    """
    rate = n_flagged / n_total if n_total else 0.0
    ratio = rate / expected_rate if expected_rate else float("inf")
    status = "alerte" if (ratio > factor or ratio < 1 / factor) else "normal"
    return {"rate": rate, "expected_rate": expected_rate, "ratio": ratio, "status": status}
