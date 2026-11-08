"""Découpage des données, entraînement du pipeline et choix du seuil de décision."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from app.config import COST_FN, COST_FP, MAX_FPR, SEED, TARGET, TEST_SIZE, VALIDATION_SIZE
from app.preprocessing import DataValidationError, build_preprocessor

MODEL_FAMILIES = {
    "logistic_regression": (LogisticRegression, {"max_iter": 5000, "solver": "lbfgs"}),
    "decision_tree": (DecisionTreeClassifier, {}),
}

# Grilles volontairement petites : modèles « légers » demandés par le sujet.
PARAM_GRIDS = {
    "logistic_regression": [
        {"C": c, "class_weight": w} for c in (0.01, 0.1, 1.0) for w in (None, "balanced")
    ],
    "decision_tree": [
        {"max_depth": d, "min_samples_leaf": leaf, "class_weight": w}
        for d in (4, 6, 8)
        for leaf in (1, 20)
        for w in (None, "balanced")
    ],
}


def split_data(
    df: pd.DataFrame,
    seed: int = SEED,
    test_size: float = TEST_SIZE,
    val_size: float = VALIDATION_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Découpage stratifié entraînement / validation / test (60 / 20 / 20 par défaut).

    Le découpage est fait AVANT toute transformation apprise. Les index d'origine
    sont conservés pour pouvoir vérifier l'absence de recouvrement.
    """
    if TARGET not in df.columns:
        raise DataValidationError(f"Colonne cible « {TARGET} » absente.")
    if not 0 < test_size < 1 or not 0 < val_size < 1 or test_size + val_size >= 1:
        raise ValueError("Proportions de découpage invalides.")
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[TARGET], random_state=seed
    )
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=relative_val, stratify=train_val[TARGET], random_state=seed
    )
    return train, val, test


def build_pipeline(
    model_name: str = "logistic_regression",
    params: dict | None = None,
    time_encoding: str = "raw",
    seed: int = SEED,
) -> Pipeline:
    """Regroupe le prétraitement appris et le classifieur dans un seul objet."""
    if model_name not in MODEL_FAMILIES:
        raise ValueError(f"Modèle inconnu : {model_name!r}. Choix : {sorted(MODEL_FAMILIES)}.")
    estimator_cls, defaults = MODEL_FAMILIES[model_name]
    kwargs = {**defaults, **(params or {})}
    if "random_state" in estimator_cls().get_params():
        kwargs.setdefault("random_state", seed)
    return Pipeline(
        [("preprocess", build_preprocessor(time_encoding)), ("model", estimator_cls(**kwargs))]
    )


def train_model(
    X: pd.DataFrame,
    y,
    model_name: str = "logistic_regression",
    params: dict | None = None,
    time_encoding: str = "raw",
    seed: int = SEED,
) -> Pipeline:
    """Entraîne le pipeline complet (prétraitement + modèle) sur les données fournies.

    Les statistiques d'imputation et de standardisation sont apprises ici, donc
    uniquement sur les données passées en argument (le jeu d'entraînement).
    """
    y = np.asarray(y).ravel()
    if len(X) != len(y):
        raise DataValidationError(f"X ({len(X)} lignes) et y ({len(y)} valeurs) ne concordent pas.")
    if len(y) == 0:
        raise DataValidationError("Impossible d'entraîner sur un jeu vide.")
    classes = set(np.unique(y).tolist())
    if not classes <= {0, 1}:
        raise DataValidationError(f"Cible non binaire : {sorted(classes)}.")
    if len(classes) < 2:
        raise DataValidationError("L'entraînement exige des exemples des deux classes.")
    pipeline = build_pipeline(model_name, params, time_encoding, seed)
    pipeline.fit(X, y)
    return pipeline


def select_threshold(
    y_true,
    y_proba,
    cost_fn: float = COST_FN,
    cost_fp: float = COST_FP,
    max_fpr: float = MAX_FPR,
) -> dict:
    """Choisit le seuil minimisant le coût attendu ``cost_fn * FN + cost_fp * FP``,
    sous la contrainte ``FPR <= max_fpr``.

    À appliquer sur le jeu de VALIDATION uniquement. Une transaction est signalée
    lorsque ``probabilité >= seuil``.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    proba = np.asarray(y_proba, dtype="float64").ravel()
    if len(y_true) != len(proba) or len(y_true) == 0:
        raise ValueError("y_true et y_proba doivent être non vides et de même longueur.")
    positives = int(y_true.sum())
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Le choix du seuil exige des exemples des deux classes.")

    order = np.argsort(-proba, kind="mergesort")
    sorted_proba, sorted_true = proba[order], y_true[order]
    cum_tp = np.cumsum(sorted_true)
    cum_fp = np.cumsum(1 - sorted_true)
    # dernier indice de chaque valeur distincte de probabilité
    last = np.r_[np.flatnonzero(np.diff(sorted_proba) != 0), len(sorted_proba) - 1]
    thresholds = sorted_proba[last]
    tp, fp = cum_tp[last], cum_fp[last]
    fn = positives - tp
    fpr = fp / negatives
    cost = cost_fn * fn + cost_fp * fp

    feasible = fpr <= max_fpr + 1e-12
    if not feasible.any():
        raise ValueError(f"Aucun seuil ne respecte la contrainte FPR <= {max_fpr}.")
    candidates = np.flatnonzero(feasible)
    best = candidates[np.argmin(cost[candidates])]
    precision = tp[best] / (tp[best] + fp[best])
    return {
        "threshold": float(thresholds[best]),
        "expected_cost": float(cost[best]),
        "recall": float(tp[best] / positives),
        "precision": float(precision),
        "fpr": float(fpr[best]),
        "rule": {"cost_fn": cost_fn, "cost_fp": cost_fp, "max_fpr": max_fpr},
    }
