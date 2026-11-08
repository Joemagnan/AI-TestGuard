"""PARTIE V - Robustesse et sensibilité.

Pour chaque scénario, le comportement attendu est explicite :

====================================  ===========================================
Scénario                              Comportement attendu
====================================  ===========================================
Valeurs manquantes                    traiter (imputation) + alerte
Valeurs infinies                      traiter (-> manquante, imputation) + alerte
Valeurs non numériques                rejeter
Montant extrêmement élevé             traiter + alerte « montant_hors_domaine »
Petite perturbation                   mesurer la sensibilité (pas d'invariance imposée)
Distribution différente               alerte de dérive (PSI)
Schéma invalide                       rejeter
====================================  ===========================================
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.config import PCA_FEATURES
from app.monitoring import batch_quality_report, drift_report, transaction_alerts
from app.prediction import predict_fraud, predict_proba
from app.preprocessing import DataValidationError


def _valid(proba):
    return np.isfinite(proba).all() and (proba >= 0).all() and (proba <= 1).all()


# --- Données manquantes ------------------------------------------------------------------


@pytest.mark.parametrize("columns", [["V1"], ["V14", "V17", "V12"], PCA_FEATURES, ["Amount"]])
def test_missing_values_are_imputed_and_flagged(sample_bundle, X_test, columns):
    X = X_test.copy()
    X.loc[X.index[:10], columns] = np.nan
    proba = predict_proba(sample_bundle, X)
    assert _valid(proba)
    alerts = transaction_alerts(X, sample_bundle["metadata"]["reference_profile"])
    assert all("valeurs_manquantes" in a for a in alerts[:10])
    assert not any("valeurs_manquantes" in a for a in alerts[10:])
    report = batch_quality_report(X, sample_bundle["metadata"]["reference_profile"])
    assert report["missing_alert"]


# --- Valeurs infinies ou non numériques -----------------------------------------------------


def test_infinite_values_are_treated_as_missing(sample_bundle, X_test):
    X = X_test.copy()
    X.iloc[0, 3] = np.inf
    X.iloc[1, 5] = -np.inf
    as_nan = X_test.copy()
    as_nan.iloc[0, 3] = np.nan
    as_nan.iloc[1, 5] = np.nan
    np.testing.assert_array_equal(
        predict_proba(sample_bundle, X), predict_proba(sample_bundle, as_nan)
    )


def test_non_numeric_values_are_rejected(sample_bundle, X_test):
    X = X_test.astype({"V2": object})
    X.loc[X.index[0], "V2"] = "inconnu"
    with pytest.raises(DataValidationError, match="Type incorrect"):
        predict_fraud(sample_bundle, X)


# --- Montants extrêmes ----------------------------------------------------------------------


@pytest.mark.parametrize("amount", [30_000.0, 1e6, 1e12])
def test_extreme_amounts_give_valid_probabilities_and_alert(sample_bundle, X_test, amount):
    X = X_test.head(50).assign(Amount=amount)
    proba = predict_proba(sample_bundle, X)
    assert _valid(proba)
    alerts = transaction_alerts(X, sample_bundle["metadata"]["reference_profile"])
    assert all("montant_hors_domaine" in a for a in alerts)


def test_amount_effect_saturates_thanks_to_log_transform(sample_bundle, X_test):
    """Au-delà du domaine d'entraînement, multiplier le montant par 1 000 ne doit pas
    faire exploser la probabilité : l'effet est borné par le passage au logarithme."""
    big = predict_proba(sample_bundle, X_test.head(200).assign(Amount=1e6))
    huge = predict_proba(sample_bundle, X_test.head(200).assign(Amount=1e9))
    assert np.abs(huge - big).mean() < 0.2


# --- Perturbations limitées : sensibilité mesurée -------------------------------------------------


def _pca_noise(X, train_std, level, seed=0):
    rng = np.random.default_rng(seed)
    noisy = X.copy()
    noise = rng.normal(size=(len(X), len(PCA_FEATURES))) * train_std.to_numpy() * level
    noisy[PCA_FEATURES] = noisy[PCA_FEATURES].to_numpy() + noise
    return noisy


def test_small_noise_has_bounded_effect(sample_bundle, sample_splits, X_test):
    """Bruit gaussien de 1 % de l'écart-type d'entraînement sur V1-V28."""
    train_std = sample_splits["train"]["X"][PCA_FEATURES].std()
    base = predict_proba(sample_bundle, X_test)
    noisy = predict_proba(sample_bundle, _pca_noise(X_test, train_std, 0.01))
    assert np.abs(noisy - base).mean() < 0.02
    flips = ((base >= sample_bundle["threshold"]) != (noisy >= sample_bundle["threshold"])).mean()
    assert flips < 0.01


def test_sensitivity_increases_with_noise_level(sample_bundle, sample_splits, X_test):
    train_std = sample_splits["train"]["X"][PCA_FEATURES].std()
    base = predict_proba(sample_bundle, X_test)
    deltas = [
        np.abs(predict_proba(sample_bundle, _pca_noise(X_test, train_std, lvl)) - base).mean()
        for lvl in (0.01, 0.1, 0.5)
    ]
    assert deltas[0] <= deltas[1] <= deltas[2]


def test_small_relative_amount_change_rarely_changes_decision(sample_bundle, X_test):
    base = predict_fraud(sample_bundle, X_test)["is_fraud"]
    shifted = predict_fraud(sample_bundle, X_test.assign(Amount=X_test["Amount"] * 1.01))[
        "is_fraud"
    ]
    assert (base != shifted).mean() < 0.01


# --- Distribution différente : dérive détectée ------------------------------------------------


def test_distribution_shift_raises_drift_alert(sample_bundle, X_test):
    profile = sample_bundle["metadata"]["reference_profile"]
    assert drift_report(X_test, profile)["status"] in {"stable", "surveillance"}
    shifted = X_test.assign(Amount=X_test["Amount"] * 5, V4=X_test["V4"] + 2)
    report = drift_report(shifted, profile)
    assert report["status"] == "alerte"
    assert {"Amount", "V4"} <= set(report["top"][:2])


# --- Schéma invalide -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutate",
    [
        lambda X: X.drop(columns="Amount"),
        lambda X: X.rename(columns={"V1": "v1"}),
        lambda X: pd.concat([X, X[["V2"]]], axis=1),
        lambda X: X.iloc[0:0],
    ],
    ids=["colonne_absente", "colonne_renommee", "colonne_dupliquee", "lot_vide"],
)
def test_invalid_schema_is_rejected(sample_bundle, X_test, mutate):
    with pytest.raises(DataValidationError):
        predict_fraud(sample_bundle, mutate(X_test))


# --- Propriété : toute transaction valide reçoit une probabilité valide ---------------------------

component = st.floats(min_value=-60, max_value=60, allow_nan=False)


@settings(max_examples=60, deadline=None)
@given(
    components=st.lists(component, min_size=28, max_size=28),
    amount=st.floats(min_value=0, max_value=1e7, allow_nan=False),
    time=st.floats(min_value=0, max_value=1e7, allow_nan=False),
)
def test_property_any_valid_transaction_gets_valid_probability(
    sample_bundle, components, amount, time
):
    transaction = {
        "Time": time,
        **dict(zip(PCA_FEATURES, components, strict=True)),
        "Amount": amount,
    }
    result = predict_fraud(sample_bundle, transaction)
    assert 0.0 <= result["fraud_probability"].iloc[0] <= 1.0
    assert result["is_fraud"].iloc[0] in (0, 1)
