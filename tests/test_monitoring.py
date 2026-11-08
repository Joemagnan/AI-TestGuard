"""PARTIE X - Contrôles de surveillance disponibles sans labels."""

import numpy as np
import pandas as pd
import pytest

from app.monitoring import (
    build_reference_profile,
    drift_report,
    flag_rate_status,
    population_stability_index,
    transaction_alerts,
)
from app.preprocessing import preprocess_data
from tests.factories import make_transactions


@pytest.fixture(scope="module")
def reference():
    X, _ = preprocess_data(make_transactions(n=5000, n_frauds=50, seed=1))
    return X, build_reference_profile(X)


def test_profile_contains_bounds_and_bins(reference):
    X, profile = reference
    amount = profile["Amount"]
    assert amount["min"] == X["Amount"].min() and amount["max"] == X["Amount"].max()
    assert len(amount["proportions"]) == len(amount["edges"]) + 1
    assert sum(amount["proportions"]) == pytest.approx(1.0)


def test_psi_is_near_zero_for_same_distribution(reference):
    _, profile = reference
    X_new, _ = preprocess_data(make_transactions(n=5000, n_frauds=50, seed=2))
    assert population_stability_index(X_new["V1"], profile["V1"]) < 0.02
    assert drift_report(X_new, profile)["status"] == "stable"


def test_psi_detects_shift(reference):
    X, profile = reference
    assert population_stability_index(X["V1"] + 1.0, profile["V1"]) > 0.25


def test_psi_of_empty_input_is_nan(reference):
    _, profile = reference
    assert np.isnan(population_stability_index([np.nan, np.inf], profile["V1"]))


def test_drift_report_ranks_drifting_features_first(reference):
    X, profile = reference
    report = drift_report(X.assign(V7=X["V7"] * 3, Amount=X["Amount"] + 0.1), profile)
    assert report["top"][0] == "V7"
    assert report["features"]["V7"]["status"] == "alerte"


def test_moderate_shift_is_flagged_for_monitoring(reference):
    X, profile = reference
    report = drift_report(X.assign(V2=X["V2"] + 0.35), profile)
    assert report["features"]["V2"]["status"] == "surveillance"


def test_transaction_alerts(reference):
    _, profile = reference
    rows = pd.DataFrame(
        {
            "Time": [10.0, 1e7, 10.0, 10.0],
            **{f"V{i}": [0.0, 0.0, 0.0, 0.0] for i in range(1, 29)},
            "Amount": [10.0, 10.0, 1e9, np.nan],
        }
    )
    rows.loc[0, "V3"] = 99.0
    alerts = transaction_alerts(rows, profile)
    assert alerts[0] == ["valeur_extreme"]
    assert alerts[1] == ["temps_hors_domaine"]
    assert alerts[2] == ["montant_hors_domaine"]
    assert alerts[3] == ["valeurs_manquantes"]


@pytest.mark.parametrize(
    "flagged,status",
    [(10, "normal"), (30, "alerte"), (2, "alerte")],
    ids=["normal", "hausse", "baisse"],
)
def test_flag_rate_status(flagged, status):
    assert flag_rate_status(flagged, 10_000, expected_rate=0.001)["status"] == status
