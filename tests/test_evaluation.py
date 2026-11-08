"""PARTIE III - Tests des métriques, des critères d'acceptation et de la
comparaison à la référence."""

import numpy as np
import pytest

from app.evaluation import bootstrap_ci, check_acceptance, compare_with_baseline, evaluate_model


def test_confusion_counts_and_rates_on_known_example():
    y_true = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
    m = evaluate_model(y_true, y_pred)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (3, 1, 1, 5)
    assert m["precision"] == pytest.approx(3 / 4)
    assert m["recall"] == pytest.approx(3 / 4)
    assert m["f1"] == pytest.approx(3 / 4)
    assert m["fpr"] == pytest.approx(1 / 6)
    assert m["fnr"] == pytest.approx(1 / 4)
    assert m["accuracy"] == pytest.approx(8 / 10)


def test_accuracy_paradox_all_normal_predictor():
    """Un modèle qui ne signale jamais rien a 99,8 % d'accuracy mais 0 % de rappel."""
    y_true = np.zeros(10_000, dtype=int)
    y_true[:17] = 1  # prévalence 0,17 % comme dans le jeu Kaggle
    m = evaluate_model(y_true, np.zeros_like(y_true))
    assert m["accuracy"] > 0.998
    assert m["recall"] == 0.0
    assert m["fnr"] == 1.0
    assert m["precision"] is None  # aucune alerte : précision non définie


def test_threshold_free_metrics_with_probabilities():
    y_true = np.array([0, 0, 1, 1])
    m = evaluate_model(y_true, [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m["roc_auc"] == 1.0 and m["pr_auc"] == 1.0


def test_auc_undefined_when_single_class():
    m = evaluate_model([0, 0, 0], [0, 1, 0], [0.1, 0.6, 0.2])
    assert m["roc_auc"] is None and m["pr_auc"] is None
    assert m["recall"] is None and m["fnr"] is None
    assert m["precision"] == 0.0


@pytest.mark.parametrize(
    "y_true,y_pred,y_proba,message",
    [
        ([], [], None, "vide"),
        ([0, 2], [0, 1], None, "0 et 1"),
        ([0, 1], [0, 1, 1], None, "longueur"),
        ([0, 1], [0, 1], [0.2, 1.4], "probabilités"),
        ([0, 1], [0, 1], [0.2, np.nan], "probabilités"),
        ([0, 1], [0, 1], [0.2], "longueur"),
    ],
)
def test_invalid_inputs_are_rejected(y_true, y_pred, y_proba, message):
    with pytest.raises(ValueError, match=message):
        evaluate_model(y_true, y_pred, y_proba)


def test_bootstrap_interval_contains_estimate_and_is_reproducible():
    rng = np.random.default_rng(0)
    y_true = (rng.random(3000) < 0.05).astype(int)
    proba = np.clip(y_true * 0.6 + rng.random(3000) * 0.5, 0, 1)
    y_pred = (proba >= 0.5).astype(int)
    point = evaluate_model(y_true, y_pred, proba)
    ci = bootstrap_ci(y_true, y_pred, proba, n_boot=200, seed=1)
    for name in ("recall", "precision", "fpr", "pr_auc"):
        low, high = ci[name]
        assert low <= point[name] <= high
    assert ci == bootstrap_ci(y_true, y_pred, proba, n_boot=200, seed=1)


def test_acceptance_criteria():
    criteria = {"recall": {"min": 0.75}, "fpr": {"max": 0.005}}
    assert check_acceptance({"recall": 0.8, "fpr": 0.001}, criteria)["accepted"]
    refused = check_acceptance({"recall": 0.7, "fpr": 0.001}, criteria)
    assert not refused["accepted"]
    assert [c["passed"] for c in refused["checks"]] == [False, True]
    assert not check_acceptance({"recall": None, "fpr": 0.0}, criteria)["accepted"]


# --- Non-régression ---------------------------------------------------------------------

BASE = {
    "validation_fingerprint": "abc",
    "validation_metrics": {"pr_auc": 0.80, "recall": 0.85, "precision": 0.70, "fpr": 0.0010},
    "libraries": {"scikit-learn": "1.9.1"},
}
TOL = {"pr_auc": 0.02, "recall": 0.04, "precision": 0.05, "fpr": 0.001}


def _candidate(**metrics):
    return {**BASE, "validation_metrics": {**BASE["validation_metrics"], **metrics}}


def test_identical_version_is_accepted():
    result = compare_with_baseline(_candidate(), BASE, TOL)
    assert result["decision"] == "accepter"
    assert all(c["status"] == "ok" for c in result["checks"])


def test_improvement_is_accepted():
    assert (
        compare_with_baseline(_candidate(recall=0.95, fpr=0.0), BASE, TOL)["decision"] == "accepter"
    )


def test_small_degradation_requires_investigation():
    result = compare_with_baseline(_candidate(recall=0.82), BASE, TOL)  # -0,03 > tol/2
    assert result["decision"] == "investiguer"


def test_degradation_beyond_tolerance_is_rejected():
    result = compare_with_baseline(_candidate(pr_auc=0.70), BASE, TOL)
    assert result["decision"] == "rejeter"
    assert any("pr_auc" in reason for reason in result["reasons"])


def test_fpr_increase_is_a_regression():
    """Pour le FPR, c'est une hausse qui dégrade."""
    assert compare_with_baseline(_candidate(fpr=0.0025), BASE, TOL)["decision"] == "rejeter"


def test_different_validation_set_cannot_be_accepted():
    candidate = {**_candidate(), "validation_fingerprint": "autre"}
    result = compare_with_baseline(candidate, BASE, TOL)
    assert result["decision"] == "investiguer"


def test_missing_metric_requires_investigation_and_versions_are_reported():
    candidate = {**_candidate(precision=None), "libraries": {"scikit-learn": "1.10.0"}}
    result = compare_with_baseline(candidate, BASE, TOL)
    assert result["decision"] == "investiguer"
    assert result["notes"] == ["Version de scikit-learn : 1.9.1 -> 1.10.0"]
