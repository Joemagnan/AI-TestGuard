"""Tests unitaires de l'entraînement, du découpage et du choix du seuil."""

import numpy as np
import pytest

from app.config import TARGET
from app.preprocessing import DataValidationError, preprocess_data
from app.training import build_pipeline, select_threshold, split_data, train_model
from tests.factories import make_transactions


def test_split_is_stratified_disjoint_and_complete():
    data = make_transactions(n=1000, n_frauds=50)
    train, val, test = split_data(data, seed=42)
    assert (len(train), len(val), len(test)) == (600, 200, 200)
    assert not set(train.index) & set(val.index)
    assert not set(train.index) & set(test.index)
    assert not set(val.index) & set(test.index)
    assert len(set(train.index) | set(val.index) | set(test.index)) == len(data)
    for part in (train, val, test):  # stratification : 5 % de fraudes partout
        assert part[TARGET].mean() == pytest.approx(0.05, abs=0.005)


def test_split_is_reproducible_with_seed():
    data = make_transactions(n=500, n_frauds=25)
    first = split_data(data, seed=7)
    second = split_data(data, seed=7)
    other = split_data(data, seed=8)
    assert all(a.index.equals(b.index) for a, b in zip(first, second, strict=True))
    assert not first[2].index.equals(other[2].index)


def test_split_rejects_invalid_proportions_and_missing_target():
    data = make_transactions()
    with pytest.raises(ValueError, match="Proportions"):
        split_data(data, test_size=0.6, val_size=0.5)
    with pytest.raises(DataValidationError, match="cible"):
        split_data(data.drop(columns=TARGET))


@pytest.mark.parametrize("model_name", ["logistic_regression", "decision_tree"])
def test_train_model_learns_the_signal(model_name):
    X, y = preprocess_data(make_transactions(n=400, n_frauds=40))
    model = train_model(X, y, model_name)
    proba = model.predict_proba(X)[:, 1]
    assert proba[y == 1].mean() > proba[y == 0].mean() + 0.3


def test_train_model_is_deterministic():
    X, y = preprocess_data(make_transactions(n=300, n_frauds=30))
    a = train_model(X, y, "decision_tree", {"max_depth": 4}, seed=1).predict_proba(X)
    b = train_model(X, y, "decision_tree", {"max_depth": 4}, seed=1).predict_proba(X)
    np.testing.assert_array_equal(a, b)


def test_train_model_rejects_single_class():
    X, y = preprocess_data(make_transactions(n=50, n_frauds=0))
    with pytest.raises(DataValidationError, match="deux classes"):
        train_model(X, y)


def test_train_model_rejects_inconsistent_lengths():
    X, y = preprocess_data(make_transactions(n=50, n_frauds=5))
    with pytest.raises(DataValidationError, match="concordent"):
        train_model(X, y[:-1])


def test_train_model_rejects_non_binary_and_empty_target():
    X, y = preprocess_data(make_transactions(n=50, n_frauds=5))
    with pytest.raises(DataValidationError, match="binaire"):
        train_model(X, np.where(y == 1, 2, 0))
    with pytest.raises(DataValidationError, match="vide"):
        train_model(X.iloc[0:0], y.iloc[0:0])


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="inconnu"):
        build_pipeline("random_forest")


# --- Choix du seuil -----------------------------------------------------------------


def test_select_threshold_minimises_cost_on_known_example():
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.65, 0.9])
    # coût FN = 10, FP = 1 ; sans contrainte de FPR
    result = select_threshold(y, p, cost_fn=10, cost_fp=1, max_fpr=1.0)
    # seuil 0.65 : FN = 0, FP = 1 (0.7) -> coût 1 (minimum)
    assert result["threshold"] == pytest.approx(0.65)
    assert result["expected_cost"] == 1
    assert result["recall"] == 1.0
    assert result["fpr"] == pytest.approx(1 / 8)


def test_select_threshold_respects_fpr_constraint():
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.65, 0.9])
    result = select_threshold(y, p, cost_fn=10, cost_fp=1, max_fpr=0.0)
    assert result["fpr"] == 0.0
    assert result["threshold"] == pytest.approx(0.9)  # seul seuil sans faux positif


def test_select_threshold_handles_ties():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.5, 0.5, 0.5, 0.5])
    result = select_threshold(y, p, cost_fn=10, cost_fp=1, max_fpr=1.0)
    assert result["threshold"] == 0.5 and result["recall"] == 1.0


def test_select_threshold_rejects_impossible_constraint_and_bad_input():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.9, 0.1, 0.8, 0.2])  # classement inversé : tout seuil donne un faux positif
    with pytest.raises(ValueError, match="Aucun seuil"):
        select_threshold(y, p, max_fpr=0.0)
    with pytest.raises(ValueError, match="deux classes"):
        select_threshold(np.zeros(4), p)
    with pytest.raises(ValueError, match="longueur"):
        select_threshold(y, p[:2])
