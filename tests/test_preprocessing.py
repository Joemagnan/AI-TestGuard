"""PARTIE II - Tests unitaires du contrat de données et du prétraitement.

Chaque test vérifie un comportement attendu (rejet explicite ou correction
documentée), pas seulement l'absence d'exception.
"""

import logging

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from sklearn.preprocessing import StandardScaler

from app.config import FEATURES, PCA_FEATURES, TARGET
from app.preprocessing import (
    DataValidationError,
    build_preprocessor,
    canonical_order,
    clean_features,
    data_fingerprint,
    deduplicate,
    log_amount,
    preprocess_data,
    time_to_cyclic,
    validate_data,
)
from tests.factories import make_transactions

# --- Exemple du sujet (2.2) ------------------------------------------------------------


def test_standard_scaler():
    data = pd.DataFrame({"Amount": [100.0, 200.0, 300.0]})
    scaled = StandardScaler().fit_transform(data)
    assert np.isfinite(scaled).all()
    assert abs(scaled.mean()) < 1e-10
    assert np.isclose(scaled.std(), 1.0)


# --- T01 : colonnes requises ----------------------------------------------------------


def test_valid_data_keeps_all_contract_columns_in_order(transactions):
    shuffled = transactions[list(reversed(transactions.columns))]
    out = validate_data(shuffled)
    assert list(out.columns) == [*FEATURES, TARGET]
    assert len(out) == len(transactions)


@pytest.mark.parametrize("column", ["Time", "V14", "Amount", TARGET])
def test_missing_required_column_is_rejected(transactions, column):
    with pytest.raises(DataValidationError, match="absentes"):
        validate_data(transactions.drop(columns=column))


def test_target_not_required_for_inference(transactions):
    out = validate_data(transactions.drop(columns=TARGET), require_target=False)
    assert list(out.columns) == FEATURES


def test_duplicated_column_is_rejected(transactions):
    duplicated = pd.concat([transactions, transactions[["V3"]]], axis=1)
    with pytest.raises(DataValidationError, match="dupliquées.*V3"):
        validate_data(duplicated)


def test_extra_column_is_dropped_and_logged(transactions, caplog):
    extended = transactions.assign(card_id="0000")
    with caplog.at_level(logging.WARNING, logger="app.preprocessing"):
        out = validate_data(extended)
    assert "card_id" not in out.columns
    assert "card_id" in caplog.text


# --- T02 : types ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    ["12.5", "abc", True],
    ids=["chaine_numerique", "chaine", "booleen"],
)
def test_wrong_type_is_rejected(transactions, value):
    corrupted = transactions.copy()
    corrupted["V5"] = corrupted["V5"].astype(object)
    corrupted.loc[:, "V5"] = value
    with pytest.raises(DataValidationError, match="Type incorrect.*V5"):
        validate_data(corrupted)


def test_boolean_column_is_rejected(transactions):
    corrupted = transactions.assign(Amount=transactions["Amount"] > 10)
    with pytest.raises(DataValidationError, match="Amount"):
        validate_data(corrupted)


def test_non_dataframe_input_is_rejected():
    with pytest.raises(DataValidationError, match="DataFrame"):
        validate_data([[1, 2, 3]])


# --- Cible -----------------------------------------------------------------------------


@pytest.mark.parametrize("bad_value", [2, -1, 0.5])
def test_target_outside_zero_one_is_rejected(transactions, bad_value):
    corrupted = transactions.copy()
    corrupted[TARGET] = corrupted[TARGET].astype(float)
    corrupted.loc[0, TARGET] = bad_value
    with pytest.raises(DataValidationError, match="cible invalides"):
        validate_data(corrupted)


def test_missing_target_is_rejected(transactions):
    corrupted = transactions.copy()
    corrupted[TARGET] = corrupted[TARGET].astype(float)
    corrupted.loc[3, TARGET] = np.nan
    with pytest.raises(DataValidationError, match="manquantes"):
        validate_data(corrupted)


def test_float_target_zero_one_is_accepted_and_cast(transactions):
    data = transactions.assign(**{TARGET: transactions[TARGET].astype(float)})
    _, y = preprocess_data(data)
    assert y.dtype == "int64"
    assert set(y.unique()) == {0, 1}


# --- Valeurs vides, manquantes, infinies, négatives ------------------------------------------


def test_empty_dataframe_is_rejected(transactions):
    with pytest.raises(DataValidationError, match="vide"):
        validate_data(transactions.iloc[0:0])


def test_missing_feature_values_are_kept_for_imputation(transactions):
    data = transactions.copy()
    data.loc[[1, 2], "V7"] = np.nan
    X, _ = preprocess_data(data)
    assert X["V7"].isna().sum() == 2  # correction prévue plus loin (imputation)


def test_infinite_values_are_converted_to_missing(transactions):
    data = transactions.copy()
    data.loc[0, "V1"] = np.inf
    data.loc[1, "Amount"] = -np.inf
    X, _ = preprocess_data(data)
    assert np.isnan(X.loc[0, "V1"]) and np.isnan(X.loc[1, "Amount"])
    assert not np.isinf(X.to_numpy()).any()


@pytest.mark.parametrize("column", ["Amount", "Time"])
def test_negative_amount_or_time_is_rejected(transactions, column):
    data = transactions.copy()
    data.loc[5, column] = -0.01
    with pytest.raises(DataValidationError, match="négatives"):
        validate_data(data)


# --- Observation unique et dimensions ---------------------------------------------------


def test_single_observation_is_processed(transactions):
    X, y = preprocess_data(transactions.iloc[[0]])
    assert X.shape == (1, len(FEATURES))
    assert len(y) == 1


@pytest.mark.parametrize("encoding,n_out", [("raw", 30), ("cyclic", 31)])
def test_dimensions_after_transformation(transactions, encoding, n_out):
    X, _ = preprocess_data(transactions)
    out = build_preprocessor(encoding).fit(X).transform(X)
    assert out.shape == (len(X), n_out)


def test_unknown_time_encoding_is_rejected():
    with pytest.raises(ValueError, match="inconnu"):
        build_preprocessor("weekday")


# --- T03 : aucune valeur non finie en sortie du prétraitement --------------------------------


def test_preprocessor_output_is_finite_despite_nan_and_inf(transactions):
    X, _ = preprocess_data(transactions)
    prep = build_preprocessor().fit(X)
    dirty = transactions.drop(columns=TARGET).copy()
    dirty.loc[0, "V1"] = np.nan
    dirty.loc[1, "V2"] = np.inf
    dirty.loc[2, PCA_FEATURES] = np.nan  # ligne entièrement vide sur les composantes
    out = prep.transform(clean_features(dirty))
    assert np.isfinite(out).all()


def test_imputation_uses_training_median(transactions):
    X, _ = preprocess_data(transactions)
    prep = build_preprocessor().fit(X)
    imputer = prep.named_steps["columns"].named_transformers_["pca"].named_steps["impute"]
    assert np.allclose(imputer.statistics_, X[PCA_FEATURES].median().to_numpy())


def test_clean_features_rejects_bad_matrix_shape():
    with pytest.raises(DataValidationError, match="forme"):
        clean_features(np.zeros((3, 5)))


def test_clean_features_accepts_matrix_with_contract_width():
    out = clean_features(np.zeros((2, len(FEATURES))))
    assert list(out.columns) == FEATURES


def test_clean_features_rejects_non_convertible_values(transactions):
    data = transactions.drop(columns=TARGET).astype({"V2": object})
    data.loc[0, "V2"] = "x"
    with pytest.raises(DataValidationError, match="Conversion"):
        clean_features(data)


# --- Transformations élémentaires -----------------------------------------------------------


def test_log_amount_is_monotonic_and_bounded_at_zero():
    values = np.array([-5.0, 0.0, 1.0, 100.0, 25_000.0])
    out = log_amount(values)
    assert out[0] == 0.0 and out[1] == 0.0
    assert np.all(np.diff(out[1:]) > 0)


def test_time_to_cyclic_has_24h_period_and_unit_norm():
    t = np.array([[0.0], [3600.0 * 6], [86_400.0], [90_000.0]])
    out = time_to_cyclic(t)
    assert np.allclose(out[0], out[2])  # même heure, jour suivant
    assert np.allclose((out**2).sum(axis=1), 1.0)
    assert np.allclose(out[1], [1.0, 0.0], atol=1e-12)  # 6 h -> sin = 1


# --- Doublons et empreinte -----------------------------------------------------------------


def test_deduplicate_removes_exact_duplicates(transactions):
    doubled = pd.concat([transactions, transactions.iloc[:5]], ignore_index=True)
    unique, report = deduplicate(doubled)
    assert len(unique) == len(transactions)
    assert report["duplicates_removed"] == 5
    assert report["conflicting_labels"] == 0


def test_deduplicate_counts_conflicting_labels(transactions):
    conflict = transactions.iloc[[0]].copy()
    conflict[TARGET] = 1 - conflict[TARGET]
    _, report = deduplicate(pd.concat([transactions, conflict], ignore_index=True))
    assert report["conflicting_labels"] == 1


def test_fingerprint_is_independent_of_row_order(transactions):
    shuffled = transactions.sample(frac=1.0, random_state=3)
    assert data_fingerprint(transactions) == data_fingerprint(shuffled)
    changed = transactions.copy()
    changed.loc[0, "Amount"] += 0.01
    assert data_fingerprint(changed) != data_fingerprint(transactions)


def test_canonical_order_is_idempotent(transactions):
    once = canonical_order(transactions.sample(frac=1.0, random_state=1))
    pd.testing.assert_frame_equal(once, canonical_order(once))


# --- Propriété : tout tableau numérique fini est préparé sans valeur non finie ---------------

finite = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
REFERENCE_PREPROCESSOR = build_preprocessor().fit(preprocess_data(make_transactions())[0])


@settings(max_examples=40, deadline=None)
@given(
    values=arrays(np.float64, st.tuples(st.integers(1, 20), st.just(28)), elements=finite),
    amount=st.floats(min_value=0, max_value=1e7, allow_nan=False),
    time=st.floats(min_value=0, max_value=1e7, allow_nan=False),
)
def test_property_finite_inputs_give_finite_outputs(values, amount, time):
    frame = pd.DataFrame(values, columns=PCA_FEATURES)
    frame.insert(0, "Time", time)
    frame["Amount"] = amount
    X, _ = preprocess_data(frame, require_target=False)
    out = REFERENCE_PREPROCESSOR.transform(X)
    assert out.shape == (len(values), 30)
    assert np.isfinite(out).all()
