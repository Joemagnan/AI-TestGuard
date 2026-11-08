"""Tests de la prédiction (T04) et du chargement contrôlé du modèle."""

import json
import shutil

import joblib
import numpy as np
import pandas as pd
import pytest

from app.config import FEATURES
from app.prediction import (
    ModelLoadingError,
    file_sha256,
    load_model,
    predict_fraud,
    predict_proba,
    save_model,
)
from app.preprocessing import DataValidationError
from tests.factories import transaction_payload


def test_prediction_probabilities_are_valid(sample_bundle, X_test):
    """Exemple du sujet (5.2) : contrat de sortie."""
    probabilities = sample_bundle["pipeline"].predict_proba(X_test)[:, 1]
    assert np.isfinite(probabilities).all()
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)


def test_predict_fraud_applies_threshold_and_keeps_index(sample_bundle, X_test):
    X = X_test.set_index(X_test.index + 1000)
    result = predict_fraud(sample_bundle, X)
    assert list(result.columns) == ["fraud_probability", "is_fraud"]
    assert result.index.equals(X.index)
    expected = (result["fraud_probability"] >= sample_bundle["threshold"]).astype(int)
    assert (result["is_fraud"] == expected).all()


def test_threshold_override_changes_decisions(sample_bundle, X_test):
    flag_all = predict_fraud(sample_bundle, X_test, threshold=0.0)
    flag_none = predict_fraud(sample_bundle["pipeline"], X_test, threshold=1.0 + 0.0)
    assert flag_all["is_fraud"].all()
    assert flag_none["is_fraud"].sum() == (flag_none["fraud_probability"] >= 1.0).sum()


def test_prediction_on_single_transaction_dict_and_list(sample_bundle):
    single = predict_fraud(sample_bundle, transaction_payload())
    batch = predict_fraud(sample_bundle, [transaction_payload(), transaction_payload(Amount=5.0)])
    assert len(single) == 1 and len(batch) == 2
    assert single["fraud_probability"].iloc[0] == pytest.approx(batch["fraud_probability"].iloc[0])


def test_prediction_ignores_target_and_extra_columns(sample_bundle, sample_splits):
    frame = sample_splits["test"]["frame"].assign(note="x")
    with_target = predict_fraud(sample_bundle, frame)
    without = predict_fraud(sample_bundle, frame[FEATURES])
    pd.testing.assert_frame_equal(with_target, without)


def test_prediction_rejects_invalid_schema(sample_bundle, X_test):
    with pytest.raises(DataValidationError, match="absentes"):
        predict_fraud(sample_bundle, X_test.drop(columns="V4"))
    with pytest.raises(TypeError):
        predict_fraud(sample_bundle, "transaction")


def test_threshold_is_required_and_bounded(sample_bundle, X_test):
    with pytest.raises(ValueError, match="obligatoire"):
        predict_fraud(sample_bundle["pipeline"], X_test)
    with pytest.raises(ValueError, match="entre 0 et 1"):
        predict_fraud(sample_bundle, X_test, threshold=1.5)


def test_output_contract_violation_is_detected(X_test):
    class BrokenModel:
        def predict_proba(self, X):
            return np.column_stack([np.zeros(len(X)), np.full(len(X), np.nan)])

    with pytest.raises(RuntimeError, match="Contrat de sortie"):
        predict_proba(BrokenModel(), X_test)


# --- Chargement contrôlé ----------------------------------------------------------------


def test_save_and_load_round_trip(saved_model, sample_bundle, X_test):
    bundle = load_model(saved_model)
    assert bundle["threshold"] == sample_bundle["threshold"]
    np.testing.assert_allclose(
        predict_proba(bundle, X_test), predict_proba(sample_bundle, X_test), rtol=0, atol=1e-12
    )
    metadata = json.loads(saved_model.with_name("metadata.json").read_text(encoding="utf-8"))
    assert metadata["artifact_sha256"] == file_sha256(saved_model)


def test_missing_model_file_is_reported(tmp_path):
    with pytest.raises(ModelLoadingError, match="introuvable"):
        load_model(tmp_path / "absent.joblib")


def test_model_without_metadata_is_refused(tmp_path, saved_model):
    alone = tmp_path / "fraud_model.joblib"
    shutil.copy(saved_model, alone)
    with pytest.raises(ModelLoadingError, match="metadata.json absent"):
        load_model(alone)


def test_tampered_model_is_refused(tmp_path, saved_model):
    folder = tmp_path / "copie"
    shutil.copytree(saved_model.parent, folder)
    target = folder / saved_model.name
    target.write_bytes(target.read_bytes() + b"\x00")
    with pytest.raises(ModelLoadingError, match="SHA-256"):
        load_model(target)


def test_corrupted_file_with_matching_hash_is_reported(tmp_path):
    target = tmp_path / "fraud_model.joblib"
    target.write_bytes(b"pas un fichier joblib")
    with pytest.raises(ModelLoadingError, match="Chargement impossible"):
        load_model(target, expected_sha256=file_sha256(target))


def test_invalid_metadata_is_reported(tmp_path, saved_model):
    folder = tmp_path / "meta"
    shutil.copytree(saved_model.parent, folder)
    (folder / "metadata.json").write_text("{pas du json", encoding="utf-8")
    with pytest.raises(ModelLoadingError, match="invalide"):
        load_model(folder / saved_model.name)


@pytest.mark.parametrize(
    "content,message",
    [
        ({"pipeline": object(), "threshold": 0.5}, "predict_proba"),
        ({"model": "x"}, "Format inattendu"),
        ("chaine", "Format inattendu"),
    ],
)
def test_unexpected_bundle_format_is_refused(tmp_path, content, message):
    target = tmp_path / "fraud_model.joblib"
    joblib.dump(content, target)
    with pytest.raises(ModelLoadingError, match=message):
        load_model(target, expected_sha256=file_sha256(target))


def test_invalid_threshold_in_bundle_is_refused(tmp_path, sample_bundle):
    target = tmp_path / "fraud_model.joblib"
    joblib.dump({"pipeline": sample_bundle["pipeline"], "threshold": 3.0}, target)
    with pytest.raises(ModelLoadingError, match="Seuil invalide"):
        load_model(target, expected_sha256=file_sha256(target))


def test_save_model_rejects_invalid_threshold(tmp_path, sample_bundle):
    with pytest.raises(ValueError, match="seuil"):
        save_model(sample_bundle["pipeline"], -0.1, {}, tmp_path / "m.joblib")
