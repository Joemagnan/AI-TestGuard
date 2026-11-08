"""PARTIE VI - Non-régression et reproductibilité (T06).

La version courante du code est rejouée sur l'échantillon versionné avec la
configuration du modèle livré, puis comparée à la référence figée
``tests/reference/baseline_sample.json``. Cette référence n'est jamais régénérée
par les tests : une dégradation au-delà des tolérances fait échouer la CI.
"""

import json

import numpy as np
import pytest

from app.config import (
    BASELINE_PATH,
    FULL_DATA_PATH,
    METADATA_PATH,
    REGRESSION_TOLERANCES,
    SAMPLE_BASELINE_PATH,
    SEED,
)
from app.evaluation import compare_with_baseline
from app.pipeline import fit_and_calibrate, library_versions


@pytest.fixture(scope="module")
def sample_baseline():
    return json.loads(SAMPLE_BASELINE_PATH.read_text(encoding="utf-8"))


def test_current_code_does_not_regress_against_frozen_baseline(
    sample_result, sample_splits, sample_baseline
):
    candidate = {
        "validation_fingerprint": sample_splits["validation_fingerprint"],
        "validation_metrics": sample_result["validation_metrics"],
        "libraries": library_versions(),
    }
    result = compare_with_baseline(candidate, sample_baseline, REGRESSION_TOLERANCES)
    details = json.dumps(result, indent=2, ensure_ascii=False)
    assert result["decision"] != "rejeter", f"Régression détectée :\n{details}"
    assert result["decision"] == "accepter", f"Écart à investiguer :\n{details}"


def test_validation_set_is_the_frozen_one(sample_splits, sample_baseline):
    assert sample_splits["validation_fingerprint"] == sample_baseline["validation_fingerprint"]


def test_threshold_matches_baseline(sample_result, sample_baseline):
    assert sample_result["selection"]["threshold"] == pytest.approx(
        sample_baseline["threshold"], abs=1e-9
    )


def test_retraining_with_same_seed_is_reproducible(sample_splits, production_config):
    first = fit_and_calibrate(sample_splits, **production_config, seed=SEED)
    second = fit_and_calibrate(sample_splits, **production_config, seed=SEED)
    np.testing.assert_allclose(first["val_proba"], second["val_proba"], rtol=0, atol=1e-12)
    assert first["selection"]["threshold"] == second["selection"]["threshold"]


def test_baselines_record_what_is_needed_to_reproduce(sample_baseline):
    full = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    for baseline in (sample_baseline, full):
        for key in (
            "code_version",
            "params",
            "libraries",
            "seed",
            "validation_metrics",
            "threshold",
            "data",
            "validation_fingerprint",
        ):
            assert key in baseline, key


def test_delivered_model_matches_full_baseline():
    """Le modèle livré (models/metadata.json) est celui de la référence figée."""
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    result = compare_with_baseline(metadata, baseline)
    assert result["decision"] == "accepter", result


@pytest.mark.slow
@pytest.mark.full_data
@pytest.mark.skipif(not FULL_DATA_PATH.is_file(), reason="data/creditcard.csv absent")
def test_full_retraining_reproduces_baseline():
    """Ré-entraînement complet (jeu Kaggle) : mêmes métriques de validation."""
    from app.pipeline import load_dataset, make_splits, prepare_dataset

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    data, info = prepare_dataset(load_dataset(FULL_DATA_PATH))
    assert info["fingerprint"] == baseline["data"]["fingerprint"]
    splits = make_splits(data, baseline["seed"])
    result = fit_and_calibrate(
        splits,
        baseline["model_name"],
        baseline["params"],
        baseline["time_encoding"],
        baseline["seed"],
    )
    candidate = {
        "validation_fingerprint": splits["validation_fingerprint"],
        "validation_metrics": result["validation_metrics"],
        "libraries": library_versions(),
    }
    assert compare_with_baseline(candidate, baseline)["decision"] == "accepter"
