"""Fixtures partagées.

Les tests rapides n'utilisent que des données synthétiques construites ici et
l'échantillon versionné ``data/sample/creditcard_sample.csv`` : ils s'exécutent
en CI sans le jeu Kaggle complet.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from app.config import METADATA_PATH, SAMPLE_DATA_PATH, SEED
from app.monitoring import build_reference_profile
from app.pipeline import fit_and_calibrate, load_dataset, make_splits, prepare_dataset
from app.prediction import save_model
from tests.factories import make_transactions


@pytest.fixture
def transactions() -> pd.DataFrame:
    return make_transactions()


@pytest.fixture(scope="session")
def production_config() -> dict:
    """Configuration du modèle livré (models/metadata.json)."""
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return {
        "model_name": metadata["model_name"],
        "params": metadata["params"],
        "time_encoding": metadata["time_encoding"],
    }


@pytest.fixture(scope="session")
def sample_splits() -> dict:
    data, _ = prepare_dataset(load_dataset(SAMPLE_DATA_PATH))
    return make_splits(data, SEED)


@pytest.fixture(scope="session")
def sample_result(sample_splits, production_config) -> dict:
    return fit_and_calibrate(sample_splits, **production_config, seed=SEED)


@pytest.fixture(scope="session")
def sample_bundle(sample_result, sample_splits) -> dict:
    """Bundle équivalent au modèle livré, entraîné sur l'échantillon."""
    return {
        "pipeline": sample_result["pipeline"],
        "threshold": sample_result["selection"]["threshold"],
        "metadata": {
            "version": "test",
            "model_name": sample_result["model_name"],
            "reference_profile": build_reference_profile(sample_splits["train"]["X"]),
            "validation_metrics": sample_result["validation_metrics"],
        },
    }


@pytest.fixture(scope="session")
def saved_model(tmp_path_factory, sample_bundle):
    """Modèle sauvegardé sur disque (bundle joblib + metadata.json)."""
    path = tmp_path_factory.mktemp("model") / "fraud_model.joblib"
    save_model(
        sample_bundle["pipeline"], sample_bundle["threshold"], sample_bundle["metadata"], path
    )
    return path


@pytest.fixture
def X_test(sample_splits) -> pd.DataFrame:
    return sample_splits["test"]["X"].copy()
