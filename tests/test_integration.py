"""PARTIE IV - Tests d'intégration et prévention des fuites de données.

Scénario complet sur l'échantillon versionné : chargement, contrôle du schéma,
découpage, ajustement sur l'entraînement, seuil sur la validation, sauvegarde,
rechargement, prédiction sur le test et rapport de métriques.
"""

import json

import numpy as np
import pandas as pd
import pytest

from app.config import BASELINE_PATH, FEATURES, PCA_FEATURES, SAMPLE_DATA_PATH, TARGET
from app.pipeline import load_dataset, main, prepare_dataset, run_training
from app.prediction import file_sha256, load_model, predict_fraud, predict_proba
from app.preprocessing import DataValidationError

SMALL_GRIDS = {
    "logistic_regression": [{"C": 0.1, "class_weight": None}],
    "decision_tree": [{"max_depth": 4, "min_samples_leaf": 20, "class_weight": None}],
}


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("run")
    report = run_training(
        SAMPLE_DATA_PATH, root / "models", root / "reports", grids=SMALL_GRIDS, n_boot=50
    )
    return root, report


def test_end_to_end_produces_artifacts_and_report(trained):
    root, report = trained
    assert (root / "models" / "fraud_model.joblib").is_file()
    assert (root / "models" / "metadata.json").is_file()
    assert (root / "reports" / "metrics.json").is_file()
    for name in ("roc_pr_test.png", "threshold_validation.png", "confusion_validation.png"):
        assert (root / "reports" / "figures" / name).stat().st_size > 10_000
    saved = json.loads((root / "reports" / "metrics.json").read_text(encoding="utf-8"))
    assert saved["selected_model"] in {"logistic_regression", "decision_tree"}
    assert set(saved["candidates"]) == {"logistic_regression", "decision_tree"}
    assert 0 <= saved["threshold"] <= 1


def test_split_sizes_and_fraud_counts_are_consistent(trained):
    _, report = trained
    sizes, frauds = report["split_sizes"], report["split_frauds"]
    assert sum(sizes.values()) == report["data"]["rows"]
    assert sum(frauds.values()) == report["data"]["frauds"]
    assert sizes["train"] == pytest.approx(0.6 * report["data"]["rows"], abs=2)


def test_reloaded_pipeline_gives_identical_predictions(trained, sample_splits):
    root, _ = trained
    bundle = load_model(root / "models" / "fraud_model.joblib")
    X = sample_splits["test"]["X"]
    before = bundle["pipeline"].predict_proba(X)[:, 1]
    reloaded = load_model(root / "models" / "fraud_model.joblib")
    after = predict_proba(reloaded, X)
    np.testing.assert_allclose(before, after, rtol=0, atol=1e-12)


def test_scaler_is_fitted_on_training_data_only(sample_splits, sample_result):
    """Absence de fuite : les statistiques du scaler sont celles de l'entraînement."""
    columns = sample_result["pipeline"].named_steps["preprocess"].named_steps["columns"]
    scaler = columns.named_transformers_["pca"].named_steps["scale"]
    train = sample_splits["train"]["X"][PCA_FEATURES]
    everything = pd.concat([train, sample_splits["test"]["X"][PCA_FEATURES]])
    np.testing.assert_allclose(scaler.mean_, train.mean().to_numpy(), rtol=1e-9)
    assert not np.allclose(scaler.mean_, everything.mean().to_numpy(), rtol=1e-6)
    assert np.all(np.ravel(scaler.n_samples_seen_) == len(train))


def test_partitions_do_not_overlap(sample_splits):
    keys = {}
    for name in ("train", "val", "test"):
        frame = sample_splits[name]["frame"]
        keys[name] = set(pd.util.hash_pandas_object(frame[FEATURES], index=False))
    assert not keys["train"] & keys["test"]
    assert not keys["train"] & keys["val"]
    assert not keys["val"] & keys["test"]


def test_duplicates_are_removed_before_split():
    raw = load_dataset(SAMPLE_DATA_PATH)
    doubled = pd.concat([raw, raw.iloc[:30]], ignore_index=True)
    data, info = prepare_dataset(doubled)
    assert info["duplicates_removed"] == 30
    assert len(data) == len(raw)


def test_metadata_traces_versions_seed_data_and_threshold(trained):
    root, _ = trained
    meta = json.loads((root / "models" / "metadata.json").read_text(encoding="utf-8"))
    for key in (
        "version",
        "code_version",
        "seed",
        "params",
        "threshold",
        "threshold_rule",
        "validation_fingerprint",
        "validation_metrics",
        "libraries",
        "reference_profile",
    ):
        assert key in meta
    assert meta["data"]["source_sha256"] == file_sha256(SAMPLE_DATA_PATH)
    assert {"python", "scikit-learn", "numpy", "pandas"} <= set(meta["libraries"])
    assert meta["features"] == FEATURES


def test_test_metrics_are_reported_but_threshold_comes_from_validation(trained):
    _, report = trained
    selected = report["candidates"][report["selected_model"]]
    assert (
        report["threshold"] == selected["threshold"] == report["threshold_selection"]["threshold"]
    )
    assert report["threshold_selection"]["rule"]["max_fpr"] == 0.005


def test_training_run_never_modifies_the_baseline(tmp_path):
    before = file_sha256(BASELINE_PATH)
    run_training(
        SAMPLE_DATA_PATH,
        tmp_path / "m",
        tmp_path / "r",
        grids=SMALL_GRIDS,
        n_boot=10,
        make_plots=False,
    )
    assert file_sha256(BASELINE_PATH) == before


def test_command_line_entry_point(tmp_path, capsys):
    code = main(
        [
            "--data",
            str(SAMPLE_DATA_PATH),
            "--models-dir",
            str(tmp_path / "m"),
            "--reports-dir",
            str(tmp_path / "r"),
            "--no-plots",
        ]
    )
    assert code == 0
    assert "Modèle retenu" in capsys.readouterr().out


# --- Gestion explicite des erreurs de chargement et d'entrée --------------------------------


def test_missing_data_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="introuvable"):
        load_dataset(tmp_path / "absent.csv")


def test_empty_or_malformed_csv_is_reported(tmp_path):
    empty = tmp_path / "vide.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(DataValidationError, match="Lecture"):
        load_dataset(empty)
    broken = tmp_path / "casse.csv"
    broken.write_text('a,b\n1,"2\n', encoding="utf-8")
    with pytest.raises(DataValidationError, match="Lecture"):
        load_dataset(broken)


def test_csv_with_wrong_schema_is_rejected(tmp_path):
    wrong = tmp_path / "schema.csv"
    load_dataset(SAMPLE_DATA_PATH).drop(columns="V9").to_csv(wrong, index=False)
    with pytest.raises(DataValidationError, match="V9"):
        prepare_dataset(load_dataset(wrong))


def test_saved_model_scores_raw_csv_rows(trained, tmp_path):
    root, _ = trained
    rows = load_dataset(SAMPLE_DATA_PATH).head(25).drop(columns=TARGET)
    rows.loc[0, "V3"] = np.nan  # traité par imputation dans le pipeline sauvegardé
    result = predict_fraud(load_model(root / "models" / "fraud_model.joblib"), rows)
    assert len(result) == 25
    assert result["fraud_probability"].between(0, 1).all()
