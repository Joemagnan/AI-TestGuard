"""PARTIE VIII - Sécurité opérationnelle de l'API : validation des entrées et des
fichiers, codes d'erreur explicites, journaux sans données sensibles."""

import io
import json
import logging

import pytest
from fastapi.testclient import TestClient

from app import api
from app.config import FEATURES, SAMPLE_DATA_PATH
from app.pipeline import load_dataset
from tests.factories import transaction_payload


@pytest.fixture
def client(saved_model, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(saved_model))
    with TestClient(api.app) as test_client:
        yield test_client


@pytest.fixture
def csv_bytes():
    frame = load_dataset(SAMPLE_DATA_PATH).head(40)
    return frame.to_csv(index=False).encode("utf-8")


def test_health_and_model_info(client, sample_bundle):
    assert client.get("/health").json() == {"status": "UP", "model_version": "test"}
    info = client.get("/model/info").json()
    assert info["threshold"] == sample_bundle["threshold"]
    assert info["features"] is None or info["features"] == FEATURES


def test_predict_valid_transaction(client, sample_bundle):
    response = client.post("/predict", json=transaction_payload())
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["fraud_probability"] <= 1
    assert body["is_fraud"] == (body["fraud_probability"] >= sample_bundle["threshold"])
    assert body["alerts"] == []
    assert response.headers["X-Request-ID"]


def test_out_of_domain_amount_is_scored_with_alert(client):
    body = client.post("/predict", json=transaction_payload(Amount=1e7)).json()
    assert "montant_hors_domaine" in body["alerts"]


def test_integer_values_are_accepted_as_numbers(client):
    assert client.post("/predict", json=transaction_payload(Amount=50, Time=10)).status_code == 200


@pytest.mark.parametrize(
    "change",
    [
        {"Amount": "49.9"},
        {"V3": True},
        {"V3": None},
        {"Amount": -1.0},
        {"Time": -5.0},
        {"card_number": "4970100000000000"},
    ],
    ids=["chaine", "booleen", "null", "montant_negatif", "temps_negatif", "champ_supplementaire"],
)
def test_invalid_transactions_are_rejected(client, change):
    response = client.post("/predict", json=transaction_payload(**change))
    assert response.status_code == 422


def test_missing_field_is_rejected(client):
    payload = transaction_payload()
    del payload["V17"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "V17" in response.text


def test_non_finite_values_are_rejected(client):
    raw = json.dumps(transaction_payload()).replace('"Amount": 49.9', '"Amount": NaN')
    response = client.post("/predict", content=raw, headers={"Content-Type": "application/json"})
    assert response.status_code == 422  # anomalie A01 : renvoyait 500 avant correction
    assert "nan" not in response.text.lower()


def test_error_responses_do_not_echo_submitted_values(client):
    response = client.post("/predict", json=transaction_payload(Amount="9876.54"))
    assert response.status_code == 422
    assert "9876.54" not in response.text


def test_batch_prediction_and_size_limit(client):
    ok = client.post("/predict/batch", json={"transactions": [transaction_payload()] * 3})
    assert ok.status_code == 200
    assert ok.json()["n_transactions"] == 3
    too_big = client.post("/predict/batch", json={"transactions": [transaction_payload()] * 1001})
    assert too_big.status_code == 422
    assert client.post("/predict/batch", json={"transactions": []}).status_code == 422


def test_csv_file_prediction_reports_quality(client, csv_bytes):
    response = client.post("/predict/file", files={"file": ("lot.csv", csv_bytes, "text/csv")})
    assert response.status_code == 200
    body = response.json()
    assert body["n_transactions"] == 40
    assert body["quality"]["missing_rate"] == 0.0


def test_csv_with_missing_values_is_treated_and_flagged(client):
    frame = load_dataset(SAMPLE_DATA_PATH).head(20)
    frame.loc[:4, "V10"] = None
    content = frame.to_csv(index=False).encode("utf-8")
    body = client.post("/predict/file", files={"file": ("lot.csv", content, "text/csv")}).json()
    assert body["quality"]["missing_rate"] == pytest.approx(0.25)
    assert body["quality"]["missing_alert"] is True
    assert body["quality"]["alert_counts"]["valeurs_manquantes"] == 5


@pytest.mark.parametrize(
    "filename,content,status",
    [
        ("lot.txt", b"Time,V1\n1,2\n", 415),
        ("lot.csv", b"", 422),
        ("lot.csv", b"Time,V1\n1,2\n", 422),
        ("lot.csv", b"\xff\xfe\x00\x81", 422),
        ("lot.csv", b"a" * (api.MAX_FILE_BYTES + 10), 413),
    ],
    ids=["extension", "vide", "schema", "encodage", "taille"],
)
def test_invalid_files_are_rejected(client, filename, content, status):
    response = client.post(
        "/predict/file", files={"file": (filename, io.BytesIO(content), "text/csv")}
    )
    assert response.status_code == status


def test_logs_contain_no_transaction_values(client, caplog):
    secret_amount = 1234.56789
    payload = transaction_payload(Amount=secret_amount, V14=-7.654321)
    with caplog.at_level(logging.INFO, logger="ai_testguard.api"):
        response = client.post("/predict", json=payload)
    probability = response.json()["fraud_probability"]
    logs = caplog.text
    assert '"path": "/predict"' in logs
    assert "1234.56789" not in logs and "7.654321" not in logs
    assert str(probability) not in logs
    entry = json.loads(caplog.records[-1].getMessage())
    assert set(entry) == {
        "event",
        "request_id",
        "method",
        "path",
        "status",
        "latency_ms",
        "model_version",
        "n_transactions",
        "n_flagged",
    }


def test_service_unavailable_when_model_cannot_be_loaded(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "absent.joblib"))
    with TestClient(api.app) as client:
        assert client.get("/health").status_code == 503
        assert client.post("/predict", json=transaction_payload()).status_code == 503
        assert client.get("/model/info").status_code == 503
