"""API de scoring (FastAPI).

Politique d'entrée :

* ``POST /predict`` (une transaction JSON) : contrat strict. Champ manquant,
  champ supplémentaire, type incorrect (chaîne, booléen, null), valeur non finie,
  montant ou temps négatif -> rejet 422.
* ``POST /predict/batch`` (liste JSON, 1 000 transactions maximum) : même contrat.
* ``POST /predict/file`` (fichier CSV, 2 Mo maximum) : schéma contrôlé par
  ``validate_data`` ; les valeurs manquantes sont imputées et signalées par une
  alerte de qualité au lieu d'être rejetées.

Journalisation : une ligne JSON par requête, sans aucune valeur de transaction
(ni montant, ni composantes V, ni probabilité individuelle).

Lancement : ``uvicorn app.api:app --port 8000``
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, create_model

from app.config import MODEL_PATH, PCA_FEATURES
from app.monitoring import batch_quality_report, transaction_alerts
from app.prediction import ModelLoadingError, load_model, predict_fraud
from app.preprocessing import DataValidationError

MAX_BATCH = 1000
MAX_FILE_BYTES = 2 * 1024 * 1024

logger = logging.getLogger("ai_testguard.api")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

_STRICT = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

Transaction = create_model(
    "Transaction",
    __config__=_STRICT,
    Time=(float, Field(ge=0, description="Secondes depuis la première transaction")),
    **{name: (float, Field(description="Composante anonymisée")) for name in PCA_FEATURES},
    Amount=(float, Field(ge=0, description="Montant en euros")),
)


class Batch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transactions: list[Transaction] = Field(min_length=1, max_length=MAX_BATCH)


class Prediction(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold: float
    model_version: str
    alerts: list[str]


STATE: dict = {"bundle": None, "error": None}


def load_state(model_path: str | Path | None = None) -> None:
    path = Path(model_path or os.getenv("MODEL_PATH", MODEL_PATH))
    try:
        STATE["bundle"], STATE["error"] = load_model(path), None
    except ModelLoadingError as exc:
        STATE["bundle"], STATE["error"] = None, str(exc)
        logger.error(json.dumps({"event": "model_loading_failed", "error": str(exc)}))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_state()
    yield


app = FastAPI(
    title="AI-TestGuard - scoring de fraude",
    version="1.0.0",
    description="Probabilité de fraude d'une transaction par carte (prototype pédagogique).",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Rejet 422 sans renvoyer les valeurs reçues.

    Anomalie A01 (corrigée) : par défaut, FastAPI recopie la valeur fautive dans la
    réponse ; une valeur NaN rendait alors la réponse non sérialisable (erreur 500
    au lieu de 422). Ne pas renvoyer les valeurs évite aussi de les exposer.
    """
    errors = [
        {
            "type": err.get("type"),
            "loc": [str(part) for part in err.get("loc", ())],
            "msg": err.get("msg"),
        }
        for err in exc.errors()
    ]
    if hasattr(request.state, "log"):
        request.state.log["validation_errors"] = len(errors)
    return JSONResponse(status_code=422, content={"detail": errors})


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = uuid.uuid4().hex[:12]
    request.state.log = {}
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        entry = {
            "event": "request",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": status,
            "latency_ms": round((time.perf_counter() - start) * 1000, 3),
            **request.state.log,
        }
        logger.info(json.dumps(entry, ensure_ascii=False))


def _bundle() -> dict:
    if STATE["bundle"] is None:
        raise HTTPException(status_code=503, detail="Modèle indisponible.")
    return STATE["bundle"]


def _score(frame: pd.DataFrame, request: Request) -> tuple[pd.DataFrame, list[list[str]], dict]:
    bundle = _bundle()
    try:
        result = predict_fraud(bundle, frame)
    except DataValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    profile = bundle["metadata"].get("reference_profile")
    alerts = transaction_alerts(frame, profile) if profile else [[] for _ in range(len(frame))]
    request.state.log.update(
        {
            "model_version": bundle["metadata"].get("version"),
            "n_transactions": int(len(result)),
            "n_flagged": int(result["is_fraud"].sum()),
        }
    )
    return result, alerts, bundle


@app.get("/health")
def health():
    if STATE["bundle"] is None:
        raise HTTPException(status_code=503, detail=STATE["error"] or "Modèle non chargé.")
    return {"status": "UP", "model_version": STATE["bundle"]["metadata"].get("version")}


@app.get("/model/info")
def model_info():
    bundle = _bundle()
    meta = bundle["metadata"]
    return {
        "version": meta.get("version"),
        "model_name": meta.get("model_name"),
        "threshold": bundle["threshold"],
        "threshold_rule": meta.get("threshold_rule"),
        "features": meta.get("features"),
        "validation_metrics": {
            k: meta.get("validation_metrics", {}).get(k)
            for k in ("recall", "precision", "fpr", "pr_auc")
        },
        "created_at": meta.get("created_at"),
    }


@app.post("/predict", response_model=Prediction)
def predict(transaction: Transaction, request: Request):
    frame = pd.DataFrame([transaction.model_dump()])
    result, alerts, bundle = _score(frame, request)
    return Prediction(
        fraud_probability=float(result["fraud_probability"].iloc[0]),
        is_fraud=bool(result["is_fraud"].iloc[0]),
        threshold=bundle["threshold"],
        model_version=bundle["metadata"].get("version", "inconnue"),
        alerts=alerts[0],
    )


@app.post("/predict/batch")
def predict_batch(batch: Batch, request: Request):
    frame = pd.DataFrame([t.model_dump() for t in batch.transactions])
    result, alerts, bundle = _score(frame, request)
    return {
        "model_version": bundle["metadata"].get("version"),
        "threshold": bundle["threshold"],
        "n_transactions": int(len(result)),
        "n_flagged": int(result["is_fraud"].sum()),
        "predictions": [
            {"fraud_probability": float(p), "is_fraud": bool(f), "alerts": a}
            for p, f, a in zip(result["fraud_probability"], result["is_fraud"], alerts, strict=True)
        ],
    }


@app.post("/predict/file")
async def predict_file(request: Request, file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="Seuls les fichiers .csv sont acceptés.")
    content = await file.read(MAX_FILE_BYTES + 1)
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="Fichier trop volumineux (2 Mo maximum).")
    if not content.strip():
        raise HTTPException(status_code=422, detail="Fichier vide.")
    try:
        frame = pd.read_csv(io.BytesIO(content))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=422, detail=f"CSV illisible : {exc}") from exc
    if len(frame) > MAX_BATCH * 10:
        raise HTTPException(status_code=413, detail="Trop de lignes (10 000 maximum).")
    result, _alerts, bundle = _score(frame, request)
    profile = bundle["metadata"].get("reference_profile")
    quality = batch_quality_report(frame, profile) if profile else {}
    return {
        "model_version": bundle["metadata"].get("version"),
        "n_transactions": int(len(result)),
        "n_flagged": int(result["is_fraud"].sum()),
        "flag_rate": float(result["is_fraud"].mean()),
        "quality": quality,
        "fraud_probability": [float(p) for p in result["fraud_probability"]],
    }
