"""Sauvegarde, chargement contrôlé et prédiction.

Le modèle est sauvegardé sous la forme d'un « bundle » joblib :
``{"pipeline": Pipeline, "threshold": float, "metadata": dict}``.

Un fichier joblib (pickle) peut exécuter du code au chargement : on ne charge donc
qu'un fichier dont l'empreinte SHA-256 correspond à celle enregistrée dans
``metadata.json`` lors de l'entraînement.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.preprocessing import preprocess_data


class ModelLoadingError(RuntimeError):
    """Le modèle est absent, corrompu, modifié ou ne respecte pas le format attendu."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_model(pipeline, threshold: float, metadata: dict, model_path: str | Path) -> dict:
    """Sauvegarde le bundle et écrit ``metadata.json`` (avec l'empreinte du fichier)."""
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("Le seuil doit être compris entre 0 et 1.")
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"pipeline": pipeline, "threshold": float(threshold), "metadata": metadata}
    joblib.dump(bundle, model_path)
    full_metadata = {
        **metadata,
        "threshold": float(threshold),
        "artifact": model_path.name,
        "artifact_sha256": file_sha256(model_path),
        "artifact_bytes": model_path.stat().st_size,
    }
    metadata_path = model_path.with_name("metadata.json")
    metadata_path.write_text(
        json.dumps(full_metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return full_metadata


def load_model(model_path: str | Path, expected_sha256: str | None = None) -> dict:
    """Charge le bundle après vérification de son existence et de son empreinte.

    Si ``expected_sha256`` n'est pas fourni, l'empreinte est lue dans le fichier
    ``metadata.json`` situé à côté du modèle ; en son absence, le chargement est refusé.
    """
    model_path = Path(model_path)
    if not model_path.is_file():
        raise ModelLoadingError(f"Modèle introuvable : {model_path}")

    if expected_sha256 is None:
        metadata_path = model_path.with_name("metadata.json")
        if not metadata_path.is_file():
            raise ModelLoadingError("metadata.json absent : impossible de vérifier le modèle.")
        try:
            expected_sha256 = json.loads(metadata_path.read_text(encoding="utf-8"))[
                "artifact_sha256"
            ]
        except (json.JSONDecodeError, KeyError) as exc:
            raise ModelLoadingError(f"metadata.json invalide : {exc}") from exc

    actual = file_sha256(model_path)
    if actual != expected_sha256:
        raise ModelLoadingError(
            "Empreinte SHA-256 inattendue : le fichier du modèle a été modifié ou corrompu."
        )

    try:
        bundle = joblib.load(model_path)
    except Exception as exc:  # fichier tronqué, format inconnu, dépendance absente...
        raise ModelLoadingError(f"Chargement impossible : {exc}") from exc

    if not isinstance(bundle, dict) or not {"pipeline", "threshold"} <= set(bundle):
        raise ModelLoadingError("Format inattendu : bundle {pipeline, threshold} attendu.")
    if not hasattr(bundle["pipeline"], "predict_proba"):
        raise ModelLoadingError("L'objet chargé ne fournit pas predict_proba.")
    threshold = bundle["threshold"]
    if not isinstance(threshold, int | float) or not 0.0 <= float(threshold) <= 1.0:
        raise ModelLoadingError(f"Seuil invalide dans le bundle : {threshold!r}")
    bundle.setdefault("metadata", {})
    return bundle


def _to_frame(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    if isinstance(X, dict):
        return pd.DataFrame([X])
    if isinstance(X, list) and all(isinstance(row, dict) for row in X):
        return pd.DataFrame(X)
    raise TypeError("X doit être un DataFrame, un dictionnaire ou une liste de dictionnaires.")


def predict_proba(model, X) -> np.ndarray:
    """Probabilités de fraude après validation du schéma et nettoyage sans état."""
    pipeline = model["pipeline"] if isinstance(model, dict) else model
    features, _ = preprocess_data(_to_frame(X), require_target=False)
    proba = pipeline.predict_proba(features)[:, 1]
    if not np.isfinite(proba).all() or (proba < 0).any() or (proba > 1).any():
        raise RuntimeError("Contrat de sortie violé : probabilité non finie ou hors [0, 1].")
    return proba


def predict_fraud(model, X, threshold: float | None = None) -> pd.DataFrame:
    """Renvoie, pour chaque transaction, la probabilité de fraude et la décision.

    ``model`` est un bundle (le seuil enregistré est utilisé) ou un pipeline nu
    (``threshold`` est alors obligatoire). Une transaction est signalée lorsque
    ``probabilité >= seuil``.
    """
    if threshold is None:
        if not isinstance(model, dict):
            raise ValueError("Seuil obligatoire avec un pipeline sans métadonnées.")
        threshold = model["threshold"]
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("Le seuil doit être compris entre 0 et 1.")
    frame = _to_frame(X)
    proba = predict_proba(model, frame)
    return pd.DataFrame(
        {"fraud_probability": proba, "is_fraud": (proba >= threshold).astype(int)},
        index=frame.index,
    )
