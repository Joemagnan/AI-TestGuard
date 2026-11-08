"""Outils partagés par les scripts d'expérimentation (jeu Kaggle complet requis)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import FULL_DATA_PATH, MODEL_PATH, REPORTS_DIR  # noqa: E402
from app.pipeline import load_dataset, make_splits, prepare_dataset  # noqa: E402
from app.prediction import load_model  # noqa: E402


def load_context() -> dict:
    """Modèle livré + découpage identique à celui de l'entraînement."""
    bundle = load_model(MODEL_PATH)
    data, info = prepare_dataset(load_dataset(FULL_DATA_PATH))
    if info["fingerprint"] != bundle["metadata"]["data"]["fingerprint"]:
        raise SystemExit("Le jeu de données ne correspond pas à celui du modèle livré.")
    splits = make_splits(data, bundle["metadata"]["seed"])
    if splits["validation_fingerprint"] != bundle["metadata"]["validation_fingerprint"]:
        raise SystemExit("Le découpage ne correspond pas à celui du modèle livré.")
    return {"bundle": bundle, "data": data, "splits": splits}


def write_json(name: str, content) -> Path:
    path = REPORTS_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_text(name: str, content: str) -> Path:
    path = REPORTS_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "n/d"
    if isinstance(value, bool):
        return "oui" if value else "non"
    if isinstance(value, float):
        if value != 0 and abs(value) < 10 ** (-digits):
            return f"{value:.1e}"
        return f"{value:.{digits}f}"
    return str(value)


def md_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(fmt(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)
