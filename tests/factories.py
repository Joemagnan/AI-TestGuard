"""Données synthétiques respectant le contrat, pour les tests unitaires."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import FEATURES, PCA_FEATURES, TARGET


def make_transactions(n: int = 200, n_frauds: int = 20, seed: int = 0) -> pd.DataFrame:
    """Jeu synthétique (les fraudes sont décalées sur V14 pour être apprenables)."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.normal(size=(n, len(PCA_FEATURES))), columns=PCA_FEATURES)
    frame.insert(0, "Time", np.sort(rng.uniform(0, 172_000, size=n)))
    frame["Amount"] = np.round(rng.lognormal(3, 1.2, size=n), 2)
    target = np.zeros(n, dtype="int64")
    target[rng.choice(n, size=n_frauds, replace=False)] = 1
    frame.loc[target == 1, "V14"] -= 4.0
    frame[TARGET] = target
    return frame[[*FEATURES, TARGET]]


def transaction_payload(**overrides) -> dict:
    """Une transaction valide au format JSON de l'API."""
    payload = {"Time": 40_000.0, **{f"V{i}": 0.1 * (i % 5 - 2) for i in range(1, 29)}}
    payload["Amount"] = 49.9
    payload.update(overrides)
    return payload
