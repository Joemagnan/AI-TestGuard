"""Construit l'échantillon versionné utilisé par la CI (data/sample/creditcard_sample.csv).

L'échantillon est ENRICHI en fraudes (120 fraudes, 3 880 transactions normales,
soit 3 % de fraudes contre 0,17 % dans le jeu complet) pour que les tests rapides
disposent d'assez de cas positifs. Ses métriques ne sont donc pas représentatives :
il sert uniquement à vérifier le comportement du code et la non-régression.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import FULL_DATA_PATH, SAMPLE_DATA_PATH, SEED, TARGET  # noqa: E402
from app.pipeline import load_dataset, prepare_dataset  # noqa: E402

N_FRAUDS = 120
N_NORMAL = 3880


def main() -> int:
    data, _ = prepare_dataset(load_dataset(FULL_DATA_PATH))
    frauds = data[data[TARGET] == 1].sample(n=N_FRAUDS, random_state=SEED)
    normal = data[data[TARGET] == 0].sample(n=N_NORMAL, random_state=SEED)
    sample = pd.concat([frauds, normal]).sort_values("Time", kind="mergesort")
    SAMPLE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(SAMPLE_DATA_PATH, index=False, lineterminator="\n")
    print(f"{SAMPLE_DATA_PATH} : {len(sample)} lignes, {int(sample[TARGET].sum())} fraudes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
