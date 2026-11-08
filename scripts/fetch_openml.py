"""Télécharge le jeu ULB depuis son miroir public OpenML (identifiant 1597) pour la CI.

Kaggle exige un compte : la CI utilise donc le miroir OpenML du même jeu. Le
contenu est vérifié par le test lent : l'empreinte du jeu dédoublonné doit être
identique à celle de la référence (calculée sur le fichier Kaggle).

    python scripts/fetch_openml.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.datasets import fetch_openml  # noqa: E402

from app.config import FEATURES, FULL_DATA_PATH, TARGET  # noqa: E402


def main() -> int:
    bunch = fetch_openml(data_id=1597, as_frame=True, parser="pandas")
    frame = bunch.frame
    frame[TARGET] = frame[TARGET].astype(str).str.strip("'\"").astype("int64")
    missing = [c for c in [*FEATURES, TARGET] if c not in frame.columns]
    if missing:
        raise SystemExit(f"Colonnes inattendues dans le miroir OpenML : {missing}")
    frame = frame[[*FEATURES, TARGET]]
    FULL_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(FULL_DATA_PATH, index=False)
    print(f"{FULL_DATA_PATH} : {len(frame)} lignes, {int(frame[TARGET].sum())} fraudes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
