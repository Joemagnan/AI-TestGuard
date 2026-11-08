"""Télécharge creditcard.csv pour le job de ré-entraînement complet de la CI.

Kaggle exige un compte : la CI utilise la copie publique du même fichier hébergée
pour le tutoriel TensorFlow « Classification on imbalanced data ». Le fichier
n'est accepté que si son empreinte SHA-256 est identique à celle du fichier
Kaggle utilisé pour l'entraînement de référence (data/README.md).

Anomalie A15 : le miroir OpenML (identifiant 1597), utilisé au départ, n'est pas
le même jeu (colonne Time absente) ; le contrôle du schéma l'a rejeté.

    python scripts/fetch_dataset.py
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import FULL_DATA_PATH  # noqa: E402

URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
EXPECTED_SHA256 = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"


def main() -> int:
    FULL_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    partial = FULL_DATA_PATH.with_suffix(".part")
    digest = hashlib.sha256()
    with urllib.request.urlopen(URL, timeout=120) as response, open(partial, "wb") as out:
        for chunk in iter(lambda: response.read(1 << 20), b""):
            digest.update(chunk)
            out.write(chunk)
    if digest.hexdigest() != EXPECTED_SHA256:
        partial.unlink(missing_ok=True)
        print(f"Empreinte inattendue : {digest.hexdigest()} (attendu {EXPECTED_SHA256})")
        return 1
    partial.replace(FULL_DATA_PATH)
    print(f"{FULL_DATA_PATH} : empreinte SHA-256 conforme au fichier Kaggle de référence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
