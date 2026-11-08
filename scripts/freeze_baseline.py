"""Fige la référence (baseline) de non-régression à partir d'un modèle entraîné.

La référence n'est JAMAIS recalculée automatiquement : ce script doit être lancé
explicitement, et refuse d'écraser une référence existante sans ``--force``.

    python scripts/freeze_baseline.py                 # modèle complet -> reports/baseline.json
    python scripts/freeze_baseline.py --sample        # échantillon CI -> tests/reference/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import (  # noqa: E402
    BASELINE_PATH,
    METADATA_PATH,
    SAMPLE_BASELINE_PATH,
    SAMPLE_DATA_PATH,
    SEED,
)
from app.pipeline import (  # noqa: E402
    fit_and_calibrate,
    git_commit,
    library_versions,
    load_dataset,
    make_splits,
    prepare_dataset,
)

BASELINE_KEYS = (
    "version",
    "code_version",
    "model_name",
    "params",
    "time_encoding",
    "threshold",
    "threshold_rule",
    "seed",
    "data",
    "split_sizes",
    "split_frauds",
    "validation_fingerprint",
    "validation_metrics",
    "libraries",
)


def baseline_from_metadata(metadata: dict) -> dict:
    return {key: metadata[key] for key in BASELINE_KEYS}


def sample_baseline(metadata: dict) -> dict:
    """Référence CI : configuration du modèle de production rejouée sur l'échantillon."""
    data, info = prepare_dataset(load_dataset(SAMPLE_DATA_PATH))
    splits = make_splits(data, SEED)
    result = fit_and_calibrate(
        splits, metadata["model_name"], metadata["params"], metadata["time_encoding"], SEED
    )
    return {
        "version": metadata["version"],
        "code_version": git_commit(),
        "model_name": metadata["model_name"],
        "params": metadata["params"],
        "time_encoding": metadata["time_encoding"],
        "threshold": result["selection"]["threshold"],
        "threshold_rule": result["selection"]["rule"],
        "seed": SEED,
        "data": info,
        "validation_fingerprint": splits["validation_fingerprint"],
        "validation_metrics": result["validation_metrics"],
        "libraries": library_versions(),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", action="store_true", help="référence de la CI (échantillon)")
    parser.add_argument("--force", action="store_true", help="écraser une référence existante")
    args = parser.parse_args(argv)

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    target = SAMPLE_BASELINE_PATH if args.sample else BASELINE_PATH
    if target.exists() and not args.force:
        print(f"Référence existante conservée : {target} (utiliser --force pour la remplacer).")
        return 1
    baseline = sample_baseline(metadata) if args.sample else baseline_from_metadata(metadata)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(baseline, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Référence figée : {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
