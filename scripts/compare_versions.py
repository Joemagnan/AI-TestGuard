"""PARTIE VI - Comparaison de nouvelles versions à la référence figée.

Chaque version candidate est entraînée sur le même jeu d'entraînement, son seuil
est choisi sur le même jeu de validation fixe, puis elle est comparée à
reports/baseline.json. Le jeu de test n'est pas utilisé.

    python scripts/compare_versions.py
"""

from __future__ import annotations

import json

from common import load_context, md_table, write_json, write_text

from app.config import BASELINE_PATH, REGRESSION_TOLERANCES
from app.evaluation import compare_with_baseline
from app.pipeline import fit_and_calibrate, library_versions

CANDIDATES = [
    {
        "version": "1.1.0-a",
        "change": "Time encodé en heure de la journée (sin, cos) au lieu des secondes brutes",
        "model_name": "decision_tree",
        "params": {"max_depth": 8, "min_samples_leaf": 20, "class_weight": None},
        "time_encoding": "cyclic",
    },
    {
        "version": "1.1.0-b",
        "change": "Régression logistique (C = 0,1), deuxième modèle de la comparaison",
        "model_name": "logistic_regression",
        "params": {"C": 0.1, "class_weight": None},
        "time_encoding": "raw",
    },
    {
        "version": "1.1.0-c",
        "change": "Arbre moins profond (profondeur 3) pour simplifier le modèle",
        "model_name": "decision_tree",
        "params": {"max_depth": 3, "min_samples_leaf": 20, "class_weight": None},
        "time_encoding": "raw",
    },
    {
        "version": "1.1.0-d",
        "change": "Règle de seuil modifiée : coût FN = coût FP (au lieu de 25 pour 1)",
        "model_name": "decision_tree",
        "params": {"max_depth": 8, "min_samples_leaf": 20, "class_weight": None},
        "time_encoding": "raw",
        "cost_fn": 1.0,
    },
]


def main() -> int:
    ctx = load_context()
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    rows = []
    for cand in CANDIDATES:
        result = fit_and_calibrate(
            ctx["splits"],
            cand["model_name"],
            cand["params"],
            cand["time_encoding"],
            cost_fn=cand.get("cost_fn", baseline["threshold_rule"]["cost_fn"]),
        )
        candidate = {
            "validation_fingerprint": ctx["splits"]["validation_fingerprint"],
            "validation_metrics": result["validation_metrics"],
            "libraries": library_versions(),
        }
        decision = compare_with_baseline(candidate, baseline, REGRESSION_TOLERANCES)
        rows.append(
            {
                **cand,
                "threshold": result["selection"]["threshold"],
                "validation_metrics": {
                    k: result["validation_metrics"][k]
                    for k in (
                        "recall",
                        "precision",
                        "fpr",
                        "pr_auc",
                        "tp",
                        "fp",
                        "fn",
                        "expected_cost",
                    )
                },
                "comparison": decision,
            }
        )
        print(f"{cand['version']} : {decision['decision']}")
    report = {
        "baseline": {
            "version": baseline["version"],
            "code_version": baseline["code_version"],
            "validation_metrics": {
                k: baseline["validation_metrics"][k]
                for k in ("recall", "precision", "fpr", "pr_auc", "tp", "fp", "fn", "expected_cost")
            },
            "threshold": baseline["threshold"],
        },
        "tolerances": REGRESSION_TOLERANCES,
        "candidates": rows,
    }
    write_json("regression_comparison.json", report)
    write_text("NON_REGRESSION.md", render(report))
    return 0


def render(report: dict) -> str:
    base = report["baseline"]["validation_metrics"]
    lines = [
        "# Rapport de non-régression",
        "",
        f"Référence figée : version {report['baseline']['version']} (commit "
        f"{report['baseline']['code_version']}), seuil {report['baseline']['threshold']:.4f}. "
        "Toutes les versions sont comparées sur le **même jeu de validation fixe** "
        "(empreinte vérifiée) ; le jeu de test n'est pas utilisé.",
        "",
        "Règle : dégradation > tolérance sur une métrique -> **rejeter** ; > moitié de la "
        "tolérance -> **investiguer** ; sinon -> **accepter**. Tolérances : "
        + ", ".join(f"{k} {v}" for k, v in report["tolerances"].items())
        + ".",
        "",
        md_table(
            ["Version", "Changement", "Seuil", "Rappel", "Précision", "FPR", "PR-AUC", "Décision"],
            [
                [
                    "1.0.0 (référence)",
                    "-",
                    report["baseline"]["threshold"],
                    base["recall"],
                    base["precision"],
                    base["fpr"],
                    base["pr_auc"],
                    "-",
                ]
            ]
            + [
                [
                    c["version"],
                    c["change"],
                    c["threshold"],
                    c["validation_metrics"]["recall"],
                    c["validation_metrics"]["precision"],
                    c["validation_metrics"]["fpr"],
                    c["validation_metrics"]["pr_auc"],
                    c["comparison"]["decision"].upper(),
                ]
                for c in report["candidates"]
            ],
        ),
        "",
        "## Motifs",
        "",
    ]
    for c in report["candidates"]:
        reasons = c["comparison"]["reasons"] or [
            "aucune dégradation au-delà de la moitié des tolérances"
        ]
        lines.append(
            f"- **{c['version']}** ({c['comparison']['decision']}) : " + " ; ".join(reasons)
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
