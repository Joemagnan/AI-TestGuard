"""PARTIE III - Rapport d'évaluation ML (reports/EVALUATION_ML.md) à partir de
reports/metrics.json produit par le pipeline.

    python scripts/write_evaluation_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import REPORTS_DIR, md_table, write_text  # noqa: E402

LABELS = {"logistic_regression": "Régression logistique", "decision_tree": "Arbre de décision"}
COLUMNS = ("recall", "precision", "f1", "fpr", "fnr", "pr_auc", "roc_auc", "accuracy")


def num(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def row(name: str, part: str, m: dict, threshold: float) -> list:
    return [name, part, threshold, m["tp"], m["fp"], m["fn"], m["tn"], *[m[c] for c in COLUMNS]]


def main() -> int:
    r = json.loads((REPORTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    sel = r["selected_model"]
    test = r["test_metrics_selected"]
    train = r["train_metrics_selected"]
    ci = r["test_confidence_intervals_95"]
    headers = [
        "Modèle",
        "Jeu",
        "Seuil",
        "VP",
        "FP",
        "FN",
        "VN",
        "Rappel",
        "Précision",
        "F1",
        "FPR",
        "FNR",
        "PR-AUC",
        "ROC-AUC",
        "Accuracy",
    ]
    rows = []
    for family, cand in r["candidates"].items():
        rows.append(row(LABELS[family], "validation", cand["validation"], cand["threshold"]))
    for family, cand in r["candidates"].items():
        rows.append(row(LABELS[family], "test", cand["test"], cand["threshold"]))
    grid = sorted(r["grid_search"], key=lambda g: (g["model"], -g["val_pr_auc"]))
    text = [
        "# Rapport d'évaluation du modèle",
        "",
        f"Données : {num(r['data']['rows_before'])} transactions, "
        f"{num(r['data']['duplicates_removed'])} doublons exacts supprimés "
        f"({r['data']['frauds_removed']} fraudes), soit {num(r['data']['rows'])} transactions dont "
        f"{r['data']['frauds']} fraudes ({r['data']['fraud_rate']:.3%}).",
        f"Découpage stratifié 60/20/20 : entraînement {r['split_sizes']['train']} "
        f"({r['split_frauds']['train']} fraudes), validation {r['split_sizes']['val']} "
        f"({r['split_frauds']['val']}), test {r['split_sizes']['test']} ({r['split_frauds']['test']}).",
        "",
        "## Protocole",
        "",
        "1. Hyperparamètres choisis sur la validation (PR-AUC), grille réduite par famille.",
        "2. Seuil choisi sur la validation : coût attendu minimal "
        f"({r['threshold_selection']['rule']['cost_fn']:g} x FN + {r['threshold_selection']['rule']['cost_fp']:g} x FP) "
        f"sous la contrainte FPR <= {r['threshold_selection']['rule']['max_fpr']:.1%}.",
        "3. Modèle retenu : coût attendu minimal sur la validation.",
        "4. Évaluation unique sur le test, avec des critères fixés avant (commit 6064d6f).",
        "",
        "## Comparaison des modèles",
        "",
        md_table(headers, rows),
        "",
        f"Modèle retenu : **{LABELS[sel]}** (seuil {r['threshold']:.4f}). Sur la validation, les "
        f"deux modèles sont presque à égalité de coût ({r['candidates']['decision_tree']['validation']['expected_cost']:.0f} "
        f"contre {r['candidates']['logistic_regression']['validation']['expected_cost']:.0f}) : le choix "
        "suit la règle fixée à l'avance mais n'est pas un écart significatif.",
        "",
        "## Évaluation finale (test) et incertitude",
        "",
        md_table(
            ["Métrique", "Valeur", "IC 95 % (bootstrap stratifié, 1 000 tirages)"],
            [
                [k, test[k], f"[{ci[k][0]:.3f} ; {ci[k][1]:.3f}]"]
                for k in ("recall", "precision", "fpr", "pr_auc")
            ],
        ),
        "",
        md_table(
            ["Critère d'acceptation", "Valeur test", "Exigence", "Résultat"],
            [
                [
                    c["metric"],
                    c["value"],
                    (f">= {c['min']}" if "min" in c else f"<= {c['max']}"),
                    "atteint" if c["passed"] else "NON ATTEINT",
                ]
                for c in r["acceptance"]["checks"]
            ],
        ),
        "",
        "## Pourquoi l'accuracy ne suffit pas",
        "",
        f"Un modèle qui déclare toutes les transactions normales obtient une accuracy de "
        f"{r['naive_all_normal_accuracy']:.2%} sur le test, sans détecter une seule fraude "
        f"(rappel 0). Le modèle retenu a une accuracy de {test['accuracy']:.2%} : l'écart réel "
        "se lit sur le rappel, la précision et la PR-AUC.",
        "",
        "## Surapprentissage (entraînement / validation / test, seuil retenu)",
        "",
        md_table(
            ["Jeu", "Rappel", "Précision", "FPR", "PR-AUC"],
            [
                [
                    "entraînement",
                    train["recall"],
                    train["precision"],
                    train["fpr"],
                    train["pr_auc"],
                ],
                [
                    "validation",
                    *[
                        r["candidates"][sel]["validation"][k]
                        for k in ("recall", "precision", "fpr", "pr_auc")
                    ],
                ],
                ["test", test["recall"], test["precision"], test["fpr"], test["pr_auc"]],
            ],
        ),
        "",
        "## Recherche d'hyperparamètres (validation)",
        "",
        md_table(
            ["Modèle", "Paramètres", "PR-AUC validation", "ROC-AUC validation"],
            [
                [LABELS[g["model"]], json.dumps(g["params"]), g["val_pr_auc"], g["val_roc_auc"]]
                for g in grid
            ],
        ),
        "",
    ]
    write_text("EVALUATION_ML.md", "\n".join(text))
    print("reports/EVALUATION_ML.md écrit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
