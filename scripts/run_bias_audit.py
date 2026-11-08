"""PARTIE VII - Écarts de performance par segment opérationnel (jeu de test).

Le jeu ne contient AUCUN attribut démographique (âge, sexe, localisation) : aucune
conclusion d'équité envers des groupes de personnes n'est possible, et aucun attribut
n'est déduit des composantes anonymisées. On compare seulement des segments
observables et légitimes (montant, heure) pour repérer des disparités de traitement
opérationnelles, avec leurs intervalles de confiance.

    python scripts/run_bias_audit.py
"""

from __future__ import annotations

from common import load_context, md_table, write_json, write_text

from app.evaluation import bootstrap_ci, evaluate_model
from app.prediction import predict_proba


def segments(frame):
    hours = (frame["Time"] % 86_400) / 3600
    amount = frame["Amount"]
    return {
        "Montant < 10 EUR": amount < 10,
        "Montant 10-100 EUR": (amount >= 10) & (amount < 100),
        "Montant >= 100 EUR": amount >= 100,
        "Nuit (0 h - 6 h)": hours < 6,
        "Jour (6 h - 24 h)": hours >= 6,
    }


def main() -> int:
    ctx = load_context()
    bundle = ctx["bundle"]
    frame = ctx["splits"]["test"]["frame"]
    X, y = ctx["splits"]["test"]["X"], ctx["splits"]["test"]["y"]
    proba = predict_proba(bundle, X)
    pred = (proba >= bundle["threshold"]).astype(int)
    rows = []
    for name, mask in segments(frame).items():
        m = mask.to_numpy()
        metrics = evaluate_model(y[m], pred[m], proba[m])
        ci = bootstrap_ci(y[m], pred[m], None, metrics=("recall", "fpr", "precision"), n_boot=1000)
        rows.append(
            {
                "segment": name,
                "transactions": int(m.sum()),
                "frauds": int(y[m].sum()),
                "recall": metrics["recall"],
                "recall_ci": ci["recall"],
                "fpr": metrics["fpr"],
                "fpr_ci": ci["fpr"],
                "precision": metrics["precision"],
                "precision_ci": ci["precision"],
            }
        )
    write_json("bias_audit.json", rows)
    table = md_table(
        [
            "Segment",
            "Transactions",
            "Fraudes",
            "Rappel [IC 95 %]",
            "FPR [IC 95 %]",
            "Précision [IC 95 %]",
        ],
        [
            [
                r["segment"],
                r["transactions"],
                r["frauds"],
                interval(r["recall"], r["recall_ci"]),
                interval(r["fpr"], r["fpr_ci"], pct=True),
                interval(r["precision"], r["precision_ci"]),
            ]
            for r in rows
        ],
    )
    write_text("SEGMENTS.md", "# Performance par segment opérationnel (test)\n\n" + table + "\n")
    print(table)
    return 0


def interval(value, ci, pct=False) -> str:
    if value is None or ci is None:
        return "n/d"
    if pct:
        return f"{value:.3%} [{ci[0]:.3%} ; {ci[1]:.3%}]"
    return f"{value:.2f} [{ci[0]:.2f} ; {ci[1]:.2f}]"


if __name__ == "__main__":
    raise SystemExit(main())
