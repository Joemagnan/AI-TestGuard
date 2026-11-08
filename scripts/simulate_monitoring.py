"""PARTIE X - Surveillance simulée après déploiement.

Le jeu de test est rejoué en 6 lots de production de composition comparable
(tirage aléatoire sans remise). Trois incidents sont
injectés pour vérifier que les contrôles disponibles SANS labels les détectent, et
pour montrer ce qui ne peut être vu qu'à l'arrivée des labels (dérive de concept) :

* F4 - dérive des données : montants multipliés par 3 (changement de devise amont) ;
* F5 - panne amont : 5 % des transactions sans les variables critiques ;
* F6 - dérive de concept : les fraudeurs adaptent leur comportement (leurs valeurs
  sur les variables critiques ressemblent à celles des transactions normales).

    python scripts/simulate_monitoring.py
"""

from __future__ import annotations

import numpy as np
from common import load_context, md_table, write_json, write_text

from app.evaluation import evaluate_model
from app.monitoring import batch_quality_report, critical_features, drift_report, flag_rate_status
from app.prediction import predict_proba


def main() -> int:
    ctx = load_context()
    bundle, splits = ctx["bundle"], ctx["splits"]
    meta = bundle["metadata"]
    profile, threshold = meta["reference_profile"], bundle["threshold"]
    critical = critical_features(bundle["pipeline"])
    expected_flag = meta["validation_flag_rate"]

    frame = splits["test"]["frame"]
    rng = np.random.default_rng(7)
    windows = np.array_split(rng.permutation(len(frame)), 6)

    # Constat préalable : en tranches chronologiques de 8 h, le cycle jour/nuit fait
    # dériver Time et certaines composantes même sans incident.
    chrono = frame.sort_values("Time", kind="mergesort")
    chronological = []
    for k, idx in enumerate(np.array_split(np.arange(len(chrono)), 6), start=1):
        part = chrono.iloc[idx].drop(columns="Class")
        report = drift_report(part, profile)
        no_time = {c: f for c, f in report["features"].items() if c != "Time"}
        top = max(no_time, key=lambda c: no_time[c]["psi"])
        chronological.append(
            {
                "window": k,
                "hours": f"{part['Time'].min() / 3600:.1f} h - {part['Time'].max() / 3600:.1f} h",
                "psi_time": report["features"]["Time"]["psi"],
                "top_other_feature": top,
                "psi_top_other": no_time[top]["psi"],
                "status": report["status"],
            }
        )
    normal_medians = splits["train"]["frame"].query("Class == 0")[critical].median()

    rows = []
    for k, idx in enumerate(windows, start=1):
        part = frame.iloc[idx]
        X = part.drop(columns="Class").copy()
        y = part["Class"].to_numpy()
        incident = "aucun"
        if k == 4:
            X["Amount"] *= 3
            incident = "dérive des données : montants x 3"
        elif k == 5:
            mask = rng.random(len(X)) < 0.05
            X.loc[mask, critical] = np.nan
            incident = f"panne amont : {', '.join(critical)} absents (5 % des lignes)"
        elif k == 6:
            frauds = y == 1
            for col in critical:
                X.loc[frauds, col] = normal_medians[col] + rng.normal(0, 0.1, frauds.sum())
            incident = "dérive de concept : fraudes déguisées"
        proba = predict_proba(bundle, X)
        flags = proba >= threshold
        drift = drift_report(X, profile)
        quality = batch_quality_report(X, profile, critical)
        flag = flag_rate_status(int(flags.sum()), len(X), expected_flag)
        labelled = evaluate_model(y, flags.astype(int), proba)
        alerts = []
        if drift["status"] != "stable":
            alerts.append(f"dérive {drift['status']} ({', '.join(drift['top'][:2])})")
        if quality["missing_alert"]:
            alerts.append(f"données manquantes {quality['missing_rate']:.1%}")
        if quality["alert_counts"].get("variable_critique_manquante"):
            alerts.append(
                f"{quality['alert_counts']['variable_critique_manquante']} revues manuelles"
            )
        if flag["status"] == "alerte":
            alerts.append(f"taux signalé x {flag['ratio']:.2f}")
        rows.append(
            {
                "window": k,
                "incident": incident,
                "transactions": int(len(X)),
                "flag_rate": flag["rate"],
                "flag_ratio": flag["ratio"],
                "drift_status": drift["status"],
                "max_psi": max(f["psi"] for f in drift["features"].values()),
                "missing_rate": quality["missing_rate"],
                "alerts_without_labels": alerts or ["aucune"],
                "labels_frauds": int(y.sum()),
                "labels_recall": labelled["recall"],
                "labels_precision": labelled["precision"],
                "labels_fpr": labelled["fpr"],
            }
        )
    report = {
        "chronological_windows_without_incident": chronological,
        "expected_flag_rate": expected_flag,
        "critical_features": critical,
        "windows": rows,
    }
    write_json("monitoring_simulation.json", report)
    write_text("SURVEILLANCE_SIMULATION.md", render(report))
    for r in rows:
        print(
            r["window"],
            r["incident"],
            "|",
            "; ".join(r["alerts_without_labels"]),
            "| rappel",
            r["labels_recall"],
        )
    return 0


def render(report: dict) -> str:
    rows = report["windows"]
    return "\n".join(
        [
            "# Surveillance simulée (6 lots de production tirés du jeu de test)",
            "",
            f"Taux de transactions signalées attendu (validation) : {report['expected_flag_rate']:.3%}. "
            f"Variables critiques surveillées : {', '.join(report['critical_features'])}. "
            "Script : `scripts/simulate_monitoring.py`.",
            "",
            "## Constat préalable : cycle jour/nuit",
            "",
            "Découpé en tranches chronologiques de 8 h, le jeu de test SANS incident déclenche "
            "des alertes de dérive : Time change évidemment de distribution, et certaines "
            "composantes suivent le rythme jour/nuit. Comparer une tranche horaire au profil "
            "d'entraînement (48 h) produit donc de fausses alertes : la surveillance doit "
            "comparer des fenêtres de 24 h, ou une tranche à la même tranche horaire de "
            "référence, et exclure Time du calcul de dérive.",
            "",
            md_table(
                [
                    "Tranche",
                    "Heures",
                    "PSI Time",
                    "Autre variable la plus dérivante",
                    "PSI",
                    "Statut",
                ],
                [
                    [
                        c["window"],
                        c["hours"],
                        c["psi_time"],
                        c["top_other_feature"],
                        c["psi_top_other"],
                        c["status"],
                    ]
                    for c in report["chronological_windows_without_incident"]
                ],
            ),
            "",
            "## Indicateurs disponibles immédiatement (sans labels)",
            "",
            md_table(
                [
                    "Fenêtre",
                    "Incident injecté",
                    "Taux signalé",
                    "Ratio / attendu",
                    "Dérive (PSI max)",
                    "Lignes incomplètes",
                    "Alertes émises",
                ],
                [
                    [
                        r["window"],
                        r["incident"],
                        f"{r['flag_rate']:.3%}",
                        r["flag_ratio"],
                        f"{r['drift_status']} ({r['max_psi']:.2f})",
                        f"{r['missing_rate']:.1%}",
                        "; ".join(r["alerts_without_labels"]),
                    ]
                    for r in rows
                ],
            ),
            "",
            "## Indicateurs disponibles à l'arrivée des labels (contestations, enquêtes)",
            "",
            md_table(
                ["Fenêtre", "Fraudes réelles", "Rappel", "Précision", "FPR"],
                [
                    [
                        r["window"],
                        r["labels_frauds"],
                        r["labels_recall"],
                        r["labels_precision"],
                        r["labels_fpr"],
                    ]
                    for r in rows
                ],
            ),
            "",
            "Lecture : la dérive des données (F4) et la panne amont (F5) sont visibles dès "
            "l'arrivée des transactions. La dérive de concept (F6) ne modifie presque pas la "
            "distribution des entrées (les fraudes représentent 0,17 % du flux) : seule une "
            "baisse du taux de transactions signalées peut l'annoncer, et elle n'est confirmée "
            "qu'à l'arrivée des labels, par la chute du rappel.",
            "",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
