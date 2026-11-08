"""PARTIE V - Campagne de robustesse et de sensibilité sur le jeu de test complet.

Produit reports/robustness.json, reports/ROBUSTESSE.md et
reports/figures/sensitivity.png. Aucune décision de réglage n'est prise ici.

    python scripts/run_robustness.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from common import REPORTS_DIR, load_context, md_table, write_json, write_text
from sklearn.model_selection import StratifiedKFold

from app.config import PCA_FEATURES, SEED, TARGET
from app.evaluation import evaluate_model
from app.monitoring import batch_quality_report, drift_report, transaction_alerts
from app.pipeline import fit_and_calibrate
from app.plots import plot_sensitivity
from app.prediction import predict_fraud, predict_proba
from app.preprocessing import DataValidationError, preprocess_data
from app.training import train_model

KEYS = ("recall", "precision", "fpr")


def metrics_for(bundle, X, y, threshold=None):
    proba = predict_proba(bundle, X)
    thr = bundle["threshold"] if threshold is None else threshold
    return proba, evaluate_model(y, (proba >= thr).astype(int), proba)


def delta(base: dict, other: dict) -> dict:
    return {
        f"d_{k}": (other[k] - base[k]) if None not in (other[k], base[k]) else None for k in KEYS
    }


def main() -> int:
    ctx = load_context()
    bundle, splits = ctx["bundle"], ctx["splits"]
    threshold = bundle["threshold"]
    profile = bundle["metadata"]["reference_profile"]
    X, y = splits["test"]["X"], splits["test"]["y"]
    X_train = splits["train"]["X"]
    train_std = X_train[PCA_FEATURES].std()
    base_proba, base = metrics_for(bundle, X, y)
    rng = np.random.default_rng(SEED)
    results = {"baseline_test": base, "threshold": threshold}

    model = bundle["pipeline"].named_steps["model"]
    names = bundle["pipeline"].named_steps["preprocess"].get_feature_names_out()
    importances = sorted(zip(names, model.feature_importances_, strict=True), key=lambda t: -t[1])
    top3 = [name for name, _ in importances[:3]]
    results["feature_importances"] = [[n, float(v)] for n, v in importances[:10]]

    # --- S1 : valeurs manquantes (traiter + alerter) -----------------------------------------
    affected = rng.random(len(X)) < 0.10
    missing_rows = []
    for label, cols in (
        ("V1 manquante", ["V1"]),
        (f"3 variables les plus importantes ({', '.join(top3)})", top3),
        ("V1 à V28 manquantes", PCA_FEATURES),
        ("Amount manquant", ["Amount"]),
    ):
        Xm = X.copy()
        Xm.loc[affected, cols] = np.nan
        _, m = metrics_for(bundle, Xm, y)
        fraud_aff = affected & (y == 1)
        proba_aff = predict_proba(bundle, Xm[fraud_aff])
        recall_aff = float((proba_aff >= threshold).mean()) if fraud_aff.any() else None
        base_aff = float((base_proba[fraud_aff] >= threshold).mean()) if fraud_aff.any() else None
        quality = batch_quality_report(Xm, profile)
        missing_rows.append(
            {
                "scenario": label,
                "rows_affected": int(affected.sum()),
                "frauds_affected": int(fraud_aff.sum()),
                "recall_affected_before": base_aff,
                "recall_affected_after": recall_aff,
                **delta(base, m),
                "alert": quality["missing_alert"],
                "decision": "traiter (imputation) + alerte",
            }
        )
    results["missing_values"] = missing_rows

    # --- S2 : infinis (traiter) et non numériques (rejeter) -------------------------------------
    Xi = X.copy()
    Xi.loc[affected, "V4"] = np.inf
    Xn = X.copy()
    Xn.loc[affected, "V4"] = np.nan
    same = bool(np.array_equal(predict_proba(bundle, Xi), predict_proba(bundle, Xn)))
    Xs = X.astype({"V4": object})
    Xs.loc[Xs.index[0], "V4"] = "erreur"
    try:
        predict_fraud(bundle, Xs)
        rejected = False
    except DataValidationError:
        rejected = True
    results["infinite_and_non_numeric"] = {
        "inf_treated_as_missing": same,
        "non_numeric_batch_rejected": rejected,
        "decision": "infini : traiter comme manquant + alerte ; non numérique : rejeter",
    }

    # --- S3 : montants extrêmes (traiter + alerter) ---------------------------------------------
    amount_rows = []
    for label, values in (
        ("Amount x 10", X["Amount"] * 10),
        ("Amount x 100", X["Amount"] * 100),
        ("Amount = 1 000 000", pd.Series(1e6, index=X.index)),
        ("Amount = 1e12", pd.Series(1e12, index=X.index)),
    ):
        Xa = X.assign(Amount=values)
        proba, m = metrics_for(bundle, Xa, y)
        alerts = transaction_alerts(Xa, profile)
        amount_rows.append(
            {
                "scenario": label,
                "valid_probabilities": bool(
                    np.isfinite(proba).all() and proba.min() >= 0 and proba.max() <= 1
                ),
                "flip_rate": float(((proba >= threshold) != (base_proba >= threshold)).mean()),
                "alert_rate": float(np.mean(["montant_hors_domaine" in a for a in alerts])),
                **delta(base, m),
                "decision": "traiter + alerte montant_hors_domaine",
            }
        )
    results["extreme_amounts"] = amount_rows

    # --- S4 : perturbations limitées (sensibilité mesurée) ---------------------------------------
    near = np.abs(base_proba - threshold) < 0.1
    sensitivity = []
    for level in (0.01, 0.05, 0.10, 0.25, 0.50):
        deltas, flips, flips_fraud, flips_near, metric_rows = [], [], [], [], []
        for seed in range(5):
            noise_rng = np.random.default_rng(1000 + seed)
            Xp = X.copy()
            noise = (
                noise_rng.normal(size=(len(X), len(PCA_FEATURES))) * train_std.to_numpy() * level
            )
            Xp[PCA_FEATURES] = Xp[PCA_FEATURES].to_numpy() + noise
            proba, m = metrics_for(bundle, Xp, y)
            changed = (proba >= threshold) != (base_proba >= threshold)
            deltas.append(np.abs(proba - base_proba).mean())
            flips.append(changed.mean())
            flips_fraud.append(changed[y == 1].mean())
            flips_near.append(changed[near].mean() if near.any() else 0.0)
            metric_rows.append(m)
        sensitivity.append(
            {
                "level": level,
                "mean_abs_delta": float(np.mean(deltas)),
                "flip_rate": float(np.mean(flips)),
                "flip_rate_frauds": float(np.mean(flips_fraud)),
                "flip_rate_near_threshold": float(np.mean(flips_near)),
                "recall": float(np.mean([m["recall"] for m in metric_rows])),
                "precision": float(np.mean([m["precision"] for m in metric_rows])),
                "fpr": float(np.mean([m["fpr"] for m in metric_rows])),
            }
        )
    results["pca_noise"] = sensitivity
    plot_sensitivity(sensitivity, REPORTS_DIR / "figures" / "sensitivity.png")

    other_rows = []
    for label, Xo in (
        (
            "Amount +/- 1 % (multiplicatif)",
            X.assign(Amount=X["Amount"] * rng.uniform(0.99, 1.01, len(X))),
        ),
        (
            "Amount +/- 5 % (multiplicatif)",
            X.assign(Amount=X["Amount"] * rng.uniform(0.95, 1.05, len(X))),
        ),
        (
            "Time +/- 60 s",
            X.assign(Time=np.clip(X["Time"] + rng.uniform(-60, 60, len(X)), 0, None)),
        ),
        (
            "Time +/- 1 h",
            X.assign(Time=np.clip(X["Time"] + rng.uniform(-3600, 3600, len(X)), 0, None)),
        ),
    ):
        proba, m = metrics_for(bundle, Xo, y)
        other_rows.append(
            {
                "scenario": label,
                "mean_abs_delta": float(np.abs(proba - base_proba).mean()),
                "flip_rate": float(((proba >= threshold) != (base_proba >= threshold)).mean()),
                **delta(base, m),
            }
        )
    results["other_perturbations"] = other_rows

    # --- S5 : distribution différente (alerte de dérive) -----------------------------------------
    shift_rows = []
    top_v = [n for n in top3 if n.startswith("V")]
    for label, Xd in (
        ("Montants x 3 (inflation, nouveau commerce)", X.assign(Amount=X["Amount"] * 3)),
        (
            f"Décalage +0,5 écart-type sur {', '.join(top_v)}",
            X.assign(**{c: X[c] + 0.5 * train_std[c] for c in top_v}),
        ),
        ("Échelle x 1,5 sur V1 à V28", X.assign(**{c: X[c] * 1.5 for c in PCA_FEATURES})),
    ):
        proba, m = metrics_for(bundle, Xd, y)
        report = drift_report(Xd, profile)
        shift_rows.append(
            {
                "scenario": label,
                "drift_status": report["status"],
                "top_features": report["top"][:3],
                "max_psi": max(f["psi"] for f in report["features"].values()),
                "flag_rate": float((proba >= threshold).mean()),
                **delta(base, m),
            }
        )
    shift_rows.insert(
        0,
        {
            "scenario": "Jeu de test sans modification (contrôle)",
            "drift_status": drift_report(X, profile)["status"],
            "top_features": drift_report(X, profile)["top"][:3],
            "max_psi": max(f["psi"] for f in drift_report(X, profile)["features"].values()),
            "flag_rate": float((base_proba >= threshold).mean()),
            "d_recall": 0.0,
            "d_precision": 0.0,
            "d_fpr": 0.0,
        },
    )
    results["distribution_shift"] = shift_rows

    # Prévalence x 5 : la précision dépend du taux de fraude, pas le rappel
    frauds = X[y == 1]
    Xp = pd.concat([X, frauds, frauds, frauds, frauds])
    yp = np.concatenate([y, np.ones(4 * len(frauds), dtype=int)])
    _, mp = metrics_for(bundle, Xp, yp)
    results["prevalence_shift"] = {"fraud_rate": float(yp.mean()), **{k: mp[k] for k in KEYS}}

    # Validation temporelle : entraînement sur le passé, test sur le futur
    data = ctx["data"]
    t_train = data[data["Time"] < 100_000]
    t_val = data[(data["Time"] >= 100_000) & (data["Time"] < 130_000)]
    t_test = data[data["Time"] >= 130_000]
    temporal = {}
    for name, part in (("train", t_train), ("val", t_val), ("test", t_test)):
        Xt, yt = preprocess_data(part)
        temporal[name] = {"X": Xt, "y": yt.to_numpy()}
    temporal["validation_fingerprint"] = "temporel"
    meta = bundle["metadata"]
    fitted = fit_and_calibrate(temporal, meta["model_name"], meta["params"], meta["time_encoding"])
    tp = fitted["pipeline"].predict_proba(temporal["test"]["X"])[:, 1]
    tm = evaluate_model(
        temporal["test"]["y"], (tp >= fitted["selection"]["threshold"]).astype(int), tp
    )
    results["temporal_validation"] = {
        "periods_seconds": {
            "train": "[0, 100000[",
            "val": "[100000, 130000[",
            "test": "[130000, 172792]",
        },
        "sizes": {k: int(len(temporal[k]["y"])) for k in ("train", "val", "test")},
        "frauds": {k: int(temporal[k]["y"].sum()) for k in ("train", "val", "test")},
        "threshold": fitted["selection"]["threshold"],
        "test": {k: tm[k] for k in ("recall", "precision", "fpr", "pr_auc", "tp", "fp", "fn")},
    }

    # Variabilité d'échantillonnage : seuil livré appliqué à 5 plis (entraînement + validation)
    pool = pd.concat([splits["train"]["frame"], splits["val"]["frame"]])
    folds = []
    for fold, (tr, te) in enumerate(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(pool, pool[TARGET])
    ):
        Xtr, ytr = preprocess_data(pool.iloc[tr])
        Xte, yte = preprocess_data(pool.iloc[te])
        pipe = train_model(Xtr, ytr, meta["model_name"], meta["params"], meta["time_encoding"])
        proba = pipe.predict_proba(Xte)[:, 1]
        m = evaluate_model(yte, (proba >= threshold).astype(int), proba)
        folds.append(
            {"fold": fold + 1, "frauds": int(yte.sum()), **{k: m[k] for k in (*KEYS, "pr_auc")}}
        )
    results["cross_validation_fixed_threshold"] = folds

    # --- S6 : schéma invalide (rejeter) ----------------------------------------------------------
    schema_cases = {
        "colonne Amount absente": X.drop(columns="Amount"),
        "colonne V1 renommée v1": X.rename(columns={"V1": "v1"}),
        "colonne V2 dupliquée": pd.concat([X, X[["V2"]]], axis=1),
        "V5 de type texte": X.astype({"V5": str}),
        "lot vide": X.iloc[0:0],
        "montant négatif": X.assign(Amount=-1.0),
        "colonne supplémentaire (card_id)": X.assign(card_id="x"),
    }
    schema_rows = []
    for label, frame in schema_cases.items():
        try:
            predict_fraud(bundle, frame)
            outcome = "accepté (colonne ignorée et journalisée)"
        except DataValidationError as exc:
            outcome = f"rejeté : {exc}"
        schema_rows.append({"scenario": label, "outcome": outcome})
    results["invalid_schema"] = schema_rows
    write_json("robustness.json", results)
    write_text("ROBUSTESSE.md", render(results, base))
    print("reports/robustness.json et reports/ROBUSTESSE.md écrits")
    return 0


def render(r: dict, base: dict) -> str:
    out = [
        "# Rapport de robustesse et de sensibilité",
        "",
        "Modèle livré v1.0.0, jeu de test complet (56 746 transactions, 95 fraudes), "
        f"seuil {r['threshold']:.4f}. Référence : rappel {base['recall']:.3f}, précision "
        f"{base['precision']:.3f}, FPR {base['fpr']:.5f}. Script : `scripts/run_robustness.py`.",
        "",
        "## 1. Valeurs manquantes (10 % des lignes) - comportement : traiter + alerte",
        "",
        md_table(
            [
                "Scénario",
                "Fraudes touchées",
                "Rappel avant (touchées)",
                "Rappel après (touchées)",
                "Δ rappel global",
                "Δ FPR global",
                "Alerte",
            ],
            [
                [
                    x["scenario"],
                    x["frauds_affected"],
                    x["recall_affected_before"],
                    x["recall_affected_after"],
                    x["d_recall"],
                    x["d_fpr"],
                    "oui" if x["alert"] else "non",
                ]
                for x in r["missing_values"]
            ],
        ),
        "",
        "## 2. Valeurs infinies et non numériques",
        "",
        f"- Infini traité exactement comme une valeur manquante : {r['infinite_and_non_numeric']['inf_treated_as_missing']}",
        f"- Lot contenant une valeur texte rejeté : {r['infinite_and_non_numeric']['non_numeric_batch_rejected']}",
        "",
        "## 3. Montants extrêmes - comportement : traiter + alerte",
        "",
        md_table(
            [
                "Scénario",
                "Probabilités valides",
                "Décisions changées",
                "Taux d'alerte",
                "Δ rappel",
                "Δ FPR",
            ],
            [
                [
                    x["scenario"],
                    x["valid_probabilities"],
                    x["flip_rate"],
                    x["alert_rate"],
                    x["d_recall"],
                    x["d_fpr"],
                ]
                for x in r["extreme_amounts"]
            ],
        ),
        "",
        "## 4. Perturbations limitées - sensibilité mesurée (moyenne sur 5 tirages)",
        "",
        "Bruit gaussien sur V1 à V28 proportionnel à l'écart-type d'entraînement de chaque "
        "composante ; Amount et Time sont perturbés séparément (amplitudes propres).",
        "",
        md_table(
            [
                "Amplitude (x écart-type)",
                "|Δ proba| moyen",
                "Décisions changées",
                "Fraudes changées",
                "Changées près du seuil",
                "Rappel",
                "Précision",
                "FPR",
            ],
            [
                [
                    x["level"],
                    x["mean_abs_delta"],
                    x["flip_rate"],
                    x["flip_rate_frauds"],
                    x["flip_rate_near_threshold"],
                    x["recall"],
                    x["precision"],
                    x["fpr"],
                ]
                for x in r["pca_noise"]
            ],
        ),
        "",
        md_table(
            [
                "Perturbation",
                "|Δ proba| moyen",
                "Décisions changées",
                "Δ rappel",
                "Δ précision",
                "Δ FPR",
            ],
            [
                [
                    x["scenario"],
                    x["mean_abs_delta"],
                    x["flip_rate"],
                    x["d_recall"],
                    x["d_precision"],
                    x["d_fpr"],
                ]
                for x in r["other_perturbations"]
            ],
        ),
        "",
        "## 5. Distribution différente - comportement : alerte de dérive (PSI)",
        "",
        md_table(
            [
                "Scénario",
                "Statut dérive",
                "PSI max",
                "Variables en tête",
                "Taux signalé",
                "Δ rappel",
                "Δ FPR",
            ],
            [
                [
                    x["scenario"],
                    x["drift_status"],
                    x["max_psi"],
                    ", ".join(x["top_features"]),
                    x["flag_rate"],
                    x["d_recall"],
                    x["d_fpr"],
                ]
                for x in r["distribution_shift"]
            ],
        ),
        "",
        f"Prévalence multipliée par 5 (taux de fraude {r['prevalence_shift']['fraud_rate']:.3%}) : rappel "
        f"{r['prevalence_shift']['recall']:.3f}, précision {r['prevalence_shift']['precision']:.3f}, FPR "
        f"{r['prevalence_shift']['fpr']:.5f}.",
        "",
        "### Validation temporelle (passé -> futur)",
        "",
        f"Entraînement sur [0 ; 100 000[ s, seuil sur [100 000 ; 130 000[ s, test sur [130 000 ; 172 792] s "
        f"({r['temporal_validation']['frauds']['test']} fraudes) : rappel "
        f"{r['temporal_validation']['test']['recall']:.3f}, précision {r['temporal_validation']['test']['precision']:.3f}, "
        f"FPR {r['temporal_validation']['test']['fpr']:.5f}, PR-AUC {r['temporal_validation']['test']['pr_auc']:.3f}.",
        "",
        "### Variabilité d'échantillonnage (5 plis, seuil livré fixe, sans le jeu de test)",
        "",
        md_table(
            ["Pli", "Fraudes", "Rappel", "Précision", "FPR", "PR-AUC"],
            [
                [x["fold"], x["frauds"], x["recall"], x["precision"], x["fpr"], x["pr_auc"]]
                for x in r["cross_validation_fixed_threshold"]
            ],
        ),
        "",
        "## 6. Schéma invalide - comportement : rejeter",
        "",
        md_table(["Cas", "Résultat"], [[x["scenario"], x["outcome"]] for x in r["invalid_schema"]]),
        "",
        "## Variables les plus importantes du modèle livré",
        "",
        md_table(["Variable", "Importance"], r["feature_importances"]),
        "",
    ]
    return "\n".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
