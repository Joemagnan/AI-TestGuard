"""Graphiques du rapport (matplotlib, rendu statique pour le rapport Word).

Palette catégorielle validée (contraste et daltonisme) : bleu, orange, vert d'eau,
toujours dans cet ordre. Rampe séquentielle bleue pour les matrices de confusion.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#9a9890"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "bleu", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_confusion_matrix(metrics: dict, title: str, path: str | Path) -> Path:
    """Matrice de confusion : couleur = taux par classe réelle, texte = effectif et taux."""
    counts = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    rates = counts / counts.sum(axis=1, keepdims=True)
    labels = [["VN", "FP"], ["FN", "VP"]]
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    ax.imshow(rates, cmap=SEQUENTIAL, vmin=0, vmax=1)
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            color = "white" if rates[i, j] > 0.55 else TEXT
            ax.text(
                j,
                i,
                f"{labels[i][j]}\n{counts[i, j]:,}".replace(",", " ") + f"\n{rates[i, j]:.2%}",
                ha="center",
                va="center",
                color=color,
                fontsize=10,
            )
    ax.set_xticks([0, 1], ["Prédit normal", "Prédit fraude"])
    ax.set_yticks([0, 1], ["Normal réel", "Fraude réelle"])
    ax.set_title(title)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return _save(fig, path)


def plot_roc_pr(curves: dict, path: str | Path, title_suffix: str = "") -> Path:
    """Courbes ROC et précision-rappel côte à côte.

    ``curves`` : ``{nom: {"y_true", "y_proba", "threshold"}}`` ; le point de
    fonctionnement (seuil retenu) est marqué sur chaque courbe.
    """
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(10, 4.2))
    prevalence = None
    for color, (name, data) in zip(SERIES, curves.items(), strict=False):
        y_true, proba, thr = data["y_true"], data["y_proba"], data["threshold"]
        prevalence = float(np.mean(y_true))
        fpr, tpr, _ = roc_curve(y_true, proba)
        precision, recall, _ = precision_recall_curve(y_true, proba)
        ax_roc.plot(fpr, tpr, color=color, lw=2, label=name)
        ax_pr.plot(recall, precision, color=color, lw=2, label=name)
        pred = proba >= thr
        tp = np.sum(pred & (y_true == 1))
        op_fpr = np.sum(pred & (y_true == 0)) / np.sum(y_true == 0)
        op_rec = tp / np.sum(y_true == 1)
        op_prec = tp / max(np.sum(pred), 1)
        for ax, x, y in ((ax_roc, op_fpr, op_rec), (ax_pr, op_rec, op_prec)):
            ax.scatter([x], [y], s=60, color=color, edgecolor="white", linewidth=2, zorder=5)
    ax_roc.plot([0, 1], [0, 1], color=NEUTRAL, lw=1, ls="--")
    ax_roc.set(
        xlabel="Taux de faux positifs (FPR)", ylabel="Rappel (TPR)", xlim=(0, 1), ylim=(0, 1.02)
    )
    ax_roc.set_title(f"Courbe ROC{title_suffix}")
    inset = ax_roc.inset_axes([0.42, 0.12, 0.54, 0.5])
    for color, (_name, data) in zip(SERIES, curves.items(), strict=False):
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_proba"])
        inset.plot(fpr, tpr, color=color, lw=1.5)
    inset.set(xlim=(0, 0.01), ylim=(0.5, 1.0))
    inset.set_title("zoom FPR <= 1 %", fontsize=8, fontweight="normal", color=TEXT_SECONDARY)
    inset.tick_params(labelsize=7)
    if prevalence is not None:
        ax_pr.axhline(prevalence, color=NEUTRAL, lw=1, ls="--")
        ax_pr.text(
            0.02,
            prevalence + 0.02,
            f"hasard (prévalence {prevalence:.2%})",
            fontsize=8,
            color=TEXT_SECONDARY,
        )
    ax_pr.set(xlabel="Rappel", ylabel="Précision", xlim=(0, 1), ylim=(0, 1.02))
    ax_pr.set_title(f"Courbe précision-rappel{title_suffix}")
    ax_pr.legend(loc="lower left")
    ax_roc.text(
        1.0,
        -0.2,
        "point = seuil retenu sur validation",
        transform=ax_roc.transAxes,
        ha="right",
        fontsize=8,
        color=TEXT_SECONDARY,
    )
    return _save(fig, path)


def plot_threshold_analysis(y_true, y_proba, selection: dict, path: str | Path) -> Path:
    """Précision / rappel puis coût attendu selon le seuil (jeu de validation)."""
    y_true = np.asarray(y_true)
    proba = np.asarray(y_proba)
    thresholds = np.unique(np.quantile(proba, np.linspace(0.90, 1.0, 400)))
    precision, recall, cost, fpr = [], [], [], []
    rule = selection["rule"]
    negatives = np.sum(y_true == 0)
    for t in thresholds:
        pred = proba >= t
        tp = np.sum(pred & (y_true == 1))
        fp = np.sum(pred & (y_true == 0))
        fn = np.sum(~pred & (y_true == 1))
        precision.append(tp / max(tp + fp, 1))
        recall.append(tp / max(tp + fn, 1))
        cost.append(rule["cost_fn"] * fn + rule["cost_fp"] * fp)
        fpr.append(fp / negatives)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(thresholds, recall, color=SERIES[0], lw=2, label="Rappel")
    ax1.plot(thresholds, precision, color=SERIES[1], lw=2, label="Précision")
    ax2.plot(thresholds, cost, color=SERIES[0], lw=2)
    infeasible = np.array(fpr) > rule["max_fpr"]
    if infeasible.any():
        limit = thresholds[infeasible].max()
        for ax in (ax1, ax2):
            ax.axvspan(thresholds.min(), limit, color=GRID, alpha=0.6, lw=0)
        ax2.text(
            limit,
            max(cost) * 0.95,
            f"  FPR > {rule['max_fpr']:.1%}\n  (interdit)",
            fontsize=8,
            color=TEXT_SECONDARY,
            va="top",
            ha="right",
        )
    for ax in (ax1, ax2):
        ax.axvline(selection["threshold"], color=TEXT, lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("Seuil de décision (échelle log)")
    ax1.set(ylabel="Valeur", ylim=(0, 1.02))
    ax1.set_title("Précision et rappel selon le seuil (validation)")
    ax1.legend(loc="lower left")
    ax2.set(ylabel=f"Coût attendu = {rule['cost_fn']:g} x FN + {rule['cost_fp']:g} x FP")
    ax2.set_title("Coût attendu selon le seuil (validation)")
    ax2.annotate(
        f"seuil retenu = {selection['threshold']:.3f}",
        xy=(selection["threshold"], selection["expected_cost"]),
        xytext=(10, 25),
        textcoords="offset points",
        fontsize=8,
        color=TEXT,
        arrowprops={"arrowstyle": "-", "color": TEXT_SECONDARY},
    )
    return _save(fig, path)


def plot_sensitivity(rows: list[dict], path: str | Path) -> Path:
    """Sensibilité au bruit : écart moyen de probabilité et part de décisions changées."""
    levels = [r["level"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    ax1.plot(levels, [r["mean_abs_delta"] for r in rows], color=SERIES[0], lw=2, marker="o", ms=6)
    ax1.set(
        xlabel="Amplitude du bruit (x écart-type d'entraînement)",
        ylabel="|delta probabilité| moyen",
    )
    ax1.set_title("Variation moyenne de la probabilité")
    ax2.plot(
        levels,
        [100 * r["flip_rate"] for r in rows],
        color=SERIES[0],
        lw=2,
        marker="o",
        ms=6,
        label="toutes transactions",
    )
    ax2.plot(
        levels,
        [100 * r["flip_rate_frauds"] for r in rows],
        color=SERIES[1],
        lw=2,
        marker="o",
        ms=6,
        label="fraudes réelles",
    )
    ax2.set(
        xlabel="Amplitude du bruit (x écart-type d'entraînement)", ylabel="Décisions changées (%)"
    )
    ax2.set_title("Part des décisions modifiées")
    ax2.legend(loc="upper left")
    return _save(fig, path)
