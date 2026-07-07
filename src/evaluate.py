"""Scoring, plots, and a business-level backtest.

The classification metrics answer "is the model any good at predicting
reversion?". The backtest answers the question a desk actually cares about:
"if we only take the trades the model flags, do we win more often?".
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix, roc_curve, precision_recall_curve)

from .plotting import save_fig


def metrics_table(y_true, preds: dict) -> pd.DataFrame:
    """One row per model with the standard classification metrics."""
    rows = []
    for name, p in preds.items():
        yp, proba = p["pred"], p["proba"]
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_true, yp),
            "precision": precision_score(y_true, yp, zero_division=0),
            "recall": recall_score(y_true, yp, zero_division=0),
            "f1": f1_score(y_true, yp, zero_division=0),
            "roc_auc": roc_auc_score(y_true, proba) if len(np.unique(y_true)) > 1 else np.nan,
            "pr_auc": average_precision_score(y_true, proba) if len(np.unique(y_true)) > 1 else np.nan,
        })
    return pd.DataFrame(rows).set_index("model").round(3)


def plot_confusion(y_true, y_pred, title, fname):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "#0d1b2a", fontsize=13)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["no revert", "revert"]); ax.set_yticklabels(["no revert", "revert"])
    ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.set_title(title)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = save_fig(fig, fname)
    return fig, path


def plot_roc_pr(y_true, preds: dict, fname):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for name, p in preds.items():
        if name == "base_rate":
            continue
        fpr, tpr, _ = roc_curve(y_true, p["proba"])
        ax1.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_true, p['proba']):.2f})")
        prec, rec, _ = precision_recall_curve(y_true, p["proba"])
        ax2.plot(rec, prec, label=name)
    ax1.plot([0, 1], [0, 1], "--", color="#9aa7b4")
    ax1.set_title("ROC"); ax1.set_xlabel("false positive rate"); ax1.set_ylabel("true positive rate"); ax1.legend()
    base = y_true.mean()
    ax2.axhline(base, ls="--", color="#9aa7b4", label=f"base rate {base:.2f}")
    ax2.set_title("Precision-Recall"); ax2.set_xlabel("recall"); ax2.set_ylabel("precision"); ax2.legend()
    fig.tight_layout()
    return fig, save_fig(fig, fname)


def plot_importance(importance: pd.Series, title, fname):
    fig, ax = plt.subplots(figsize=(6, 4))
    importance.iloc[::-1].plot.barh(ax=ax, color="#2f6f9f")
    ax.set_title(title); ax.set_xlabel("importance")
    fig.tight_layout()
    return fig, save_fig(fig, fname)


def backtest(test: pd.DataFrame, proba, threshold: float) -> dict:
    """Compare taking every extreme event vs only model-approved events.

    'Win' = the trade reverted (label == 1). This is a hit-rate backtest, the
    cleanest readout of whether the filter adds value. Returns a summary dict.
    """
    take = proba >= threshold
    all_win = test["reverted"].mean()
    filtered = test.loc[take, "reverted"]
    return {
        "trades_all": int(len(test)),
        "winrate_all": round(float(all_win), 3),
        "trades_model": int(take.sum()),
        "winrate_model": round(float(filtered.mean()), 3) if take.sum() else float("nan"),
        "coverage": round(float(take.mean()), 3),
        "winrate_uplift": round(float(filtered.mean() - all_win), 3) if take.sum() else float("nan"),
    }


# ---------------------------------------------------------------------------
# Business-metric evaluation: precision vs coverage, threshold choice, calibration
# ---------------------------------------------------------------------------
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve


def precision_coverage(y_true, proba, thresholds=None) -> pd.DataFrame:
    """Win rate (precision of the 'trade' signal) and coverage at each threshold."""
    y_true = np.asarray(y_true)
    if thresholds is None:
        thresholds = np.round(np.arange(0.40, 0.86, 0.05), 2)
    rows = []
    for t in thresholds:
        take = proba >= t
        n = int(take.sum())
        rows.append({"threshold": float(t),
                     "coverage": round(float(take.mean()), 3),
                     "trades": n,
                     "winrate": round(float(y_true[take].mean()), 3) if n else np.nan})
    return pd.DataFrame(rows)


def pick_threshold(y_true, proba, min_coverage: float = 0.25) -> float:
    """Threshold that maximises win rate subject to a minimum coverage.

    Chosen on a validation slice, never on the test set.
    """
    pc = precision_coverage(y_true, proba, np.round(np.arange(0.40, 0.91, 0.01), 2))
    ok = pc[pc["coverage"] >= min_coverage].dropna()
    if ok.empty:
        return 0.5
    return float(ok.sort_values("winrate", ascending=False).iloc[0]["threshold"])


def plot_precision_coverage(y_true, proba, fname, base_rate=None):
    pc = precision_coverage(y_true, proba, np.round(np.arange(0.40, 0.86, 0.02), 2))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(pc["coverage"], pc["winrate"], marker="o", color="#3fb950")
    if base_rate is not None:
        ax.axhline(base_rate, ls="--", color="#b5495b", label=f"trade-all win rate {base_rate:.2f}")
        ax.legend()
    ax.set_xlabel("coverage (share of events traded)")
    ax.set_ylabel("win rate (precision)")
    ax.set_title("Win rate vs coverage")
    fig.tight_layout()
    return fig, save_fig(fig, fname)


def plot_calibration(y_true, proba, fname, n_bins: int = 8):
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=n_bins, strategy="quantile")
    brier = brier_score_loss(y_true, proba)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="#9aa7b4", label="perfect")
    ax.plot(mean_pred, frac_pos, marker="o", color="#2f6f9f", label=f"model (Brier {brier:.3f})")
    ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency")
    ax.set_title("Reliability curve"); ax.legend()
    fig.tight_layout()
    return fig, save_fig(fig, fname), float(brier)
