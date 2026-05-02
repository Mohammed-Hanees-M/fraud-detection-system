"""
evaluation.py
─────────────
All evaluation utilities: metrics, curves, plots.

Exports
───────
evaluate_classifier()   → dict of metrics + saves plots
evaluate_iso_forest()   → dict of metrics for unsupervised model
find_best_threshold()   → threshold that maximises F1
plot_roc_curve()
plot_pr_curve()
plot_confusion_matrix()
ks_statistic()
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.ensemble import IsolationForest

from src.config import REPORTS_DIR


# ── 1. KS Statistic ───────────────────────────────────────────────────────────

def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Kolmogorov–Smirnov statistic: max separation between the cumulative
    fraud and non-fraud score distributions.  Higher = better separation.
    """
    fraud_scores     = np.sort(y_score[y_true == 1])
    non_fraud_scores = np.sort(y_score[y_true == 0])

    thresholds = np.linspace(0, 1, 200)
    ks = 0.0
    for t in thresholds:
        tpr = (fraud_scores >= t).mean()
        fpr = (non_fraud_scores >= t).mean()
        ks  = max(ks, abs(tpr - fpr))
    return ks


# ── 2. Best threshold ─────────────────────────────────────────────────────────

def find_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Returns the probability threshold that maximises F1."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    return float(thresholds[np.argmax(f1s[:-1])])


# ── 3. Evaluation: supervised models ─────────────────────────────────────────

def evaluate_classifier(
    model_name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float | None = None,
    save_plots: bool = True,
) -> dict:
    """
    Full evaluation suite for probabilistic classifiers.

    Parameters
    ----------
    model_name : display name (used in titles / file names)
    y_true     : ground-truth labels (0/1)
    y_score    : predicted fraud probability
    threshold  : if None, best F1 threshold is computed automatically
    save_plots : write PNG files to reports/

    Returns
    -------
    dict with keys: roc_auc, pr_auc, ks, precision, recall, f1, threshold
    """
    if threshold is None:
        threshold = find_best_threshold(y_true, y_score)

    y_pred = (y_score >= threshold).astype(int)

    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc  = average_precision_score(y_true, y_score)
    ks      = ks_statistic(y_true, y_score)
    prec    = precision_score(y_true, y_pred, zero_division=0)
    rec     = recall_score(y_true, y_pred, zero_division=0)
    f1      = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n{'─'*55}")
    print(f"  {model_name}  (threshold={threshold:.3f})")
    print(f"{'─'*55}")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print(f"  PR-AUC    : {pr_auc:.4f}")
    print(f"  KS-Stat   : {ks:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=["Legit", "Fraud"]))

    if save_plots:
        _plot_roc(model_name, y_true, y_score, roc_auc)
        _plot_pr(model_name, y_true, y_score, pr_auc)
        _plot_cm(model_name, y_true, y_pred)

    return dict(
        model=model_name,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        ks=ks,
        precision=prec,
        recall=rec,
        f1=f1,
        threshold=threshold,
    )


# ── 4. Evaluation: Isolation Forest ──────────────────────────────────────────

def evaluate_iso_forest(
    model: IsolationForest,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_plots: bool = True,
) -> dict:
    """Wraps iso_forest_scores → evaluate_classifier."""
    from src.models import iso_forest_scores
    scores = iso_forest_scores(model, X_test)
    return evaluate_classifier(
        "Isolation Forest", y_test, scores, save_plots=save_plots
    )


# ── 5. Plot helpers ───────────────────────────────────────────────────────────

def _safe_name(model_name: str) -> str:
    return model_name.lower().replace(" ", "_")


def _plot_roc(name, y_true, y_score, auc_val):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC-AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title=f"ROC Curve — {name}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = REPORTS_DIR / f"roc_{_safe_name(name)}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plot] Saved {path}")


def _plot_pr(name, y_true, y_score, auc_val):
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    baseline = y_true.mean()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, lw=2, label=f"PR-AUC = {auc_val:.4f}")
    ax.axhline(baseline, color="gray", linestyle="--", lw=1,
               label=f"Baseline (fraud rate={baseline:.3%})")
    ax.set(xlabel="Recall", ylabel="Precision",
           title=f"Precision-Recall Curve — {name}")
    ax.legend(loc="upper right")
    plt.tight_layout()
    path = REPORTS_DIR / f"pr_{_safe_name(name)}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plot] Saved {path}")


def _plot_cm(name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Legit", "Fraud"],
        yticklabels=["Legit", "Fraud"], ax=ax,
    )
    ax.set(xlabel="Predicted", ylabel="Actual",
           title=f"Confusion Matrix — {name}")
    plt.tight_layout()
    path = REPORTS_DIR / f"cm_{_safe_name(name)}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plot] Saved {path}")
