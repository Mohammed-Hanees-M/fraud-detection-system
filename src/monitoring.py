"""
monitoring.py
─────────────
Model-drift monitoring via Population Stability Index (PSI).

PSI measures how much a feature distribution has shifted between
a baseline (training) window and a production (scoring) window.

PSI < 0.10  → No significant change
PSI 0.10–0.20 → Minor shift — monitor
PSI > 0.20  → Major shift — consider retraining

Exports
───────
compute_psi()           → PSI for a single feature
compute_feature_psi()   → PSI for all features (DataFrame)
simulate_drift()        → creates time-windowed splits for demo
plot_psi_dashboard()    → saves a bar chart of PSI values
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    PSI_BUCKETS, PSI_WARNING_THRESHOLD, PSI_ALERT_THRESHOLD,
    REPORTS_DIR, RANDOM_STATE,
)


# ── 1. Core PSI computation ───────────────────────────────────────────────────

def compute_psi(
    baseline: np.ndarray,
    current: np.ndarray,
    buckets: int = PSI_BUCKETS,
) -> float:
    """
    Compute PSI between baseline and current distributions.

    Parameters
    ----------
    baseline : 1-D array of feature values from training window
    current  : 1-D array of feature values from scoring window
    buckets  : number of equal-frequency bins (default 10)

    Returns
    -------
    float : PSI value
    """
    # Build quantile-based bin edges from baseline
    breakpoints = np.nanpercentile(
        baseline, np.linspace(0, 100, buckets + 1)
    )
    breakpoints = np.unique(breakpoints)            # remove duplicate edges
    if len(breakpoints) < 2:
        return 0.0                                  # degenerate feature

    def _proportions(arr):
        counts, _ = np.histogram(arr, bins=breakpoints)
        counts = np.where(counts == 0, 0.0001, counts)   # avoid log(0)
        return counts / counts.sum()

    p_base = _proportions(baseline)
    p_curr = _proportions(current)

    psi = np.sum((p_curr - p_base) * np.log(p_curr / p_base))
    return float(psi)


# ── 2. Feature-level PSI DataFrame ───────────────────────────────────────────

def compute_feature_psi(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns [feature, psi, status].
    """
    records = []
    common = [c for c in baseline_df.columns if c in current_df.columns]
    for col in common:
        psi = compute_psi(baseline_df[col].values, current_df[col].values)
        if psi < PSI_WARNING_THRESHOLD:
            status = "✅ Stable"
        elif psi < PSI_ALERT_THRESHOLD:
            status = "⚠️  Warning"
        else:
            status = "🚨 Alert"
        records.append({"feature": col, "psi": psi, "status": status})

    result = pd.DataFrame(records).sort_values("psi", ascending=False)
    return result.reset_index(drop=True)


# ── 3. Simulate drift (demo helper) ──────────────────────────────────────────

def simulate_drift(X: np.ndarray, feature_names: list, n_windows: int = 4):
    """
    Splits X by row-index into n_windows equal time slices and computes
    PSI of each window against the first (baseline) window.

    Returns
    -------
    list of DataFrames, one per subsequent window
    """
    window_size = len(X) // n_windows
    baseline_df = pd.DataFrame(
        X[:window_size], columns=feature_names
    )

    results = []
    for i in range(1, n_windows):
        start = i * window_size
        end   = start + window_size
        curr_df = pd.DataFrame(X[start:end], columns=feature_names)
        psi_df  = compute_feature_psi(baseline_df, curr_df)
        psi_df["window"] = f"W{i+1} vs W1"
        results.append(psi_df)

    return results


# ── 4. Plot PSI dashboard ─────────────────────────────────────────────────────

def plot_psi_dashboard(psi_df: pd.DataFrame, title: str = "Feature PSI Dashboard"):
    """
    Saves a horizontal bar chart colouring features by drift severity.
    """
    top = psi_df.head(20).copy()          # show top 20 by PSI

    colours = []
    for v in top["psi"]:
        if v < PSI_WARNING_THRESHOLD:
            colours.append("#27ae60")     # green
        elif v < PSI_ALERT_THRESHOLD:
            colours.append("#f39c12")     # amber
        else:
            colours.append("#e74c3c")     # red

    fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.5)))
    bars = ax.barh(top["feature"], top["psi"], color=colours)
    ax.axvline(PSI_WARNING_THRESHOLD, color="#f39c12", linestyle="--",
               lw=1.5, label=f"Warning ({PSI_WARNING_THRESHOLD})")
    ax.axvline(PSI_ALERT_THRESHOLD, color="#e74c3c", linestyle="--",
               lw=1.5, label=f"Alert ({PSI_ALERT_THRESHOLD})")
    ax.set(xlabel="PSI", title=title)
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    plt.tight_layout()

    path = REPORTS_DIR / "psi_dashboard.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plot] PSI dashboard saved → {path}")
    return path
