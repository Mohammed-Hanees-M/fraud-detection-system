"""
pipeline.py
───────────
Orchestrates the full training pipeline end-to-end.

Run directly:
    python -m src.pipeline

Or import and call:
    from src.pipeline import run_pipeline
    results = run_pipeline()
"""

import json
import numpy as np
from src.features import build_train_test
from src.models import (
    train_isolation_forest,
    train_xgboost,
    train_lightgbm,
    iso_forest_scores,
)
from src.evaluation import (
    evaluate_classifier,
    evaluate_iso_forest,
)
from src.monitoring import simulate_drift, plot_psi_dashboard
from src.config import REPORTS_DIR


def run_pipeline(optuna_trials: int = 30) -> dict:
    """
    Full ML pipeline.

    Steps
    ─────
    1. Feature engineering + SMOTE
    2. Train Isolation Forest
    3. Train XGBoost (Optuna)
    4. Train LightGBM (Optuna)
    5. Evaluate all three on held-out test set
    6. Simulate drift across 4 time windows
    7. Save comparison report as JSON

    Returns
    -------
    dict : evaluation metrics for all three models
    """
    print("\n" + "="*60)
    print("  FRAUD DETECTION PIPELINE — START")
    print("="*60)

    # ── Step 1: Features ──────────────────────────────────────────
    print("\n[1/6] Feature Engineering …")
    X_train, X_val, X_test, y_train, y_val, y_test, feat_names = build_train_test()

    # ── Step 2: Isolation Forest ──────────────────────────────────
    print("\n[2/6] Isolation Forest …")
    iso = train_isolation_forest(X_train)

    # ── Step 3: XGBoost ───────────────────────────────────────────
    print(f"\n[3/6] XGBoost ({optuna_trials} Optuna trials) …")
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val, n_trials=optuna_trials)

    # ── Step 4: LightGBM ──────────────────────────────────────────
    print(f"\n[4/6] LightGBM ({optuna_trials} Optuna trials) …")
    lgbm_model = train_lightgbm(X_train, y_train, X_val, y_val, n_trials=optuna_trials)

    # ── Step 5: Evaluation ────────────────────────────────────────
    print("\n[5/6] Evaluation …")
    res_iso  = evaluate_iso_forest(iso, X_test, y_test)
    res_xgb  = evaluate_classifier(
        "XGBoost", y_test,
        xgb_model.predict_proba(X_test)[:, 1],
    )
    res_lgbm = evaluate_classifier(
        "LightGBM", y_test,
        lgbm_model.predict_proba(X_test)[:, 1],
    )

    # ── Step 6: Drift ─────────────────────────────────────────────
    print("\n[6/6] Drift Monitoring …")
    drift_results = simulate_drift(X_test, feat_names, n_windows=4)
    if drift_results:
        plot_psi_dashboard(drift_results[0], title="PSI — W2 vs W1 (test split)")

    # ── Save comparison JSON ──────────────────────────────────────
    comparison = {m["model"]: m for m in [res_iso, res_xgb, res_lgbm]}
    report_path = REPORTS_DIR / "model_comparison.json"
    with open(report_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n[✓] Comparison report → {report_path}")

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print("="*60)
    return comparison


if __name__ == "__main__":
    run_pipeline()
