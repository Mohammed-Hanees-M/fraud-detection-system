"""
models.py
─────────
Model training, hyperparameter tuning, and persistence.

Models
──────
1. Isolation Forest   – unsupervised anomaly detector
2. XGBoost            – gradient boosting (supervised)
3. LightGBM           – fast gradient boosting (supervised)
"""

import numpy as np
import joblib
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb

from src.config import (
    ISO_FOREST_PATH, XGB_PATH, LGBM_PATH, RANDOM_STATE,
)


# ── 1. Isolation Forest ───────────────────────────────────────────────────────

def train_isolation_forest(X_train: np.ndarray) -> IsolationForest:
    """
    Fits an Isolation Forest on the (unbalanced) training features.
    contamination is set to the approximate fraud rate in the dataset.
    """
    print("[IsoForest] Training …")
    model = IsolationForest(
        n_estimators=200,
        contamination=0.0017,       # ~0.17 % fraud rate
        max_samples="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train)
    joblib.dump(model, ISO_FOREST_PATH)
    print(f"[IsoForest] Saved → {ISO_FOREST_PATH}")
    return model


def iso_forest_scores(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """
    Returns anomaly scores in [0, 1] where higher = more anomalous.
    Isolation Forest's decision_function returns negative values for
    inliers, so we invert and normalise.
    """
    raw = model.decision_function(X)          # lower = more anomalous
    scores = 1 - (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    return scores


# ── 2. XGBoost ────────────────────────────────────────────────────────────────

def _xgb_objective(trial, X_train, y_train, X_val, y_val):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.0, 1.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
        "use_label_encoder": False,
        "eval_metric": "aucpr",
        "random_state": RANDOM_STATE,
        "tree_method": "hist",
        "n_jobs": -1,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    proba = model.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, proba)


def train_xgboost(
    X_train, y_train, X_val, y_val, n_trials: int = 40
) -> xgb.XGBClassifier:
    print("[XGBoost] Hyperparameter search …")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda t: _xgb_objective(t, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    best = study.best_params
    print(f"[XGBoost] Best PR-AUC={study.best_value:.4f} | params={best}")

    model = xgb.XGBClassifier(
        **best,
        use_label_encoder=False,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        tree_method="hist",
        n_jobs=-1,
    )
    model.fit(
        np.vstack([X_train, X_val]),
        np.concatenate([y_train, y_val]),
    )
    joblib.dump(model, XGB_PATH)
    print(f"[XGBoost] Saved → {XGB_PATH}")
    return model


# ── 3. LightGBM ───────────────────────────────────────────────────────────────

def _lgbm_objective(trial, X_train, y_train, X_val, y_val):
    params = {
        "n_estimators":       trial.suggest_int("n_estimators", 100, 500),
        "max_depth":          trial.suggest_int("max_depth", 3, 8),
        "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves":         trial.suggest_int("num_leaves", 20, 150),
        "subsample":          trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples":  trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha":          trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda":         trial.suggest_float("reg_lambda", 0.0, 1.0),
        "scale_pos_weight":   trial.suggest_float("scale_pos_weight", 1.0, 10.0),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(-1)],
    )
    proba = model.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, proba)


def train_lightgbm(
    X_train, y_train, X_val, y_val, n_trials: int = 40
) -> lgb.LGBMClassifier:
    print("[LightGBM] Hyperparameter search …")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda t: _lgbm_objective(t, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    best = study.best_params
    print(f"[LightGBM] Best PR-AUC={study.best_value:.4f} | params={best}")

    model = lgb.LGBMClassifier(
        **best,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        np.vstack([X_train, X_val]),
        np.concatenate([y_train, y_val]),
    )
    joblib.dump(model, LGBM_PATH)
    print(f"[LightGBM] Saved → {LGBM_PATH}")
    return model


# ── 4. Load persisted models ──────────────────────────────────────────────────

def load_models():
    """Returns (iso_forest, xgb_model, lgbm_model, scaler)."""
    from src.config import SCALER_PATH
    return (
        joblib.load(ISO_FOREST_PATH),
        joblib.load(XGB_PATH),
        joblib.load(LGBM_PATH),
        joblib.load(SCALER_PATH),
    )
