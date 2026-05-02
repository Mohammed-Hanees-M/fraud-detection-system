"""
features.py
───────────
Feature engineering pipeline for the fraud detection system.

Functions
─────────
load_raw()              → raw DataFrame
engineer_features()     → feature-engineered DataFrame
build_train_test()      → X_train, X_test, y_train, y_test (scaled)
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from src.config import (
    RAW_CSV, SCALER_PATH, RANDOM_STATE, TEST_SIZE,
    SMOTE_STRATEGY, AMOUNT_LOG_FEATURE, AMOUNT_ZSCORE_FEATURE, HOUR_FEATURE,
)


# ── 1. Load ────────────────────────────────────────────────────────────────────

def load_raw() -> pd.DataFrame:
    """Return the raw credit-card dataset."""
    df = pd.read_csv(RAW_CSV)
    print(f"[load_raw] shape={df.shape}, fraud rate={df['Class'].mean():.4%}")
    return df


# ── 2. Engineer features ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds three domain-informed features to the raw data:

    Amount_Log    – log1p transform; reduces right-skew of transaction values
    Amount_ZScore – standardised amount (μ=0, σ=1) within the dataset
    Hour          – time-of-day bucket derived from 'Time' (seconds elapsed
                    since first transaction, mod 86400 → hour bucket)
    """
    df = df.copy()

    # Log-transform Amount
    df[AMOUNT_LOG_FEATURE]    = np.log1p(df["Amount"])

    # Z-score of Amount
    mu, sigma = df["Amount"].mean(), df["Amount"].std()
    df[AMOUNT_ZSCORE_FEATURE] = (df["Amount"] - mu) / sigma

    # Hour of day  (Time is seconds elapsed since first tx; wrap at 24h)
    df[HOUR_FEATURE] = (df["Time"] % 86_400) // 3_600

    # Drop original Amount and Time (replaced by engineered versions)
    df.drop(columns=["Time", "Amount"], inplace=True)

    return df


# ── 3. Build train / test split with SMOTE ────────────────────────────────────

def build_train_test(apply_smote: bool = True):
    """
    Full preprocessing pipeline:
      load → engineer → split → scale → (optionally) SMOTE

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test  (numpy arrays)
    feature_names                                     (list[str])
    """
    df = engineer_features(load_raw())

    y = df["Class"].values
    X = df.drop(columns=["Class"])
    feature_names = X.columns.tolist()

    # 80 / 20 initial split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 10 % of train → validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=RANDOM_STATE, stratify=y_temp
    )

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[build_train_test] Scaler saved → {SCALER_PATH}")

    # SMOTE on training set only
    if apply_smote:
        sm = SMOTE(sampling_strategy=SMOTE_STRATEGY, random_state=RANDOM_STATE)
        X_train_s, y_train = sm.fit_resample(X_train_s, y_train)
        print(
            f"[build_train_test] After SMOTE: "
            f"fraud={y_train.sum()}, non-fraud={(y_train==0).sum()}"
        )

    return X_train_s, X_val_s, X_test_s, y_train, y_val, y_test, feature_names


# ── 4. Convenience: scale a single transaction dict ───────────────────────────

def preprocess_single(txn: dict) -> np.ndarray:
    """
    Given a dict of raw feature values (as the user would submit in the UI),
    apply the saved scaler and return a 1-row numpy array.
    """
    scaler: StandardScaler = joblib.load(SCALER_PATH)
    df = pd.DataFrame([txn])

    # Ensure engineered features are present
    if AMOUNT_LOG_FEATURE not in df.columns:
        df[AMOUNT_LOG_FEATURE]    = np.log1p(df["Amount"])
        mu, sigma = 88.35, 250.12          # dataset-level constants (approximation)
        df[AMOUNT_ZSCORE_FEATURE] = (df["Amount"] - mu) / sigma
        df[HOUR_FEATURE]          = (df["Time"] % 86_400) // 3_600
        df.drop(columns=["Time", "Amount"], inplace=True)

    return scaler.transform(df)
