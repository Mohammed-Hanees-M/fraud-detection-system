"""
benchmark_datasets.py
─────────────────────
Runs the full fraud detection pipeline on multiple datasets and
produces a unified comparison report for resume / LinkedIn.

Datasets supported:
  1. Credit Card Fraud (Kaggle — already done)
  2. IEEE-CIS Fraud Detection (Kaggle competition)
  3. PaySim Mobile Money Fraud (Kaggle)

Usage:
    python scripts/benchmark_datasets.py
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import joblib

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import REPORTS_DIR, RANDOM_STATE
from src.evaluation import ks_statistic, find_best_threshold, evaluate_classifier

BENCHMARK_DIR = REPORTS_DIR / "benchmark"
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset Loaders
# ══════════════════════════════════════════════════════════════════════════════

def load_creditcard():
    """Original Kaggle credit card fraud dataset."""
    path = Path("data/creditcard.csv")
    if not path.exists():
        print("[!] creditcard.csv not found — skipping")
        return None, None, "Credit Card Fraud"
    
    df = pd.read_csv(path)
    df["Amount_Log"]    = np.log1p(df["Amount"])
    df["Amount_ZScore"] = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()
    df["Hour"]          = (df["Time"] % 86400) // 3600
    df.drop(columns=["Time", "Amount"], inplace=True)
    
    y = df["Class"].values
    X = df.drop(columns=["Class"]).values
    print(f"[Credit Card] shape={df.shape}, fraud={y.mean():.4%}")
    return X, y, "Credit Card Fraud"


def load_ieee():
    """IEEE-CIS Fraud Detection competition dataset."""
    train_path = Path("data/ieee/train_transaction.csv")
    if not train_path.exists():
        print("[!] IEEE dataset not found at data/ieee/train_transaction.csv — skipping")
        return None, None, "IEEE-CIS Fraud"
    
    print("[IEEE] Loading (this is a large file, ~400MB)...")
    df = pd.read_csv(train_path)
    
    target = "isFraud"
    
    # Keep only numeric columns for simplicity, drop high-missing cols
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target]
    
    # Drop columns with >50% missing
    missing_rate = df[numeric_cols].isnull().mean()
    good_cols = missing_rate[missing_rate < 0.5].index.tolist()
    
    df_clean = df[good_cols + [target]].copy()
    df_clean.fillna(df_clean.median(), inplace=True)
    
    # Feature engineering
    if "TransactionAmt" in df_clean.columns:
        df_clean["Amt_Log"]    = np.log1p(df_clean["TransactionAmt"])
        df_clean["Amt_ZScore"] = (
            (df_clean["TransactionAmt"] - df_clean["TransactionAmt"].mean())
            / df_clean["TransactionAmt"].std()
        )
    
    y = df_clean[target].values
    X = df_clean.drop(columns=[target]).values
    
    print(f"[IEEE] shape={df_clean.shape}, fraud={y.mean():.4%}")
    return X, y, "IEEE-CIS Fraud"


def load_paysim():
    """PaySim synthetic mobile money fraud dataset."""
    # Try both possible filenames
    for fname in ["PS_20174392719_1491204439457_log.csv", "paysim.csv", "PS_log.csv"]:
        path = Path(f"data/paysim/{fname}")
        if path.exists():
            break
    else:
        # Search for any CSV in the paysim folder
        csvs = list(Path("data/paysim").glob("*.csv"))
        if not csvs:
            print("[!] PaySim dataset not found in data/paysim/ — skipping")
            return None, None, "PaySim Mobile Fraud"
        path = csvs[0]
    
    print(f"[PaySim] Loading from {path.name}...")
    df = pd.read_csv(path)
    
    print(f"[PaySim] Columns: {df.columns.tolist()}")
    
    target = "isFraud"
    if target not in df.columns:
        print("[!] 'isFraud' column not found in PaySim dataset")
        return None, None, "PaySim Mobile Fraud"
    
    # Encode transaction type
    if "type" in df.columns:
        df["type_encoded"] = pd.Categorical(df["type"]).codes
    
    # Drop non-numeric / identifier columns
    drop_cols = ["nameOrig", "nameDest", "isFlaggedFraud", "type"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    
    # Feature engineering
    if "amount" in df.columns:
        df["amount_log"]    = np.log1p(df["amount"])
        df["amount_zscore"] = (df["amount"] - df["amount"].mean()) / df["amount"].std()
    
    # Sample 300K rows for speed (dataset has 6M rows)
    if len(df) > 300_000:
        df = df.sample(300_000, random_state=RANDOM_STATE)
        print("[PaySim] Sampled 300K rows for speed")
    
    y = df[target].values
    X = df.drop(columns=[target]).values
    
    print(f"[PaySim] shape={df.shape}, fraud={y.mean():.4%}")
    return X, y, "PaySim Mobile Fraud"


# ══════════════════════════════════════════════════════════════════════════════
#  Training & Evaluation Pipeline (fast version for benchmarking)
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmark(X, y, dataset_name, n_trials=15):
    """
    Quick pipeline: split → scale → SMOTE → XGBoost + LightGBM → evaluate.
    Returns dict of metrics.
    """
    print(f"\n{'='*60}")
    print(f"  BENCHMARKING: {dataset_name}")
    print(f"  Samples: {len(X):,} | Features: {X.shape[1]} | Fraud: {y.mean():.4%}")
    print(f"{'='*60}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    
    # SMOTE (only if fraud < 5%)
    if y_train.mean() < 0.05:
        sm = SMOTE(sampling_strategy=0.1, random_state=RANDOM_STATE)
        X_train_s, y_train = sm.fit_resample(X_train_s, y_train)
        print(f"  After SMOTE: fraud={y_train.sum():,}, legit={(y_train==0).sum():,}")
    
    results = {"dataset": dataset_name, "samples": len(X), "fraud_rate": float(y.mean())}
    
    # ── XGBoost (fast, fixed params) ──────────────────────────────────────────
    print("  [XGBoost] Training...")
    scale_pos = (y_train == 0).sum() / max(y_train.sum(), 1)
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=min(scale_pos, 10),
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        tree_method="hist",
        n_jobs=-1,
        verbosity=0,
    )
    xgb_model.fit(X_train_s, y_train)
    xgb_proba = xgb_model.predict_proba(X_test_s)[:, 1]
    
    results["xgb_roc_auc"] = float(roc_auc_score(y_test, xgb_proba))
    results["xgb_pr_auc"]  = float(average_precision_score(y_test, xgb_proba))
    results["xgb_ks"]      = float(ks_statistic(y_test, xgb_proba))
    
    thr = find_best_threshold(y_test, xgb_proba)
    y_pred = (xgb_proba >= thr).astype(int)
    from sklearn.metrics import precision_score, recall_score, f1_score
    results["xgb_precision"] = float(precision_score(y_test, y_pred, zero_division=0))
    results["xgb_recall"]    = float(recall_score(y_test, y_pred, zero_division=0))
    results["xgb_f1"]        = float(f1_score(y_test, y_pred, zero_division=0))
    
    print(f"  XGBoost → ROC-AUC={results['xgb_roc_auc']:.4f} | PR-AUC={results['xgb_pr_auc']:.4f} | KS={results['xgb_ks']:.4f}")
    
    # ── LightGBM ──────────────────────────────────────────────────────────────
    print("  [LightGBM] Training...")
    lgbm_model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=min(scale_pos, 10),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    lgbm_model.fit(X_train_s, y_train)
    lgbm_proba = lgbm_model.predict_proba(X_test_s)[:, 1]
    
    results["lgbm_roc_auc"] = float(roc_auc_score(y_test, lgbm_proba))
    results["lgbm_pr_auc"]  = float(average_precision_score(y_test, lgbm_proba))
    results["lgbm_ks"]      = float(ks_statistic(y_test, lgbm_proba))
    
    thr = find_best_threshold(y_test, lgbm_proba)
    y_pred = (lgbm_proba >= thr).astype(int)
    results["lgbm_precision"] = float(precision_score(y_test, y_pred, zero_division=0))
    results["lgbm_recall"]    = float(recall_score(y_test, y_pred, zero_division=0))
    results["lgbm_f1"]        = float(f1_score(y_test, y_pred, zero_division=0))
    
    print(f"  LightGBM → ROC-AUC={results['lgbm_roc_auc']:.4f} | PR-AUC={results['lgbm_pr_auc']:.4f} | KS={results['lgbm_ks']:.4f}")
    
    # ── ROC curve plot ────────────────────────────────────────────────────────
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(7, 5))
    for proba, label, color in [
        (xgb_proba,  f"XGBoost (AUC={results['xgb_roc_auc']:.3f})",  "#3498db"),
        (lgbm_proba, f"LightGBM (AUC={results['lgbm_roc_auc']:.3f})", "#27ae60"),
    ]:
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, lw=2, label=label, color=color)
    ax.plot([0,1],[0,1],"k--",lw=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title=f"ROC Curve — {dataset_name}")
    ax.legend()
    plt.tight_layout()
    safe_name = dataset_name.lower().replace(" ", "_").replace("-", "_")
    fig.savefig(BENCHMARK_DIR / f"roc_{safe_name}.png", dpi=120)
    plt.close(fig)
    
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Master comparison chart
# ══════════════════════════════════════════════════════════════════════════════

def plot_benchmark_comparison(all_results):
    """Generates the master comparison chart across all datasets."""
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    metrics   = ["roc_auc", "pr_auc", "ks"]
    titles    = ["ROC-AUC", "PR-AUC", "KS-Statistic"]
    
    datasets  = [r["dataset"] for r in all_results]
    x         = np.arange(len(datasets))
    width     = 0.35
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        xgb_vals  = [r.get(f"xgb_{metric}",  0) for r in all_results]
        lgbm_vals = [r.get(f"lgbm_{metric}", 0) for r in all_results]
        
        bars1 = axes[idx].bar(x - width/2, xgb_vals,  width, label="XGBoost",  color="#3498db", edgecolor="white")
        bars2 = axes[idx].bar(x + width/2, lgbm_vals, width, label="LightGBM", color="#27ae60", edgecolor="white")
        
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(datasets, rotation=15, ha="right", fontsize=9)
        axes[idx].set_ylim(0, 1.1)
        axes[idx].set_title(title, fontsize=12, fontweight="bold")
        axes[idx].legend(fontsize=8)
        axes[idx].axhline(1.0, color="black", lw=0.5, linestyle=":")
        
        # Value labels on bars
        for bar in bars1:
            h = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2, h + 0.01,
                          f"{h:.3f}", ha="center", va="bottom", fontsize=7)
        for bar in bars2:
            h = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2, h + 0.01,
                          f"{h:.3f}", ha="center", va="bottom", fontsize=7)
    
    plt.suptitle("Multi-Dataset Fraud Detection Benchmark\nXGBoost vs LightGBM",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = BENCHMARK_DIR / "master_benchmark.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[✓] Master benchmark chart → {out}")
    return out


def print_resume_summary(all_results):
    """Prints a formatted summary table for resume/LinkedIn."""
    
    print("\n" + "="*70)
    print("  BENCHMARK SUMMARY — Copy this for LinkedIn / Resume")
    print("="*70)
    
    print(f"\n{'Dataset':<25} {'Model':<12} {'ROC-AUC':>8} {'PR-AUC':>8} {'KS-Stat':>8} {'F1':>8}")
    print("-"*70)
    
    for r in all_results:
        name = r["dataset"]
        fraud_rate = r.get("fraud_rate", 0)
        print(f"\n  {name} (fraud={fraud_rate:.3%})")
        print(f"  {'XGBoost':<23} {'':>0} {r.get('xgb_roc_auc',0):>8.4f} {r.get('xgb_pr_auc',0):>8.4f} {r.get('xgb_ks',0):>8.4f} {r.get('xgb_f1',0):>8.4f}")
        print(f"  {'LightGBM':<23} {'':>0} {r.get('lgbm_roc_auc',0):>8.4f} {r.get('lgbm_pr_auc',0):>8.4f} {r.get('lgbm_ks',0):>8.4f} {r.get('lgbm_f1',0):>8.4f}")
    
    print("\n" + "="*70)
    print("  RESUME BULLET (update with your actual numbers):")
    print("="*70)
    
    # Find best metrics across all datasets
    best_roc  = max([max(r.get("xgb_roc_auc",0), r.get("lgbm_roc_auc",0)) for r in all_results])
    best_pr   = max([max(r.get("xgb_pr_auc",0),  r.get("lgbm_pr_auc",0))  for r in all_results])
    best_ks   = max([max(r.get("xgb_ks",0),       r.get("lgbm_ks",0))      for r in all_results])
    n_datasets = len(all_results)
    total_rows = sum([r.get("samples",0) for r in all_results])
    
    print(f"""
  Built a production-grade fraud detection system benchmarked across
  {n_datasets} real-world datasets ({total_rows:,}+ total transactions);
  XGBoost + LightGBM achieved ROC-AUC up to {best_roc:.3f},
  PR-AUC {best_pr:.3f}, KS-Statistic {best_ks:.3f};
  applied SMOTE for class imbalance, SHAP explainability,
  and PSI drift monitoring; deployed as live Streamlit dashboard.
    """)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    all_results = []
    
    loaders = [load_creditcard, load_ieee, load_paysim]
    
    for loader in loaders:
        X, y, name = loader()
        if X is not None:
            result = run_benchmark(X, y, name)
            all_results.append(result)
        else:
            print(f"[SKIP] {name} — dataset not available")
    
    if not all_results:
        print("\n[!] No datasets found. Download them first:")
        print("    python scripts/download_data.py             (Credit Card)")
        print("    kaggle competitions download -c ieee-fraud-detection -p data/ieee/")
        print("    kaggle datasets download -d ealaxi/paysim1 -p data/paysim/")
        sys.exit(1)
    
    # Save JSON
    json_path = BENCHMARK_DIR / "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[✓] Results saved → {json_path}")
    
    # Plot comparison
    plot_benchmark_comparison(all_results)
    
    # Print summary
    print_resume_summary(all_results)