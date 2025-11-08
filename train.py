#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train multi-target QSAR (CatBoost via Optuna) with feature selection.
- Split train/test
- VarianceThreshold + correlation filter (train-only)
- Optuna HPO on CatBoost wrapped in MultiOutputRegressor
- Save: trained model, feature columns, and AD PCA stats for later screening
Outputs:
  - model_catboost_multi.pkl
  - feature_cols.txt
  - ad_params.joblib
  - parity_plot.png, residual_plot.png, pIC50_distr.png
"""

import argparse
import json
import joblib
import numpy as np
import optuna
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer
from sklearn.decomposition import PCA

from catboost import CatBoostRegressor


def drop_correlated(X: pd.DataFrame, thr: float = 0.9) -> pd.DataFrame:
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > thr)]
    return X.drop(columns=to_drop), to_drop


def parity_and_residual_plots(y_train, y_train_pred, y_val, y_val_pred, targets=('IDO1','TDO')):
    # Parity
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i in range(2):
        ax = axes[i]
        ax.scatter(y_train.iloc[:, i], y_train_pred[:, i], alpha=0.6, label=f"Train")
        ax.scatter(y_val.iloc[:, i], y_val_pred[:, i], alpha=0.6, label=f"Val")
        lo = min(y_train.iloc[:, i].min(), y_val.iloc[:, i].min())
        hi = max(y_train.iloc[:, i].max(), y_val.iloc[:, i].max())
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
        ax.set_title(f"{targets[i]} Parity Plot")
        ax.set_xlabel("Actual pIC50")
        ax.set_ylabel("Predicted pIC50")
        ax.legend()
    plt.tight_layout()
    plt.savefig("parity_plot.png", dpi=300)

    # Residual
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i in range(2):
        ax = axes[i]
        resid_tr = y_train_pred[:, i] - y_train.iloc[:, i].values
        resid_va = y_val_pred[:, i] - y_val.iloc[:, i].values
        ax.scatter(y_train.iloc[:, i], resid_tr, alpha=0.6, label="Train")
        ax.scatter(y_val.iloc[:, i], resid_va, alpha=0.6, label="Val")
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set_title(f"{targets[i]} Residual Plot")
        ax.set_xlabel("Actual pIC50")
        ax.set_ylabel("Residual")
        ax.legend()
    plt.tight_layout()
    plt.savefig("residual_plot.png", dpi=300)


def main(args):
    df = pd.read_csv(args.in_csv)

    y = df[['pIC50_IDO1', 'pIC50_TDO']].copy()
    X = df.drop(columns=['pIC50_IDO1', 'pIC50_TDO', 'smiles'], errors='ignore')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Plot distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    axes[0].hist(y_train['pIC50_IDO1'], bins=20, alpha=0.7, edgecolor="black")
    axes[0].hist(y_test['pIC50_IDO1'], bins=20, alpha=0.7, edgecolor="black")
    axes[0].set_title("pIC50_IDO1 Distribution")
    axes[1].hist(y_train['pIC50_TDO'], bins=20, alpha=0.7, edgecolor="black")
    axes[1].hist(y_test['pIC50_TDO'], bins=20, alpha=0.7, edgecolor="black")
    axes[1].set_title("pIC50_TDO Distribution")
    plt.tight_layout()
    plt.savefig("pIC50_distr.png", dpi=300)

    # Feature selection (train only)
    var = VarianceThreshold(threshold=0.1)
    X_train_var = var.fit_transform(X_train)
    kept_var_cols = X_train.columns[var.get_support()]
    X_train_var_df = pd.DataFrame(X_train_var, columns=kept_var_cols)

    X_train_clean, dropped_corr = drop_correlated(X_train_var_df, thr=0.9)
    # Align test
    X_test_clean = X_test[kept_var_cols].drop(columns=[c for c in dropped_corr if c in kept_var_cols], errors='ignore')

    # Optuna HPO on CatBoost
    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 200, 1000),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "random_strength": trial.suggest_float("random_strength", 1e-9, 10.0, log=True),
            "verbose": 0,
            "task_type": "CPU",
            "random_seed": 42,
        }
        model = MultiOutputRegressor(CatBoostRegressor(**params))
        score = cross_val_score(
            model, X_train_clean, y_train, cv=cv,
            scoring=make_scorer(r2_score, multioutput='uniform_average'),
            n_jobs=-1
        ).mean()
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    best_params = study.best_params
    base = CatBoostRegressor(
        random_seed=42,
        early_stopping_rounds=30,
        verbose=0,
        **best_params
    )
    model = MultiOutputRegressor(base)
    X_tr, X_va, y_tr, y_va = train_test_split(X_train_clean, y_train, test_size=0.2, random_state=42)
    model.fit(X_tr, y_tr)

    # Evaluate
    y_tr_pred = model.predict(X_tr)
    y_va_pred = model.predict(X_va)

    print(f"Train R2: {r2_score(y_tr, y_tr_pred, multioutput='uniform_average'):.3f}")
    print(f"Val   R2: {r2_score(y_va, y_va_pred, multioutput='uniform_average'):.3f}")

    parity_and_residual_plots(y_tr, y_tr_pred, y_va, y_va_pred)

    # Save model
    joblib.dump(model, "model_catboost_multi.pkl")

    # Save feature columns (order matters)
    feature_cols = X_train_clean.columns.tolist()
    with open("feature_cols.txt", "w") as f:
        for c in feature_cols:
            f.write(f"{c}\n")

    # Save AD PCA parameters for screening (fit PCA on train_clean)
    pca = PCA(n_components=20)
    X_train_pca = pca.fit_transform(X_train_clean[feature_cols])
    mean_vec = np.mean(X_train_pca, axis=0)
    cov = np.cov(X_train_pca, rowvar=False)
    inv_cov = np.linalg.pinv(cov)
    # chi2 threshold for df=20, 95%
    from scipy.stats import chi2
    threshold = float(np.sqrt(chi2.ppf(0.95, df=20)))

    ad_blob = {
        "pca": pca,
        "mean_vec": mean_vec,
        "inv_cov": inv_cov,
        "threshold": threshold,
        "n_components": 20,
        "feature_cols": feature_cols
    }
    joblib.dump(ad_blob, "ad_params.joblib")

    print("[OK] Saved model_catboost_multi.pkl, feature_cols.txt, ad_params.joblib")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", default="ido1_tdo_ecfp4.csv")
    ap.add_argument("--n-trials", type=int, default=10)
    main(ap.parse_args())
