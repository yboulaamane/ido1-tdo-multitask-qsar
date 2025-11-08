#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation utilities:
- Y-randomization (100 repeats)
- 10-fold Q2 (cross_val_predict)
- Williams plot (leverage vs std residuals) per target
Requires: model_catboost_multi.pkl, feature_cols.txt (from train.py)
"""

import argparse
import copy
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.decomposition import PCA


def load_feature_cols(path="feature_cols.txt"):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def main(args):
    df = pd.read_csv(args.in_csv)
    y = df[['pIC50_IDO1', 'pIC50_TDO']].copy()
    X = df.drop(columns=['pIC50_IDO1', 'pIC50_TDO', 'smiles'], errors='ignore')

    feature_cols = load_feature_cols(args.feature_cols)
    X = X[feature_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = joblib.load(args.model)

    # Y-randomization
    np.random.seed(42)
    n_repeats = 100
    r2_avg, r2_ido1, r2_tdo = [], [], []
    for _ in range(n_repeats):
        y_scr = y_train.copy()
        for c in y_scr.columns:
            y_scr[c] = np.random.permutation(y_scr[c].values)

        m = copy.deepcopy(model)
        m.fit(X_train, y_scr)
        y_pred = m.predict(X_test)

        r2_per = r2_score(y_test, y_pred, multioutput='raw_values')
        r2_ido1.append(r2_per[0])
        r2_tdo.append(r2_per[1])
        r2_avg.append(r2_score(y_test, y_pred, multioutput='uniform_average'))

    print(f"Y-scramble R2 (avg of {n_repeats}): {np.mean(r2_avg):.3f}")
    print(f"  IDO1: {np.mean(r2_ido1):.3f}")
    print(f"  TDO : {np.mean(r2_tdo):.3f}")

    # Q2 with 10-fold CV on train set
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(model, X_train, y_train, cv=cv, n_jobs=-1)
    q2_avg = r2_score(y_train, y_pred_cv, multioutput='uniform_average')
    q2_raw = r2_score(y_train, y_pred_cv, multioutput='raw_values')
    print(f"Q2 (10-fold) avg: {q2_avg:.3f} | IDO1: {q2_raw[0]:.3f} | TDO: {q2_raw[1]:.3f}")

    # Williams plot using PCA(5) leverage on whole set
    X_total = pd.concat([X_train, X_test], ignore_index=True)
    y_total = pd.concat([y_train.reset_index(drop=True), y_test.reset_index(drop=True)], ignore_index=True)
    y_pred_total = pd.DataFrame(model.predict(X_total), columns=['IDO1_pred', 'TDO_pred'])

    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X_total)
    X_pca_const = np.hstack([np.ones((X_pca.shape[0], 1)), X_pca])
    H = X_pca_const @ np.linalg.pinv(X_pca_const.T @ X_pca_const) @ X_pca_const.T
    leverage = np.diag(H)

    p = X_pca.shape[1]
    n = X_pca.shape[0]
    h_star = 3 * (p + 1) / n

    for i, target in enumerate(['IDO1', 'TDO']):
        resid = y_total.iloc[:, i] - y_pred_total.iloc[:, i]
        std_resid = resid / resid.std()

        n_tr = len(X_train)
        plt.figure(figsize=(10, 6))
        plt.axhline(3, color='red', linestyle='--', label='±3 std')
        plt.axhline(-3, color='red', linestyle='--')
        plt.axvline(h_star, color='blue', linestyle='--', label=f'h*={h_star:.3f}')
        plt.scatter(leverage[:n_tr], std_resid[:n_tr], alpha=0.6, label='Train', marker='o')
        plt.scatter(leverage[n_tr:], std_resid[n_tr:], alpha=0.6, label='Test', marker='^')
        plt.xlabel("Leverage (PCA-based)")
        plt.ylabel("Standardized Residuals")
        plt.title(f"Williams Plot – {target}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"williams_{target.lower()}.png", dpi=300)

    print("[OK] Saved williams_ido1.png, williams_tdo.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", default="ido1_tdo_ecfp4.csv")
    ap.add_argument("--model", default="model_catboost_multi.pkl")
    ap.add_argument("--feature-cols", default="feature_cols.txt")
    main(ap.parse_args())
