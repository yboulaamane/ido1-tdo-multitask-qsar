#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Virtual screening on COCONUT (or any CSV with SMILES + basic props).
- Filters by RO5 + rotatable bonds <= 10
- Generates ECFP4 aligned to training feature columns
- Predicts pIC50 for IDO1/TDO using trained model
- Applicability Domain via Mahalanobis distance in PCA(20) space (params from train.py)
Outputs:
  - coconut_predictions.csv
  - coconut_with_predictions_ad.csv
  - coconut_filtered_active_inside_ad.csv (pIC50>6 both & inside AD)
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm


def get_ecfp4_array(smiles: str, nBits: int = 2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nBits)
        return list(fp)
    return None


def mahalanobis_dist(X, mean_vec, inv_cov):
    diffs = X - mean_vec
    # (x-mu)^T * inv_cov * (x-mu)
    return np.sqrt(np.einsum("ij,jk,ik->i", diffs, inv_cov, diffs))


def main(args):
    model = joblib.load(args.model)
    ad = joblib.load(args.ad_params)
    feature_cols = ad["feature_cols"]

    df = pd.read_csv(args.in_csv)
    # Simple RO5 + RB filter if columns exist
    if set(["molecular_weight", "alogp", "hydrogen_bond_donors_lipinski",
            "hydrogen_bond_acceptors_lipinski", "rotatable_bond_count"]).issubset(df.columns):
        df = df[
            (df['molecular_weight'] <= 500) &
            (df['alogp'] <= 5) &
            (df['hydrogen_bond_donors_lipinski'] <= 5) &
            (df['hydrogen_bond_acceptors_lipinski'] <= 10) &
            (df['rotatable_bond_count'] <= 10)
        ].copy()

    # Fingerprints
    tqdm.pandas()
    df['ecfp4'] = df['SMILES'].progress_apply(get_ecfp4_array)
    df = df[df['ecfp4'].notnull()].reset_index(drop=True)
    fp_df = pd.DataFrame(df['ecfp4'].to_list(), columns=[f'fp_{i}' for i in range(2048)])
    df = pd.concat([df.drop(columns=['ecfp4']), fp_df], axis=1)

    # Align to training feature columns (missing -> 0)
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0
    X = df[feature_cols].copy()

    # Predictions
    y_pred = model.predict(X)
    df['Predicted_pIC50_IDO1'] = y_pred[:, 0]
    df['Predicted_pIC50_TDO'] = y_pred[:, 1]

    # AD (PCA space saved in ad_params)
    pca = ad["pca"]
    mean_vec = ad["mean_vec"]
    inv_cov = ad["inv_cov"]
    threshold = ad["threshold"]

    X_pca = pca.transform(X)
    md = mahalanobis_dist(X_pca, mean_vec, inv_cov)

    df['Mahalanobis'] = md
    df['Inside_AD'] = md <= threshold

    # Save basic predictions without fp columns
    out_basic_cols = [c for c in df.columns if not c.startswith('fp_')]
    df[out_basic_cols].to_csv("coconut_with_predictions_ad.csv", index=False)

    # Filter: inside AD and good pIC50 on both
    filt = df[(df['Inside_AD']) & (df['Predicted_pIC50_IDO1'] > 6) & (df['Predicted_pIC50_TDO'] > 6)]
    filt[out_basic_cols].to_csv("coconut_filtered_active_inside_ad.csv", index=False)

    print("[OK] Saved coconut_with_predictions_ad.csv and coconut_filtered_active_inside_ad.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", default="coconut_csv_lite-07-2025.csv")
    ap.add_argument("--model", default="model_catboost_multi.pkl")
    ap.add_argument("--ad-params", default="ad_params.joblib")
    main(ap.parse_args())
