#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SMILES standardization:
- Salt stripping, normalization, uncharging, tautomer canonicalization
Input: dual_target_qsar.csv
Output: dual_target_qsar_standardized.csv
"""

import argparse
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog('rdApp.*')

remover = SaltRemover.SaltRemover()
normalizer = rdMolStandardize.Normalizer()
uncharger = rdMolStandardize.Uncharger()
tautomer_enumerator = rdMolStandardize.TautomerEnumerator()


def standardize_smiles(smiles: str):
    if pd.isnull(smiles):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = remover.StripMol(mol)
    mol = normalizer.normalize(mol)
    mol = uncharger.uncharge(mol)
    mol = tautomer_enumerator.Canonicalize(mol)
    return Chem.MolToSmiles(mol, canonical=True)


def main(args):
    df = pd.read_csv(args.in_csv)
    df["smiles"] = df["smiles"].apply(standardize_smiles)
    df = df.dropna(subset=["smiles"]).drop_duplicates(subset=["smiles"]).reset_index(drop=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] Saved {args.out_csv} with {len(df)} standardized molecules.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", default="dual_target_qsar.csv")
    ap.add_argument("--out-csv", default="dual_target_qsar_standardized.csv")
    main(ap.parse_args())
