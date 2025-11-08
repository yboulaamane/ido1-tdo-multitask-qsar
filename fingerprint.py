#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate ECFP4 fingerprints (2048 bits).
Input: dual_target_qsar_standardized.csv
Output: ido1_tdo_ecfp4.csv
"""

import argparse
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


def main(args):
    df = pd.read_csv(args.in_csv)
    tqdm.pandas()
    df['ecfp4'] = df['smiles'].progress_apply(get_ecfp4_array)

    df = df[df['ecfp4'].notnull()].reset_index(drop=True)
    fp_df = pd.DataFrame(df['ecfp4'].to_list(), columns=[f'fp_{i}' for i in range(2048)])
    out = pd.concat([df.drop(columns=['ecfp4']), fp_df], axis=1)
    out.to_csv(args.out_csv, index=False)
    print(f"[OK] Saved {args.out_csv} with shape {out.shape}.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", default="dual_target_qsar_standardized.csv")
    ap.add_argument("--out-csv", default="ido1_tdo_ecfp4.csv")
    main(ap.parse_args())
