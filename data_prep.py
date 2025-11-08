#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data preparation for dual-target IDO1/TDO QSAR.
- Loads ChEMBL CSVs for IDO1 (CHEMBL4685) and TDO (CHEMBL2140)
- Cleans by keeping '=' standard_relation
- Aggregates duplicate SMILES by mean standard_value
- Computes pIC50 and merges into a single CSV
Output: dual_target_qsar.csv
"""

import argparse
import numpy as np
import pandas as pd


def compute_pIC50(nm_val: float) -> float:
    """Convert IC50 in nM to pIC50."""
    return -np.log10(float(nm_val) * 1e-9)


def main(args):
    ido = pd.read_csv(args.ido_csv)
    tdo = pd.read_csv(args.tdo_csv)

    # Clean relation
    for df in (ido, tdo):
        df.loc[:, 'standard_relation'] = (
            df['standard_relation'].astype(str).str.replace("'", "").str.strip()
        )

    # Keep only '='
    ido_eq = ido[ido['standard_relation'] == '='].copy()
    tdo_eq = tdo[tdo['standard_relation'] == '='].copy()

    # Convert values to float
    ido_eq["standard_value"] = ido_eq["standard_value"].astype(float)
    tdo_eq["standard_value"] = tdo_eq["standard_value"].astype(float)

    # Average duplicates by SMILES
    ido_u = ido_eq.groupby("smiles", as_index=False)["standard_value"].mean()
    tdo_u = tdo_eq.groupby("smiles", as_index=False)["standard_value"].mean()

    # pIC50
    ido_u["pIC50_IDO1"] = ido_u["standard_value"].apply(compute_pIC50)
    tdo_u["pIC50_TDO"] = tdo_u["standard_value"].apply(compute_pIC50)

    # Merge
    merged = pd.merge(ido_u[["smiles", "pIC50_IDO1"]],
                      tdo_u[["smiles", "pIC50_TDO"]],
                      on="smiles", how="inner")

    merged.to_csv(args.out_csv, index=False)
    print(f"[OK] Saved {args.out_csv} with {len(merged)} rows.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ido-csv", default="ido_CHEMBL4685.csv")
    p.add_argument("--tdo-csv", default="tdo_CHEMBL2140.csv")
    p.add_argument("--out-csv", default="dual_target_qsar.csv")
    main(p.parse_args())
