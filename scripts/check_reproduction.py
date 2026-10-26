"""Compare a run of configs/study_reproduction.yaml with the published study
results (data package `study-results`).

    python scripts/check_reproduction.py <run output_dir> <data package dir>

Tolerances (docs/reproduction.md):
  - block keys identical;
  - K and rule/fallback status identical per model;
  - type ids identical (ARI = 1.0 and same integer ids);
  - descriptors within 1e-4 of each column's range. The enclosed
    tessellation (car_mean) differs at ~1e-5 between runs and platforms.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

REL_TOL = 1e-4


def main(run_dir, pkg_dir):
    run, pkg = Path(run_dir), Path(pkg_dir)
    ok = True
    mine = pd.read_csv(run / "descriptors.csv", index_col="block_key")
    ref = pd.read_csv(pkg / "descriptors_treated.csv", index_col="block_key")
    ref = ref[mine.columns]
    if set(mine.index) != set(ref.index):
        print(f"block keys differ: {len(mine)} vs {len(ref)}"); return 1
    mine = mine.loc[ref.index]
    rng = (ref.max() - ref.min()).replace(0, 1)
    d = ((mine - ref).abs() / rng).max()
    nan = int((mine.isna() != ref.isna()).sum().sum())
    print(f"descriptors: {len(ref)} blocks, max diff {d.max():.2e} of range ({d.idxmax()}), "
          f"NaN pattern mismatches {nan}")
    ok &= bool(d.max() <= REL_TOL and nan == 0)
    a = pd.read_csv(run / "assignments.csv", index_col="block_key").loc[ref.index]
    r = pd.read_csv(pkg / "assignments.csv", index_col="block_key").loc[ref.index]
    ks = pd.read_csv(pkg / "models.csv").set_index("model")
    for model in ks.index:
        rep = json.loads((run / "models" / model / "report.json").read_text())
        if model == "pooled":
            x, y = a["pooled_type"], r["pooled_type"]
        else:
            m = a["area"] == model
            x, y = a.loc[m, "local_type"], r.loc[m, "local_type"]
        ari = adjusted_rand_score(y, x)
        same = bool((x.values == y.values).all())
        good = (rep["selected_k"] == ks.at[model, "k"] and rep["status"] == ks.at[model, "status"]
                and same)
        ok &= good
        print(f"{model:>7}: K {rep['selected_k']} ({rep['status']}) vs {ks.at[model, 'k']} "
              f"({ks.at[model, 'status']}), ARI {ari:.4f}, identical ids {same}")
    print("REPRODUCTION_OK" if ok else "REPRODUCTION_DIFFERS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
