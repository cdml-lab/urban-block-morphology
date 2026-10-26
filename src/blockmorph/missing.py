"""Explicit missing-data treatment (study procedure).

Heights: a building without a height takes (1) the median known height of
the buildings in its own block, else (2) the median height of its k nearest
buildings with a known height (k = 10). Floor area and volume are recomputed
from the imputed height (floors: known floors, else round(height / 3.2)).
The pre-imputation ('measured') height and a flag are kept for reporting.

Typology shares: blocks without any classified building keep NaN shares
(see descriptors.typology_shares); they are filled with the column median
only inside the preprocessing for PCA, never with zeros.
"""
from __future__ import annotations

import numpy as np

from .params import Params


def impute_heights(chars, p: Params):
    from scipy.spatial import cKDTree
    c = chars.copy()
    known = c["height_m"].notna()
    c["height_measured_m"] = c["height_m"]
    c["height_imputed"] = ~known
    h = c["height_m"].copy()
    inblk = c["block_id"].notna()
    med = c.loc[known & inblk].groupby(c.loc[known & inblk, "block_id"].astype(int))["height_m"].median()
    r1 = (~known) & inblk
    h.loc[r1] = c.loc[r1, "block_id"].astype(int).map(med).values
    still = h.isna()
    stats = {"buildings": int(len(c)), "known": int(known.sum()),
             "imputed_block_median": int(h.notna().sum() - known.sum()),
             "imputed_knn": int(still.sum())}
    if still.any() and known.any():
        tree = cKDTree(np.c_[c.loc[known].geometry.centroid.x, c.loc[known].geometry.centroid.y])
        _, idx = tree.query(np.c_[c.loc[still].geometry.centroid.x,
                                  c.loc[still].geometry.centroid.y],
                            k=min(p.impute_knn, int(known.sum())))
        idx = np.atleast_2d(idx)
        h.loc[still] = np.median(c.loc[known, "height_m"].values[idx], axis=1)
    floors = c["floors"].fillna((h / p.floor_height_m).round())
    c["height_m"] = h
    c["volume_m3"] = c["footprint_m2"] * h
    c["floor_area_m2"] = c["footprint_m2"] * floors
    return c, stats
