"""Physical type profiles and representative blocks.

Profiles are medians across a type's blocks. Heights are those available
BEFORE imputation ('measured'): the footprint-weighted mean of the block's
buildings with a height, FSI from their storeys; `heights_measured` is the
mean share of a block's buildings with such a height.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PROFILE = [  # column, label, unit, log scale for similarity
    ("blk_area_ha", "block area", "ha", True),
    ("n_buildings", "buildings", "", True),
    ("gsi", "coverage (GSI)", "", False),
    ("fsi_measured", "FSI", "", False),
    ("height_measured_m", "height", "m", False),
    ("height_p90_m", "height p90", "m", False),
    ("footprint_mean_m2", "mean footprint", "m²", True),
    ("spacing_m", "spacing to neighbours", "m", False),
    ("shared_walls", "shared-wall ratio", "", False),
    ("edge_share", "street-edge share", "", False),
]


def block_stats(chars, feats, block_keys, floor_height_m=3.2) -> pd.DataFrame:
    """Per-block physical statistics with pre-imputation heights. chars: the
    buildings with height_measured_m, floors, footprint_m2, block_key."""
    b = chars[chars["block_key"].notna()].copy()
    k = b[b["height_measured_m"].notna()]
    fa = k["footprint_m2"]
    hw = (k["height_measured_m"] * fa).groupby(k["block_key"]).sum() / fa.groupby(k["block_key"]).sum()
    st = k["floors_measured"].fillna((k["height_measured_m"] / floor_height_m).round())
    sw = (st * fa).groupby(k["block_key"]).sum() / fa.groupby(k["block_key"]).sum()
    cov = k.groupby("block_key").size().reindex(b.groupby("block_key").size().index).fillna(0) \
        / b.groupby("block_key").size()
    p90 = k.groupby("block_key")["height_measured_m"].quantile(0.9)
    f = feats.loc[block_keys]
    out = pd.DataFrame(index=block_keys)
    out["blk_area_ha"] = f["blk_area"].values / 1e4
    out["n_buildings"] = f["n_buildings"].values
    out["gsi"] = f["gsi"].values
    out["fsi_measured"] = (f["gsi"] * sw.reindex(block_keys)).values
    out["height_measured_m"] = hw.reindex(block_keys).values
    out["height_p90_m"] = p90.reindex(block_keys).values
    out["footprint_mean_m2"] = f["blg_area_mean"].values
    out["spacing_m"] = f["neighbor_dist_mean"].values
    out["shared_walls"] = f["shared_walls_mean"].values
    out["edge_share"] = f["edge_building_share"].values
    out["heights_measured"] = cov.reindex(block_keys).fillna(0).values
    return out


def type_profiles(stats, labels, names, area_of) -> pd.DataFrame:
    rows = []
    for t in sorted(np.unique(labels)):
        m = labels == t
        s = stats[m]
        r = {"type": int(t), "name": names.get(int(t), f"type {t}"), "blocks": int(m.sum())}
        for a in sorted(set(area_of)):
            r[f"blocks_{a}"] = int(((np.asarray(area_of) == a) & m).sum())
        for col, *_ in PROFILE:
            r[col] = float(s[col].median())
        r["heights_measured"] = float(s["heights_measured"].mean())
        rows.append(r)
    return pd.DataFrame(rows)


def representatives(stats, labels, area_of, n=3, min_measured=0.8, min_share=0.10):
    """Blocks nearest each type's median profile (robust z), among blocks
    with >= min_measured of heights before imputation and an area inside the
    type's 10-90 % range; one per area holding >= min_share of the type."""
    cols = [c for c, *_ in PROFILE]
    z = stats[cols].astype(float).copy()
    for col, _, _, lg in PROFILE:
        v = np.log10(z[col].clip(lower=1e-6)) if lg else z[col]
        iqr = v.quantile(.75) - v.quantile(.25)
        z[col] = (v - v.median()) / (iqr if iqr > 0 else 1.0)
    z = z.fillna(0)
    area_of = pd.Series(np.asarray(area_of), index=stats.index)
    out = {}
    for t in sorted(np.unique(labels)):
        idx = stats.index[labels == t]
        d = ((z.loc[idx] - z.loc[idx].median()) ** 2).sum(axis=1) ** 0.5
        a = stats.loc[idx, "blk_area_ha"]
        ok = (stats.loc[idx, "heights_measured"] >= min_measured) & \
            a.between(a.quantile(.1), a.quantile(.9))
        cand = d[ok].sort_values() if ok.sum() >= n else d.sort_values()
        pct = d.rank(pct=True)
        counts = area_of.loc[idx].value_counts()
        picks = []
        for ar in counts.index:
            if counts[ar] / len(idx) >= min_share and len(picks) < n:
                cc = cand[[i for i in cand.index if area_of[i] == ar]]
                if len(cc) and pct[cc.index[0]] <= 0.25:
                    picks.append(cc.index[0])
        for i in cand.index:
            if len(picks) >= n:
                break
            if i not in picks:
                picks.append(i)
        out[int(t)] = picks
    return out
