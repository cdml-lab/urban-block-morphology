"""Automatic draft names for block types (controlled vocabulary of the
study, read off each type's median physical profile) and a Ward ordering of
types for legends. Verbatim from the study code; names are drafts to be
reviewed - user-approved names can replace them (see docs/outputs.md)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, ward

# --------------------------------------------------- interpretation helpers
# Controlled vocabulary for cluster names (2026-10-04). A name is read off
# the cluster's MEDIAN raw profile in common physical units — metres, m²,
# GSI, shared-wall ratio, block area — so the same word means the same
# fabric in every city, set and model: a "mid-rise perimeter" in one city is
# what a "mid-rise perimeter" is in any other, and two typologies share a
# name only when they fall in the same cell of the vocabulary. Structure:
#   FAMILY  (height class × contiguity × coverage × grain)
#   · up to two QUALIFIERS from a closed list, ordered by how far the
#     cluster sits from the 13-city reference median on that character
#   · a height / density ORDINAL when two types of one model still collide
#     — never a number.
# (Replaced the "two most distinctive z-scores" naming, which produced 294
# names for 676 typologies, six spellings of "open and loose" and doubled
# phrases such as "tower-topped · tower-topped".)
NAMING_KEYS = ["height_wmean", "height_p90", "slenderness_max", "gsi",
               "shared_walls_mean", "blg_area_mean", "neighbor_dist_mean",
               "edge_building_share", "blk_area", "courtyards_per_blg",
               "road_ped_share", "n_buildings"]
# median / IQR of the block characters over the 42,470 blocks of the
# universal set (13 cities, six continents) — the yardstick for "how
# unusual is this character here"
NAMING_REFERENCE = {
    "gsi": (0.382, 0.244), "height_wmean": (11.37, 10.47),
    "blg_area_mean": (169.0, 262.0), "shared_walls_mean": (0.0, 0.107),
    "neighbor_dist_mean": (7.94, 8.55), "blk_area": (6030.0, 9109.0),
    "courtyards_per_blg": (0.0, 1.0), "road_ped_share": (0.033, 0.508),
    "edge_building_share": (0.643, 0.613), "n_buildings": (11.0, 19.0),
}
# ordinal words per character, for 2 / 3 / 4 colliding types of one model
_ORDINALS = {
    "height_wmean": {2: ["lower", "taller"],
                     3: ["lowest", "medium height", "tallest"],
                     4: ["lowest", "lower", "taller", "tallest"]},
    "gsi": {2: ["sparser", "denser"],
            3: ["sparsest", "medium density", "densest"],
            4: ["sparsest", "sparser", "denser", "densest"]},
    "blg_area_mean": {2: ["finer", "coarser"],
                      3: ["finest", "medium grain", "coarsest"],
                      4: ["finest", "finer", "coarser", "coarsest"]},
    "n_buildings": {2: ["fewer buildings", "more buildings"],
                    3: ["fewest buildings", "medium count", "most buildings"],
                    4: ["fewest buildings", "fewer buildings",
                        "more buildings", "most buildings"]},
    "blk_area": {2: ["smaller blocks", "larger blocks"],
                 3: ["smallest blocks", "medium blocks", "largest blocks"],
                 4: ["smallest blocks", "smaller blocks", "larger blocks",
                     "largest blocks"]},
    "neighbor_dist_mean": {2: ["tighter", "more spaced"],
                           3: ["tightest", "medium spacing", "most spaced"],
                           4: ["tightest", "tighter", "more spaced",
                               "most spaced"]},
}
_LOG_KEYS = {"blg_area_mean", "n_buildings", "blk_area"}
# which qualifier words speak about which character
_QUALIFIER_KEYS = {
    "blg_area_mean": {"fine grain", "coarse grain"},
    "gsi": {"open", "dense"},
    "neighbor_dist_mean": {"tight", "spaced"},
    "blk_area": {"small blocks", "superblocks"},
    "n_buildings": {"single building"},
}


def _spread(vals: list[float], key: str) -> float:
    v = np.array([x for x in vals if np.isfinite(x)], dtype=float)
    if len(v) < 2:
        return 0.0
    if key in _LOG_KEYS:
        v = np.log(np.maximum(v, 1.0))
        return float(v.max() - v.min()) / 0.9
    med, iqr = NAMING_REFERENCE[key]
    return float(v.max() - v.min()) / (iqr or 1.0)


def _tiers(members, prof, key, k):
    """Rank the members on `key` and map them to the k-word table."""
    order = sorted(members, key=lambda c: (prof[c][key]
                                           if np.isfinite(prof[c][key])
                                           else -1))
    words = _ORDINALS[key][k]
    if len(order) == k:
        return dict(zip(order, words))
    # more members than words: tiers by rank
    out = {}
    for i, c in enumerate(order):
        out[c] = words[min(k - 1, int(i * k / len(order)))]
    return out


def _family(p: dict) -> str:
    """Form family from the median profile (NaN compares False, so a
    missing character falls through to the plainer branch)."""
    h, g, sw, fp = (p["height_wmean"], p["gsi"], p["shared_walls_mean"],
                    p["blg_area_mean"])
    edge, p90, sl = (p["edge_building_share"], p["height_p90"],
                     p["slenderness_max"])
    if g < 0.10:
        return "open land"
    if h >= 30 or (p90 >= 45 and sl >= 3):
        return "towers"
    if h >= 20:
        return ("high-rise perimeter" if (sw >= 0.25 and g >= 0.35)
                else "high-rise slabs")
    if h >= 9:
        if sw >= 0.25:
            return ("mid-rise perimeter" if (g >= 0.35 and edge >= 0.6)
                    else "mid-rise rows")
        return "mid-rise estate" if g < 0.30 else "mid-rise freestanding"
    if fp >= 700 and g < 0.45 and sw < 0.25:
        return "low-rise halls"
    if g >= 0.40 and fp < 150:
        return "low-rise compact"
    if sw >= 0.25:
        return "low-rise rows"
    return "detached houses" if fp < 450 else "low-rise large footprints"


_QUALIFIERS = (
    ("blg_area_mean", lambda v: "fine grain" if v < 110
                      else ("coarse grain" if v > 450 else None)),
    ("gsi", lambda v: "open" if v < 0.20 else ("dense" if v > 0.50 else None)),
    ("neighbor_dist_mean", lambda v: "tight" if v < 5.5
                           else ("spaced" if v > 14 else None)),
    ("blk_area", lambda v: "small blocks" if v < 3000
                 else ("superblocks" if v > 20000 else None)),
    ("courtyards_per_blg", lambda v: "courtyards" if v > 0.3 else None),
    ("edge_building_share", lambda v: "set back" if v < 0.3 else None),
    ("n_buildings", lambda v: "single building" if v <= 1.5 else None),
)
# qualifiers a family already states (never repeated after it)
_IMPLIED = {
    "low-rise compact": {"dense", "fine grain", "tight"},
    "mid-rise perimeter": {"dense"}, "high-rise perimeter": {"dense"},
    "mid-rise estate": {"open", "spaced"},
    "low-rise halls": {"coarse grain"},
    "low-rise large footprints": {"coarse grain"},
}


def _qualifiers(p: dict, fam: str) -> list[str]:
    """Qualifier words, most distinctive first (robust z from the
    reference median; log scale for areas; the street-interface word is
    a last resort)."""
    if fam == "open land":
        return []
    cands = []
    for key, fn in _QUALIFIERS:
        v = p.get(key, np.nan)
        if not np.isfinite(v):
            continue
        w = fn(v)
        if w is None or w in _IMPLIED.get(fam, set()):
            continue
        med, iqr = NAMING_REFERENCE[key]
        if key in ("blg_area_mean", "blk_area"):
            z = abs(np.log(max(v, 1.0)) - np.log(max(med, 1.0))) / 0.9
        else:
            z = abs(v - med) / (iqr or 1.0)
        if w == "single building":
            z = 2.5                     # always worth saying
        cands.append((z, w))
    cands.sort(reverse=True)
    return [w for _, w in cands]


def name_clusters(features: pd.DataFrame, labels: np.ndarray) -> dict[int, str]:
    """Controlled-vocabulary names (see the module notes above): family ·
    first qualifier · (second qualifier or ordinal). Unique within a model
    by construction; shared across models only when the profiles fall in
    the same cell."""
    labels = np.asarray(labels)
    ids = sorted(set(labels.tolist()))
    prof, fams, quals = {}, {}, {}
    for c in ids:
        sel = features[labels == c]
        p = {k: (float(np.nanmedian(sel[k])) if k in sel.columns
                 and sel[k].notna().any() else np.nan) for k in NAMING_KEYS}
        prof[c] = p
        fams[c] = _family(p)
        quals[c] = _qualifiers(p, fams[c])
    depth = {c: 1 for c in ids}

    def build(c):
        return " · ".join([fams[c]] + quals[c][:depth[c]])
    names = {c: build(c) for c in ids}
    # first collision pass: a second qualifier where one exists
    groups: dict[str, list] = {}
    for c in ids:
        groups.setdefault(names[c], []).append(c)
    for members in groups.values():
        if len(members) > 1:
            for c in members:
                if len(quals[c]) >= 2:
                    depth[c] = 2
                    names[c] = build(c)
    # second pass: an ordinal replaces the second qualifier (three parts),
    # on the character that spreads the colliding group most; five or
    # more collisions take a two-character grid
    groups = {}
    for c in ids:
        groups.setdefault(names[c], []).append(c)
    for n, members in groups.items():
        if len(members) < 2:
            continue
        base = " · ".join(n.split(" · ")[:2])
        # never grade a group on the character its qualifier already
        # states ("spaced · more spaced")
        said = {w for w in n.split(" · ")[1:]}
        skip = {k for k, ws in _QUALIFIER_KEYS.items() if said & ws}
        spreads = sorted(((_spread([prof[c][k] for c in members], k), k)
                          for k in _ORDINALS if k not in skip),
                         reverse=True)
        k1 = spreads[0][1]
        m = len(members)
        if m <= 4:
            tiers = _tiers(members, prof, k1, m)
            for c in members:
                names[c] = f"{base} · {tiers[c]}"
        else:
            k2 = spreads[1][1]
            t1 = _tiers(members, prof, k1, 3)
            t2 = _tiers(members, prof, k2, 2)
            for c in members:
                names[c] = f"{base} · {t1[c]} · {t2[c]}"
    # anything still equal (identical profiles): a variant number
    groups = {}
    for c in ids:
        groups.setdefault(names[c], []).append(c)
    for n, members in groups.items():
        if len(members) > 1:
            for i, c in enumerate(members):
                names[c] = f"{n} · variant {i + 1}"
    return names


def order_clusters(features: pd.DataFrame, labels: np.ndarray) -> list[int]:
    """Ward-order cluster ids so similar types sit together in legends."""
    ids = sorted(set(labels))
    if len(ids) < 3:
        return ids
    x = features.fillna(features.median())
    x = (x - x.mean()) / x.std().replace(0, 1.0)
    profiles = np.vstack([x[labels == c].mean().fillna(0.0).values for c in ids])
    return [ids[i] for i in leaves_list(ward(profiles))]
