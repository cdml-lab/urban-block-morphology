"""Blocks: street enclosures holding buildings, and the enclosed tessellation."""
from __future__ import annotations

import geopandas as gpd
import momepy
import numpy as np

from .inputs import log
from .params import Params


def enclosures(streets: gpd.GeoDataFrame, area: gpd.GeoSeries) -> gpd.GeoDataFrame:
    enc = momepy.enclosures(streets, limit=area, clip=True)
    enc = enc[enc.geometry.area > 0].reset_index(drop=True)
    enc["eID"] = enc.index
    return enc


def make_blocks(buildings, enc, p: Params):
    """An enclosure is a block when its building coverage is at least
    min_block_gsi, its area at least min_block_area_m2 and it holds a
    building (by representative point). Returns (blocks, open_space, buildings
    with block_id)."""
    pts = buildings[["geometry"]].copy()
    pts["geometry"] = buildings.geometry.representative_point()
    j = gpd.sjoin(pts, enc[["eID", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    b = buildings.copy()
    b["eID"] = j["eID"]
    fp = b.dropna(subset=["eID"]).groupby("eID")["footprint_m2"].sum()
    e = enc.set_index("eID")
    e["footprint_m2"] = fp.reindex(e.index).fillna(0.0)
    e["area_m2"] = e.geometry.area
    e["gsi"] = e["footprint_m2"] / e["area_m2"]
    is_block = (e["gsi"] >= p.min_block_gsi) & (e["area_m2"] >= p.min_block_area_m2) \
        & (e["footprint_m2"] > 0)
    blocks = e[is_block].copy()
    blocks["block_id"] = np.arange(len(blocks))
    b["block_id"] = b["eID"].map(blocks["block_id"])
    log(f"[blocks] {len(blocks)} blocks; {int(b['block_id'].notna().sum())} of {len(b)} "
        "buildings inside blocks")
    return blocks.reset_index(), e[~is_block].reset_index(), b


def tessellate(buildings, enc) -> gpd.GeoDataFrame:
    """Enclosed tessellation, one cell per building (largest part kept)."""
    t = momepy.enclosed_tessellation(buildings, enc.geometry, n_jobs=-1)
    t = t[t.index >= 0]
    t["__a"] = t.geometry.area
    return t.sort_values("__a").groupby(level=0).tail(1).drop(columns="__a")
