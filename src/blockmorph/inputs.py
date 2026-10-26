"""Study areas, buildings, heights, streets and building-typology predictions.

Every area is analysed in one projected CRS (metres): either given in the
configuration or chosen automatically as the UTM zone of the study areas.
"""
from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import unary_union

from .params import Params

OSM_DROP_TYPES = {"roof", "carport", "shed", "garage", "garages", "greenhouse", "hut",
                  "tent", "ruins", "construction"}


def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------- CRS
def choose_crs(boundaries: list[gpd.GeoSeries], crs: str | int | None = "auto"):
    """Projected CRS for the analysis: the configured one, or the UTM zone
    estimated from the union of all study areas."""
    if crs not in (None, "auto", ""):
        from pyproj import CRS
        c = CRS.from_user_input(crs)
        if not c.is_projected or c.axis_info[0].unit_name not in ("metre", "meter"):
            raise ValueError(f"crs {crs}: the analysis CRS must be projected, in metres")
        return crs
    allb = gpd.GeoSeries(pd.concat([b.to_crs(4326) for b in boundaries]), crs=4326)
    return allb.estimate_utm_crs()


# ------------------------------------------------------------- boundaries
def load_boundary(path) -> gpd.GeoSeries:
    """Any vector file with polygons; all features are merged into one area."""
    g = gpd.read_file(path)
    if g.crs is None:
        raise ValueError(f"{path}: boundary has no CRS")
    geom = unary_union([x if x.is_valid else x.buffer(0) for x in g.geometry])
    return gpd.GeoSeries([geom], crs=g.crs)


# --------------------------------------------------------------- buildings
def _osm_buildings(area_wgs) -> gpd.GeoDataFrame:
    import osmnx as ox
    raw = ox.features_from_polygon(area_wgs, {"building": True})
    raw = raw[raw.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    cols = {"building": "btype", "building:levels": "levels", "height": "height_tag"}
    g = raw[[c for c in cols if c in raw.columns] + ["geometry"]].rename(columns=cols)
    for c in cols.values():
        if c not in g.columns:
            g[c] = None
    g = g.reset_index(drop=True)
    g["btype"] = g["btype"].fillna("yes").astype(str)
    g = g[~g["btype"].isin(OSM_DROP_TYPES)]
    g["floors"] = pd.to_numeric(g["levels"], errors="coerce")
    g["height_m"] = g["height_tag"].astype("string").str.extract(r"([\d.]+)")[0].astype(float)
    log(f"[inputs] {len(g)} OSM buildings")
    return g[["geometry", "floors", "height_m"]]


def _file_buildings(path) -> gpd.GeoDataFrame:
    """User footprints. Optional columns: height_m (or height), floors (or
    levels); other columns are ignored."""
    g = gpd.read_parquet(path) if str(path).endswith(".parquet") else gpd.read_file(path)
    ren = {}
    for c in g.columns:
        lc = c.lower()
        if lc in ("height", "height_m") and "height_m" not in ren.values():
            ren[c] = "height_m"
        elif lc in ("floors", "levels", "building:levels") and "floors" not in ren.values():
            ren[c] = "floors"
    g = g.rename(columns=ren)
    for c in ("height_m", "floors"):
        g[c] = pd.to_numeric(g[c], errors="coerce") if c in g.columns else np.nan
    return g[["geometry", "floors", "height_m"]]


def enrich_heights(gdf, sources, crs, p: Params):
    """Fill missing heights from polygon layers with a height attribute
    (e.g. 3D-GloBFP tiles: column 'Height'), by representative point."""
    for src in sources or []:
        path, col = (src["path"], src.get("column", "Height")) if isinstance(src, dict) \
            else (src, "Height")
        bbox = tuple(gpd.GeoSeries([gdf.geometry.union_all().envelope], crs=crs)
                     .to_crs(4326).total_bounds)
        hb = gpd.read_file(path, bbox=bbox)
        if hb.empty:
            continue
        hb = hb[[col, "geometry"]].to_crs(crs)
        pts = gpd.GeoDataFrame(geometry=gdf.geometry.representative_point(), index=gdf.index)
        j = gpd.sjoin(pts, hb, how="left", predicate="within")
        h = pd.to_numeric(j[~j.index.duplicated(keep="first")][col], errors="coerce")
        h = h.where((h >= 2.0) & (h <= p.max_height_m))
        before = gdf["height_m"].notna().mean()
        gdf["height_m"] = gdf["height_m"].fillna(h)
        log(f"[inputs] heights from {Path(path).name}: {before:.0%} -> "
            f"{gdf['height_m'].notna().mean():.0%}")
    return gdf


def prepare_buildings(source, area: gpd.GeoSeries, crs, p: Params, height_sources=None):
    """Clean footprints with heights, floors, footprint, volume and floor area.
    Height priority: source height > height layers > floors x floor height.
    The height known here is the 'measured' height; missing values are
    imputed later (missing.py)."""
    area_m = area.to_crs(crs)
    if source == "osm":
        g = _osm_buildings(area.to_crs(4326).iloc[0])
    else:
        g = _file_buildings(source)
    g = g.to_crs(crs)
    g["floors"] = g["floors"].where(g["floors"] > 0)
    g["height_m"] = g["height_m"].where((g["height_m"] >= 2.0) & (g["height_m"] <= p.max_height_m))
    g["geometry"] = g.geometry.make_valid()
    g = g[~g.geometry.is_empty & g.geometry.notna()]
    ex = g.explode(index_parts=False)
    ex["__a"] = ex.geometry.area
    g = ex.sort_values("__a").groupby(level=0).tail(1).drop(columns="__a")
    g = g[g.geometry.geom_type == "Polygon"]
    g = g[g.geometry.area >= p.min_building_area_m2]
    wkb = g.geometry.to_wkb()                       # stacked duplicate parts: tallest wins
    g = g.sort_values("height_m", na_position="first")
    agg = {c: "last" for c in g.columns}
    agg.update({"floors": "max", "height_m": "max"})
    g = gpd.GeoDataFrame(g.groupby(wkb, sort=False).agg(agg), geometry="geometry", crs=crs)
    g = g.reset_index(drop=True)
    g = enrich_heights(g, height_sources, crs, p)
    g["height_m"] = g["height_m"].fillna(g["floors"] * p.floor_height_m)
    g["floors"] = g["floors"].fillna((g["height_m"] / p.floor_height_m).round())
    g["footprint_m2"] = g.geometry.area
    g["volume_m3"] = g["footprint_m2"] * g["height_m"]
    g["floor_area_m2"] = g["footprint_m2"] * g["floors"]
    g = g[g.geometry.representative_point().within(area_m.iloc[0])].reset_index(drop=True)
    log(f"[inputs] {len(g)} buildings in the study area; height known "
        f"{g['height_m'].notna().mean():.0%}")
    return g


# ------------------------------------------------------------------ streets
def prepare_streets(source, area: gpd.GeoSeries, crs, p: Params,
                    cut_all_roads: bool = False) -> gpd.GeoDataFrame:
    """Street centrelines with an OSM-style 'highway' column. By default only
    the block-cutting classes (Params.cutting_highways) are kept; with
    cut_all_roads every mapped way cuts blocks (the study's rule for City C,
    whose OSM tags many streets as 'service')."""
    if source == "osm":
        import osmnx as ox
        G = ox.graph_from_polygon(area.to_crs(4326).iloc[0], network_type="all",
                                  retain_all=True, truncate_by_edge=True)
        e = ox.graph_to_gdfs(G, nodes=False).reset_index()
    else:
        e = gpd.read_file(source) if not str(source).endswith(".parquet") \
            else gpd.read_parquet(source)
    if "highway" not in e.columns:
        raise ValueError("streets need a 'highway' column (OSM classes)")
    e["highway"] = e["highway"].map(lambda h: h[0] if isinstance(h, (list, tuple)) else h)
    e = e[["highway", "geometry"]].to_crs(crs)
    if not cut_all_roads:
        e = e[e["highway"].isin(p.cutting_highways)]
    e = e.reset_index(drop=True)
    log(f"[inputs] {len(e)} block-cutting street segments"
        + (" (all mapped ways cut)" if cut_all_roads else ""))
    return e


# ------------------------------------------------- building-typology predictions
def load_typology(path, crs) -> gpd.GeoDataFrame:
    """Footprints labelled by the building-typology repository (schema 1.0:
    column 'class_name' with the 10 analysis classes or 'None')."""
    g = gpd.read_file(path) if not str(path).endswith(".parquet") else gpd.read_parquet(path)
    if "class_name" not in g.columns:
        raise ValueError(f"{path}: no 'class_name' column (building-typology output schema 1.0)")
    g = g[["class_name", "geometry"]].to_crs(crs)
    g["class_name"] = g["class_name"].fillna("None").astype(str)
    return g
