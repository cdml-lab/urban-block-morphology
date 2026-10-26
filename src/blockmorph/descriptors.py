"""Block descriptors (momepy 1.0).

Full model (the study's 57 descriptors) = 46 morphometric and contextual
descriptors + 10 building-typology shares + 1 classified-building count.
Morphometric-only model (`descriptors: morphometric46`) = the 46 alone; it is
a DIFFERENT model from the manuscript's and is labelled as such in every
output. Typology shares are never filled with zeros for missing predictions:
a block without any classified building has NaN shares (imputed by the
preprocessing median, as in the study).
"""
from __future__ import annotations

import geopandas as gpd
import momepy
import numpy as np
import pandas as pd
from libpysal.graph import Graph

from .inputs import log
from .params import Params

TYPOLOGY_CLASSES = ["Carpet Houses", "Geometric Building", "Geometric Shape", "H Shape",
                    "L Shape", "Long Rectangle", "Private House_A", "Private House_B",
                    "Rectangle Building", "U Shape"]
TYPOLOGY_COLUMNS = ["rcnn_" + c.lower().replace(" ", "_") for c in TYPOLOGY_CLASSES]
MORPHO_COLUMNS = [
    "blk_area", "blk_perimeter", "blk_compactness", "blk_eri", "blk_orientation",
    "n_buildings", "blg_density_ha", "gsi", "fsi", "height_mean", "height_std", "height_max",
    "volume_sum", "volume_density", "height_wmean", "height_p90", "height_cv",
    "slenderness_mean", "slenderness_max", "osr", "layers_mean", "bulk_density",
    "blg_area_mean", "blg_area_std", "elongation_mean", "eri_mean", "car_mean",
    "dist_to_street_mean", "shared_walls_mean", "courtyards_per_blg", "neighbor_dist_mean",
    "alignment_mean", "street_alignment_mean", "adjacency_mean", "edge_building_share",
    "dist_to_edge_norm_mean", "road_border_density", "road_ped_share", "road_frontage_segs",
    "road_seg_len_mean", "ctx_gsi", "ctx_fsi", "ctx_height_mean", "ctx_blg_area_mean",
    "ctx_shared_walls_mean", "ctx_blg_density_ha"]
FULL57 = (MORPHO_COLUMNS[:36] + TYPOLOGY_COLUMNS + ["rcnn_labelled"] + MORPHO_COLUMNS[36:])
PED_TYPES = {"pedestrian", "footway", "living_street"}
assert len(MORPHO_COLUMNS) == 46 and len(FULL57) == 57


def building_characters(b, tess, streets, blocks, p: Params) -> gpd.GeoDataFrame:
    b = b.copy()
    geom = b.geometry
    b["blg_area"] = geom.area
    b["blg_perimeter"] = geom.length
    b["blg_elongation"] = momepy.elongation(geom)
    b["blg_eri"] = momepy.equivalent_rectangular_index(geom)
    b["blg_orientation"] = momepy.orientation(geom)
    contig = Graph.build_fuzzy_contiguity(b, buffer=0.25)
    knn = Graph.build_knn(geom.centroid, k=5, coplanar="clique")
    b["shared_walls_ratio"] = (momepy.shared_walls(b) / b["blg_perimeter"]).clip(0, 1)
    b["courtyards"] = momepy.courtyards(b, contig)
    b["neighbor_distance"] = momepy.neighbor_distance(b, knn)
    b["alignment"] = momepy.alignment(b["blg_orientation"], knn)
    b["building_adjacency"] = momepy.building_adjacency(contig, knn)
    sidx = momepy.get_nearest_street(b, streets, max_distance=p.street_search_m)
    sor = momepy.orientation(streets.geometry)
    m = sidx.notna()
    sa = pd.Series(np.nan, index=b.index)
    sa[m] = momepy.street_alignment(b.loc[m, "blg_orientation"], sor, sidx[m]).values
    b["street_alignment"] = sa
    dist = pd.Series(np.nan, index=b.index)
    near = streets.geometry.reindex(sidx[m].astype(int)).set_axis(b.index[m])
    dist[m] = geom[m].distance(near)
    b["dist_to_street"] = dist
    cell = tess.geometry.reindex(b.index)
    b["cell_area"] = cell.area
    b["car"] = (b["blg_area"] / cell.area).clip(0, 1)
    co = momepy.orientation(gpd.GeoSeries(cell[cell.notna()]))
    b["cell_alignment"] = momepy.cell_alignment(b["blg_orientation"], co.reindex(b.index))
    blk = blocks.set_index("block_id").geometry
    cen = geom.centroid
    d_edge = pd.Series(np.nan, index=b.index)
    norm = pd.Series(np.nan, index=b.index)
    hb = b["block_id"].notna()
    bid = b.loc[hb, "block_id"].astype(int)
    d_edge[hb] = cen[hb].distance(blk.boundary.reindex(bid).set_axis(b.index[hb]))
    norm[hb] = d_edge[hb] / np.sqrt(blk.area / np.pi).reindex(bid).set_axis(b.index[hb])
    b["dist_to_block_edge"] = d_edge
    b["dist_to_edge_norm"] = norm
    b["is_edge_building"] = (d_edge <= p.edge_band_m).astype(float)
    return b


def _agg(s, by, how):
    return s.groupby(by).agg(how)


def _roads(blocks, streets):
    ring = blocks[["block_id"]].copy()
    ring["geometry"] = blocks.geometry.boundary.buffer(12)
    ring = gpd.GeoDataFrame(ring, geometry="geometry", crs=blocks.crs)
    inter = gpd.overlay(streets[["highway", "geometry"]], ring, how="intersection",
                        keep_geom_type=True)
    inter["len"] = inter.geometry.length
    inter = inter[inter["len"] > 1.0]
    out = pd.DataFrame(index=blocks["block_id"])
    perim = blocks.set_index("block_id").geometry.length
    tot = inter.groupby("block_id")["len"].sum()
    ped = inter[inter["highway"].isin(PED_TYPES)].groupby("block_id")["len"].sum()
    out["road_border_density"] = (tot / perim).reindex(out.index)
    out["road_ped_share"] = (ped / tot).reindex(out.index)
    out["road_frontage_segs"] = (inter.groupby("block_id").size() * 100 / perim).reindex(out.index)
    out["road_seg_len_mean"] = inter.groupby("block_id")["len"].mean().reindex(out.index)
    return out.fillna(0.0)


def typology_shares(blocks, typology) -> pd.DataFrame:
    """Share of each of the 10 classes among a block's classified buildings
    (placed by representative point) and their count. Blocks without any
    classified building: count 0, shares NaN (missing, not zero)."""
    out = pd.DataFrame(index=blocks["block_id"])
    lab = typology[typology["class_name"].isin(TYPOLOGY_CLASSES)].copy()
    lab["geometry"] = lab.geometry.representative_point()
    j = gpd.sjoin(lab[["class_name", "geometry"]], blocks[["block_id", "geometry"]],
                  how="inner", predicate="within")
    ct = pd.crosstab(j["block_id"], j["class_name"]).reindex(columns=TYPOLOGY_CLASSES,
                                                             fill_value=0)
    shares = ct.div(ct.sum(axis=1), axis=0)
    shares.columns = TYPOLOGY_COLUMNS
    out = out.join(shares)
    out["rcnn_labelled"] = ct.sum(axis=1).reindex(out.index).fillna(0.0)
    has = out["rcnn_labelled"] > 0
    out.loc[has, TYPOLOGY_COLUMNS] = out.loc[has, TYPOLOGY_COLUMNS].fillna(0.0)
    out.loc[~has, TYPOLOGY_COLUMNS] = np.nan
    log(f"[descriptors] typology: {int(out['rcnn_labelled'].sum())} classified buildings; "
        f"{int((~has).sum())} blocks without any (shares missing)")
    return out


def block_descriptors(chars, blocks, streets, typology, p: Params) -> pd.DataFrame:
    b = chars[chars["block_id"].notna()].copy()
    b["block_id"] = b["block_id"].astype(int)
    gid = b["block_id"]
    blk = blocks.set_index("block_id")
    f = pd.DataFrame(index=blk.index)
    f["blk_area"] = blk.geometry.area
    f["blk_perimeter"] = blk.geometry.length
    f["blk_compactness"] = momepy.circular_compactness(blk.geometry)
    f["blk_eri"] = momepy.equivalent_rectangular_index(blk.geometry)
    f["blk_orientation"] = momepy.orientation(blk.geometry)
    f["n_buildings"] = gid.value_counts()
    f["blg_density_ha"] = f["n_buildings"] / (f["blk_area"] / 10_000)
    f["gsi"] = _agg(b["blg_area"], gid, "sum") / f["blk_area"]
    f["fsi"] = _agg(b["floor_area_m2"].fillna(b["blg_area"]), gid, "sum") / f["blk_area"]
    f["height_mean"] = _agg(b["height_m"], gid, "mean")
    f["height_std"] = _agg(b["height_m"], gid, "std")
    f["height_max"] = _agg(b["height_m"], gid, "max")
    f["volume_sum"] = _agg(b["volume_m3"], gid, "sum")
    f["volume_density"] = f["volume_sum"] / f["blk_area"]
    f["height_wmean"] = _agg(b["height_m"] * b["blg_area"], gid, "sum") / _agg(b["blg_area"], gid, "sum")
    f["height_p90"] = _agg(b["height_m"], gid, lambda s: s.quantile(0.9))
    f["height_cv"] = f["height_std"] / f["height_mean"]
    sl = b["height_m"] / np.sqrt(b["blg_area"])
    f["slenderness_mean"] = _agg(sl, gid, "mean")
    f["slenderness_max"] = _agg(sl, gid, "max")
    f["osr"] = (1 - f["gsi"]) / f["fsi"]
    f["layers_mean"] = f["fsi"] / f["gsi"]
    bulk = b["height_m"] * b["floor_area_m2"].fillna(b["blg_area"])
    f["bulk_density"] = _agg(bulk, gid, "sum") / f["blk_area"]
    f["blg_area_mean"] = _agg(b["blg_area"], gid, "mean")
    f["blg_area_std"] = _agg(b["blg_area"], gid, "std")
    f["elongation_mean"] = _agg(b["blg_elongation"], gid, "mean")
    f["eri_mean"] = _agg(b["blg_eri"], gid, "mean")
    f["car_mean"] = _agg(b["car"], gid, "mean")
    f["dist_to_street_mean"] = _agg(b["dist_to_street"], gid, "mean")
    f["shared_walls_mean"] = _agg(b["shared_walls_ratio"], gid, "mean")
    f["courtyards_per_blg"] = _agg(b["courtyards"], gid, "sum") / f["n_buildings"]
    f["neighbor_dist_mean"] = _agg(b["neighbor_distance"], gid, "mean")
    f["alignment_mean"] = _agg(b["alignment"], gid, "mean")
    f["street_alignment_mean"] = _agg(b["street_alignment"], gid, "mean")
    f["adjacency_mean"] = _agg(b["building_adjacency"], gid, "mean")
    f["edge_building_share"] = _agg(b["is_edge_building"], gid, "mean")
    f["dist_to_edge_norm_mean"] = _agg(b["dist_to_edge_norm"], gid, "mean")
    if typology is not None:
        f = f.join(typology_shares(blocks, typology))
    f = f.join(_roads(blocks, streets))
    g = Graph.build_fuzzy_contiguity(blk, buffer=p.context_buffer_m).transform("r")
    for col in ["gsi", "fsi", "height_mean", "blg_area_mean", "shared_walls_mean",
                "blg_density_ha"]:
        f[f"ctx_{col}"] = g.lag(f[col].fillna(f[col].median()))
    f = f.replace([np.inf, -np.inf], np.nan)
    cols = FULL57 if typology is not None else MORPHO_COLUMNS
    return f.reindex(columns=cols)
