"""End-to-end analysis: areas -> blocks -> descriptors -> treatment ->
local models (one per area) and/or a pooled model -> outputs.

Outputs (see docs/outputs.md):
  <out>/areas/<id>/blocks.gpkg, buildings.gpkg, streets.gpkg
  <out>/descriptors.csv              model input, one row per block_key
  <out>/assignments.csv              block_key, area, local / pooled type + name
  <out>/models/<model>/k_selection.csv, profiles.csv, representatives.json
  <out>/models/<model>/report.json   K, rule or fallback, PCA variance, quality
  <out>/run.json                     configuration, versions, descriptor set
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml

from . import __version__
from .blocks import enclosures, make_blocks, tessellate
from .descriptors import FULL57, MORPHO_COLUMNS, block_descriptors, building_characters
from .inputs import (choose_crs, load_boundary, load_typology, log, prepare_buildings,
                     prepare_streets)
from .missing import impute_heights
from .model import pca_space, select_k, stability
from .naming import name_clusters, order_clusters
from .params import Params
from .profiles import block_stats, representatives, type_profiles


def load_config(path):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = path.parent

    def rel(v):
        if v in (None, "osm") or isinstance(v, (int, float)):
            return v
        if isinstance(v, dict):
            return {k: rel(x) if k == "path" else x for k, x in v.items()}
        return str((base / v).resolve()) if not Path(v).is_absolute() else v
    for a in cfg["areas"]:
        for k in ("boundary", "buildings", "streets", "typology_predictions"):
            if k in a:
                a[k] = rel(a[k])
        a["heights"] = [rel(h) for h in a.get("heights", []) or []]
    cfg["output_dir"] = str((base / cfg.get("output_dir", "outputs")).resolve())
    if cfg.get("names"):
        cfg["names"] = rel(cfg["names"])
    return cfg


def prepare_area(a, crs, p: Params, descriptor_set):
    t0 = time.time()
    area = load_boundary(a["boundary"]).to_crs(crs)
    b = prepare_buildings(a.get("buildings", "osm"), area, crs, p, a.get("heights"))
    s = prepare_streets(a.get("streets", "osm"), area, crs, p, bool(a.get("cut_all_roads", False)))
    enc = enclosures(s, area)
    blocks, open_space, b = make_blocks(b, enc, p)
    tess = tessellate(b, enc)
    chars = building_characters(b, tess, s, blocks, p)
    chars["floors_measured"] = b["floors"].where(b["height_m"].notna())
    chars, imp = impute_heights(chars, p)
    typ = None
    if descriptor_set == "full57":
        if not a.get("typology_predictions"):
            raise ValueError(
                f"area {a['id']}: descriptors 'full57' need typology_predictions "
                "(building-typology output). Use descriptors: morphometric46 for a "
                "morphometric-only model (a different model from the manuscript's).")
        typ = load_typology(a["typology_predictions"], crs)
    feats = block_descriptors(chars, blocks, s, typ, p)
    log(f"[area {a['id']}] {len(blocks)} blocks, {feats.shape[1]} descriptors "
        f"({time.time() - t0:.0f}s)")
    return {"area": area, "blocks": blocks, "buildings": chars, "streets": s,
            "features": feats, "imputation": imp}


def _names(cfg_names, model_id, feats, labels):
    auto = name_clusters(feats, labels)
    given = (cfg_names or {}).get(model_id, {})
    return {int(k): given.get(str(int(k)), v) for k, v in auto.items()}, \
        {int(k): (str(int(k)) in given) for k in auto}


def fit_model(model_id, feats, area_of, stats, p: Params, cfg_names, out: Path,
              run_stability=True):
    z, pca = pca_space(feats, p)
    labels, sweep, status = select_k(z, p)
    names, approved = _names(cfg_names, model_id, feats, labels)
    order = order_clusters(feats, labels)
    d = out / "models" / model_id
    d.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(d / "k_selection.csv", index=False)
    prof = type_profiles(stats, labels, names, area_of)
    prof["order"] = [order.index(t) for t in prof["type"]]
    prof.to_csv(d / "profiles.csv", index=False)
    reps = representatives(stats, labels, area_of)
    json.dump({str(k): v for k, v in reps.items()}, open(d / "representatives.json", "w"), indent=1)
    sel = sweep[sweep["selected"]].iloc[0]
    rep = {"model": model_id, "blocks": int(len(feats)), "descriptors": int(feats.shape[1]),
           **status, "pca_components": int(z.shape[1]),
           "pca_variance": float(pca.explained_variance_ratio_.sum()),
           "silhouette": float(sel["silhouette"]), "silhouette_gap": float(sel["silhouette_gap"]),
           "calinski": float(sel["calinski"]), "davies": float(sel["davies"]),
           "type_order": [int(t) for t in order],
           "names": {str(k): v for k, v in names.items()},
           "names_approved": {str(k): v for k, v in approved.items()}}
    if run_stability and p.stability_reps:
        rep["stability"] = stability(feats, status["selected_k"], p)
    json.dump(rep, open(d / "report.json", "w"), indent=1)
    flag = "" if status["status"] == "rule" else "  ** FALLBACK - exploratory **"
    log(f"[model {model_id}] K = {status['selected_k']} ({status['status']}){flag}")
    return labels, names, rep


def run(config_path, areas_only=None):
    cfg = load_config(config_path)
    p = Params.from_dict(cfg.get("parameters"))
    descriptor_set = cfg.get("descriptors", "full57")
    if descriptor_set not in ("full57", "morphometric46"):
        raise ValueError("descriptors: full57 | morphometric46")
    areas = [a for a in cfg["areas"] if not areas_only or a["id"] in areas_only]
    out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    crs = choose_crs([load_boundary(a["boundary"]) for a in areas], cfg.get("crs", "auto"))
    log(f"[run] {len(areas)} areas, CRS {crs}, descriptors {descriptor_set}")
    names = json.loads(Path(cfg["names"]).read_text(encoding="utf-8")) if cfg.get("names") else {}
    res, feats_all, stats_all, area_of = {}, [], [], []
    for a in areas:
        r = prepare_area(a, crs, p, descriptor_set)
        res[a["id"]] = r
        keys = [f"{a['id']}-{int(i)}" for i in r["features"].index]
        f = r["features"].copy(); f.index = keys
        ch = r["buildings"].copy()
        ch["block_key"] = [f"{a['id']}-{int(i)}" if pd.notna(i) else None for i in ch["block_id"]]
        st = block_stats(ch, f, keys, p.floor_height_m)
        r["keys"], r["features_k"], r["stats"] = keys, f, st
        feats_all.append(f); stats_all.append(st); area_of += [a["id"]] * len(keys)
        d = out / "areas" / a["id"]; d.mkdir(parents=True, exist_ok=True)
        bl = r["blocks"][["block_id", "geometry"]].copy()
        bl["block_key"] = [f"{a['id']}-{int(i)}" for i in bl["block_id"]]
        bl["area"] = a["id"]
        bl.to_file(d / "blocks.gpkg")
        keep = ["block_id", "block_key", "height_m", "height_measured_m", "height_imputed",
                "footprint_m2", "geometry"]
        bb = ch[keep].copy()
        bb["height_imputed"] = bb["height_imputed"].astype(int)
        if descriptor_set == "full57":
            t = load_typology(a["typology_predictions"], crs)
            pts = gpd.GeoDataFrame(geometry=bb.geometry.representative_point(), index=bb.index, crs=crs)
            j = gpd.sjoin(pts, t, how="left", predicate="within")
            bb["class_name"] = j[~j.index.duplicated()]["class_name"].reindex(bb.index).fillna("None")
        bb.to_file(d / "buildings.gpkg")
        r["streets"].to_file(d / "streets.gpkg")
        r["area"].to_frame("geometry").to_file(d / "boundary.gpkg")
    F = pd.concat(feats_all); S = pd.concat(stats_all)
    F.index.name = "block_key"
    F.to_csv(out / "descriptors.csv")
    S.to_csv(out / "block_stats.csv", index_label="block_key")
    asg = pd.DataFrame({"block_key": F.index, "area": area_of})
    reports = {}
    an = cfg.get("analysis", {"local": True, "pooled": True})
    if an.get("local", True):
        loc_t, loc_n = [], []
        for a in areas:
            r = res[a["id"]]
            lab, nm, rep = fit_model(a["id"], r["features_k"], [a["id"]] * len(r["keys"]),
                                     r["stats"], p, names, out)
            loc_t += list(lab); loc_n += [nm[int(x)] for x in lab]
            reports[a["id"]] = rep
        asg["local_type"] = loc_t; asg["local_name"] = loc_n
    if an.get("pooled", True) and len(areas) > 1:
        lab, nm, rep = fit_model("pooled", F, area_of, S, p, names, out)
        asg["pooled_type"] = lab; asg["pooled_name"] = [nm[int(x)] for x in lab]
        reports["pooled"] = rep
    asg.to_csv(out / "assignments.csv", index=False)
    run_info = {"package": "urban-block-morphology", "version": __version__,
                "python": platform.python_version(), "crs": str(crs),
                "descriptor_set": descriptor_set,
                "descriptor_note": ("manuscript model: 46 morphometric + 10 typology shares + 1 count"
                                    if descriptor_set == "full57" else
                                    "MORPHOMETRIC-ONLY model (46 descriptors) - not the manuscript's 57-descriptor model"),
                "descriptors": FULL57 if descriptor_set == "full57" else MORPHO_COLUMNS,
                "areas": [{"id": a["id"], "name": a.get("name", a["id"]),
                           "cut_all_roads": bool(a.get("cut_all_roads", False))} for a in areas],
                "imputation": {a["id"]: res[a["id"]]["imputation"] for a in areas},
                "parameters": {k: (list(v) if isinstance(v, tuple) else v)
                               for k, v in p.__dict__.items()},
                "models": {k: {x: v[x] for x in ("selected_k", "status", "reason")}
                           for k, v in reports.items()}}
    try:
        import geopandas, momepy, sklearn
        run_info["libraries"] = {"momepy": momepy.__version__, "scikit-learn": sklearn.__version__,
                                 "geopandas": geopandas.__version__,
                                 "numpy": np.__version__, "pandas": pd.__version__}
    except Exception:                                             # noqa: BLE001
        pass
    json.dump(run_info, open(out / "run.json", "w"), indent=1, default=str)
    log(f"[run] done -> {out}")
    return out
