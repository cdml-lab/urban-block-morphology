"""Results dashboard: one self-contained HTML page built from an output folder.

    blockmorph dashboard <output_dir> [--title ...] [--colours colours.json]

Layout: a control column, one map panel per area
(blocks, streets and buildings, 2D or 3D) with hover cards, and two further
views - Types (composition by area, profiles and representative blocks in
3D with their surroundings) and Model quality (K selection, stability,
local x pooled). Every value is read from the run's outputs.

The interactive maps use deck.gl (pinned, loaded from a CDN). Without it
(offline, no WebGL) the maps fall back to a 2D canvas; the Types and Model
quality views need no library.

Colours: `colours.json` in the output folder or --colours
({"pooled": {"0": "#3ca951", ...}, "<area id>": {...}}); otherwise the
pooled types take the editorial palette in id order and each local type the
colour of the pooled type it overlaps most (so similar types share colours).
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .descriptors import TYPOLOGY_CLASSES

# editorial palette (Observable-10 lineage), as in the manuscript figures
PALETTE = ["#4269d0", "#efb118", "#ff725c", "#6cc5b0", "#a463f2", "#3ca951", "#ff8ab7",
           "#56b3e6", "#d5477a", "#9c6b4e", "#a3b031", "#6157d6", "#e88c30", "#2e9e8f",
           "#c87dd2", "#d9a46b"]
STREET_CLASS = {**{k: 0 for k in ("motorway", "motorway_link", "trunk", "trunk_link", "primary",
                                  "primary_link", "secondary", "secondary_link", "tertiary",
                                  "tertiary_link")},
                **{k: 1 for k in ("residential", "living_street", "unclassified", "service")}}
DECK = "https://unpkg.com/deck.gl@9.0.35/dist.min.js"


# ------------------------------------------------------------------ geometry
def _ring(coords, cx, cy):
    c = np.asarray(coords)[:-1]
    return np.round((c - (cx, cy)) * 10).astype(int).ravel().tolist()     # decimetres


def _poly(g, cx, cy, tol):
    g = g.simplify(tol, preserve_topology=True) if tol else g
    out = []
    for p in getattr(g, "geoms", [g]):
        out.append([_ring(p.exterior.coords, cx, cy)] +
                   [_ring(r.coords, cx, cy) for r in p.interiors])
    return out          # list of polygons, each a list of rings


def _area_payload(out: Path, aid, name, models_present, asg, stats, forms=True):
    d = out / "areas" / aid
    bl = gpd.read_file(d / "blocks.gpkg")
    bd = gpd.read_file(d / "buildings.gpkg")
    st = gpd.read_file(d / "streets.gpkg")
    x0, y0, x1, y1 = bl.total_bounds
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    lng, lat = gpd.GeoSeries(gpd.points_from_xy([cx], [cy]), crs=bl.crs).to_crs(4326).iloc[0].coords[0]
    keys = bl["block_key"].tolist()
    kidx = {k: i for i, k in enumerate(keys)}
    a = asg.set_index("block_key").loc[keys]
    s = stats.loc[keys]
    bd = bd[bd.geometry.notna() & ~bd.geometry.is_empty]
    polys = [_poly(g, cx, cy, 0.2)[0][0] for g in bd.geometry]
    form = {c: i for i, c in enumerate(TYPOLOGY_CLASSES)}
    sl = st.explode(index_parts=False)
    sl = sl[sl.geometry.geom_type == "LineString"].simplify(0.5)
    rnd = lambda v: [None if pd.isna(x) else round(float(x), 3) for x in v]       # noqa: E731
    return {
        "id": aid, "name": name, "origin": [lng, lat],
        "ext": [round(x0 - cx, 1), round(y0 - cy, 1), round(x1 - cx, 1), round(y1 - cy, 1)],
        "blocks": {"key": keys, "g": [_poly(g, cx, cy, 0.3) for g in bl.geometry],
                   "local": [int(v) for v in a["local_type"]] if "local_type" in a else None,
                   "pooled": [int(v) for v in a["pooled_type"]] if "pooled_type" in a else None,
                   "gsi": rnd(s["gsi"]), "fsi": rnd(s["fsi_measured"]),
                   "h": rnd(s["height_measured_m"]), "n": [int(v) for v in s["n_buildings"]],
                   "ha": rnd(s["blk_area_ha"]), "known": rnd(s["heights_measured"])},
        "bld": {"g": polys, "h": [int(round(float(h) * 10)) for h in bd["height_m"].fillna(3.2)],
                "imp": [int(v) for v in bd["height_imputed"]],
                "b": [kidx.get(k, -1) for k in bd["block_key"].fillna("")],
                "f": ([form.get(c, -1) for c in bd["class_name"]]
                      if forms and "class_name" in bd else None)},
        "st": {"g": [_ring(list(g.coords) + [g.coords[-1]], cx, cy) for g in sl.geometry],
               "c": [STREET_CLASS.get(h, 2) for h in st.loc[sl.index, "highway"]]},
    }


# -------------------------------------------------------------------- models
def _colours(models, asg, given):
    col = {}
    if "pooled" in models:
        col["pooled"] = {t: PALETTE[i % len(PALETTE)]
                         for i, t in enumerate(sorted(int(x) for x in models["pooled"]["names"]))}
    for mid in models:
        if mid == "pooled":
            continue
        types = sorted(int(x) for x in models[mid]["names"])
        c = {}
        if "pooled" in col:
            sub = asg[asg["area"] == mid]
            ct = pd.crosstab(sub["local_type"], sub["pooled_type"])
            for t in ct.sum(axis=1).sort_values(ascending=False).index:
                p = int(ct.loc[t].idxmax())
                if col["pooled"][p] not in c.values():
                    c[int(t)] = col["pooled"][p]
        spare = [x for x in PALETTE if x not in c.values()]
        for t in types:
            c.setdefault(t, spare.pop(0) if spare else "#9498a0")
        col[mid] = c
    for mid, m in (given or {}).items():
        if mid in col:
            col[mid].update({int(k): v for k, v in m.items()})
    return {m: {str(k): v for k, v in c.items()} for m, c in col.items()}


def _model(d: Path):
    rep = json.loads((d / "report.json").read_text(encoding="utf-8"))
    prof = pd.read_csv(d / "profiles.csv").sort_values("type")
    sw = pd.read_csv(d / "k_selection.csv")
    return {"id": rep["model"], "k": rep["selected_k"], "status": rep["status"],
            "reason": rep["reason"], "elbow": rep.get("elbow_k"),
            "admissible": rep.get("admissible_k", []), "blocks": rep["blocks"],
            "pca_var": rep["pca_variance"], "silhouette": rep["silhouette"],
            "sil_gap": rep["silhouette_gap"], "stability": rep.get("stability"),
            "names": rep["names"],
            "profiles": json.loads(prof.to_json(orient="records")),
            "sweep": json.loads(sw.to_json(orient="records")),
            "reps": json.loads((d / "representatives.json").read_text(encoding="utf-8"))}


def build(out_dir, title=None, colours=None, building_forms=True) -> Path:
    """building_forms=False leaves the per-building form predictions out of
    the page (block-level results are unaffected), e.g. when the imagery
    terms do not allow redistributing building-level predictions."""
    out = Path(out_dir)
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    asg = pd.read_csv(out / "assignments.csv")
    stats = pd.read_csv(out / "block_stats.csv", index_col="block_key")
    models = {d.name: _model(d) for d in sorted((out / "models").iterdir())
              if (d / "report.json").exists()}
    cpath = Path(colours) if colours else out / "colours.json"
    given = json.loads(cpath.read_text(encoding="utf-8")) if cpath.exists() else {}
    cols = _colours(models, asg, given)
    for mid in models:
        models[mid]["colours"] = cols[mid]
    areas = [_area_payload(out, a["id"], a.get("name") or a["id"], models, asg, stats,
                           building_forms)
             for a in sorted(run["areas"], key=lambda a: str(a.get("name") or a["id"]))]
    data = {"run": {k: run.get(k) for k in ("version", "crs", "descriptor_set",
                                            "descriptor_note", "descriptors")},
            "areas": areas, "models": models, "forms": TYPOLOGY_CLASSES,
            "order": [m for m in ["pooled"] + [a["id"] for a in areas] if m in models]}
    title = title or "Urban-block types"
    page = (TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__DECK__", DECK)
            .replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/")))
    path = out / "dashboard.html"
    path.write_text(page, encoding="utf-8")
    return path


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Lato:wght@300;400;700&display=swap" rel="stylesheet">
<script src="__DECK__"></script>
<style>
:root{--bg:#ffffff;--panel:#fafafa;--line:#e8e8e8;--ink:#222222;--muted:#6b6b6b;--faint:#9a9a9a;
--accent:#222222;--warn:#c4456f;--chip:#f3f3f3}
*{box-sizing:border-box}
[hidden]{display:none!important}
html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:300 14px/1.45 Lato,"Segoe UI",system-ui,sans-serif;
display:flex;flex-direction:column}
b,strong,h1,h2,h3{font-weight:700}
header{display:flex;align-items:baseline;gap:18px;padding:12px 20px 0;border-bottom:1px solid var(--line);flex-wrap:wrap}
header h1{font-size:18px;margin:0;font-weight:400;letter-spacing:.2px}
header .sub{color:var(--muted);font-size:13px}
nav{margin-left:auto;display:flex;gap:2px}
nav button{border:0;background:none;font:inherit;font-size:13.5px;color:var(--muted);padding:8px 12px 10px;
cursor:pointer;border-bottom:2px solid transparent}
nav button.on{color:var(--ink);border-bottom-color:var(--ink);font-weight:400}
.wrap{flex:1;display:flex;min-height:0}
aside{width:258px;flex:none;background:var(--panel);border-right:1px solid var(--line);padding:14px 16px 24px;overflow:auto}
aside h4{font-size:10.5px;font-weight:700;letter-spacing:1.3px;color:var(--muted);margin:16px 0 7px;text-transform:uppercase}
aside h4:first-child{margin-top:2px}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{border:1px solid var(--line);background:#fff;border-radius:999px;padding:3px 11px;font:inherit;font-size:13px;
color:var(--ink);cursor:pointer}
.chip.on{background:var(--ink);border-color:var(--ink);color:#fff}
.opt{display:flex;align-items:center;gap:8px;padding:3px 6px;margin:0 -6px;border-radius:5px;cursor:pointer;font-size:13.5px}
.opt:hover{background:#f0f0f0}
.opt input{accent-color:var(--ink);margin:0}
.legend .row{display:flex;gap:8px;align-items:flex-start;padding:3px 6px;margin:0 -6px;border-radius:5px;cursor:pointer;font-size:13px}
.legend .row:hover{background:#f0f0f0}
.legend .row.dim{opacity:.35}
.sw{width:13px;height:13px;border-radius:2px;flex:none;margin-top:3px}
.legend .n{margin-left:auto;color:var(--muted);padding-left:6px}
.code{font-weight:700;margin-right:4px}
.note{font-size:12px;color:var(--warn);margin-top:10px;line-height:1.4}
.small{font-size:12px;color:var(--muted)}
main{flex:1;min-width:0;position:relative;overflow:auto}
#mapview{position:absolute;inset:0;display:flex}
.panel{flex:1;min-width:0;position:relative;overflow:hidden;border-right:1px solid var(--line)}
.panel:last-child{border-right:0}
.panel .ptitle{position:absolute;left:14px;top:10px;z-index:2;pointer-events:none}
.panel .ptitle b{font-size:15px;font-weight:700}
.panel .ptitle div{font-size:12px;color:var(--muted)}
.panel .reset{position:absolute;right:10px;top:10px;z-index:2;border:1px solid var(--line);background:#fff;
border-radius:6px;font:inherit;font-size:12px;padding:3px 9px;cursor:pointer;color:var(--muted)}
.panel canvas{display:block}
.cards{position:absolute;right:14px;top:46px;width:250px;z-index:3;pointer-events:none;display:flex;flex-direction:column;gap:8px}
.card{background:rgba(255,255,255,.96);border:1px solid var(--line);border-radius:8px;padding:10px 12px;font-size:12.5px;
box-shadow:0 2px 10px rgba(0,0,0,.05)}
.card h5{margin:0 0 4px;font-size:10.5px;letter-spacing:1.3px;color:var(--muted);font-weight:700}
.card .hint{color:var(--faint);font-style:italic}
.card table{border-collapse:collapse;width:100%;margin-top:4px}
.card td{padding:1px 0}.card td+td{text-align:right}
.page{max-width:1180px;margin:0 auto;padding:22px 28px 60px}
.page h2{font-size:17px;font-weight:400;margin:0 0 4px}
.page .lede{color:var(--muted);font-size:13px;margin:0 0 18px}
table.comp{border-collapse:collapse;width:100%;font-size:13.5px}
table.comp th{font-weight:400;text-align:left;color:var(--ink);padding:0 10px 4px 0}
table.comp th small{display:block;color:var(--muted);font-size:12px}
table.comp td{padding:6px 10px 6px 0;border-bottom:1px solid #f0f0f0;vertical-align:middle}
.bar{display:flex;align-items:center;gap:6px}
.bar i{display:block;height:12px;border-radius:1px}
.trow{display:grid;grid-template-columns:230px repeat(3,minmax(0,1fr));gap:10px;border-bottom:1px solid #ececec;padding:14px 0 6px}
.trow .tx .code{font-size:16px}
.trow .tx .nm{font-size:16px;line-height:1.3;margin:6px 0 6px}
.trow .tx .meta{color:var(--muted);font-size:12.5px;line-height:1.55}
.trow canvas{width:100%;height:auto;display:block}
.trow .lab{color:var(--muted);font-size:11.5px;margin-top:-2px}
@media (max-width:900px){.trow{grid-template-columns:1fr 1fr}.trow .tx{grid-column:1/-1}}
.foot{color:var(--muted);font-size:12px;margin-top:12px;line-height:1.6}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:26px 34px}
.kchart h3{font-size:15px;font-weight:700;margin:0 0 2px}
.kchart .st{font-size:12px;color:var(--muted);margin-bottom:4px}
.kchart svg{width:100%;height:auto;font-family:inherit}
.mcards{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-bottom:26px}
.mcard{border:1px solid var(--line);border-radius:8px;padding:12px 14px;font-size:13px}
.mcard h3{margin:0 0 6px;font-size:14px}
.mcard .k{font-size:24px;font-weight:300}
.fb{color:var(--warn)}
table.ct{border-collapse:collapse;font-size:12.5px}
table.ct td,table.ct th{padding:3px 6px;text-align:right;font-weight:400}
table.ct td.l,table.ct th.l{text-align:left}
.ctwrap{overflow-x:auto;margin-bottom:22px}
.klegend{display:flex;flex-wrap:wrap;gap:6px 22px;font-size:12.5px;margin:6px 0 20px;color:var(--ink)}
.klegend span{display:inline-flex;align-items:center;gap:6px}
@media (max-width:760px){.wrap{flex-direction:column}aside{width:auto;border-right:0;border-bottom:1px solid var(--line)}
main{min-height:70vh}#mapview{flex-direction:column}.panel{min-height:60vh;border-right:0;border-bottom:1px solid var(--line)}}
</style></head><body>
<header><h1>__TITLE__</h1><span class="sub" id="sub"></span>
<nav id="tabs"><button data-v="map" class="on">Map</button><button data-v="types">Types</button><button data-v="quality">Model quality</button></nav></header>
<div class="wrap"><aside id="side"></aside><main id="main">
<div id="mapview"></div><div class="cards" id="cards"></div>
<div id="pageview" hidden></div></main></div>
<script>
const D = __DATA__;
const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const fmt = (v, d=1) => v == null || isNaN(v) ? "–" : (+v).toLocaleString("en", {minimumFractionDigits:d, maximumFractionDigits:d});
const AREA = Object.fromEntries(D.areas.map(a => [a.id, a]));
const hasPooled = !!D.models.pooled;
const st = {tab:"map", model: hasPooled ? "pooled" : "local", areas: D.areas.map(a=>a.id), colour:"type",
  three:true, ground:true, streets:true, focus:null, tmodel: D.order[0], hover:{}};
const code = (mid, t) => (mid === "pooled" ? "P" : mid) + t;
const modelOf = aid => st.model === "pooled" ? "pooled" : aid;
const lab = (a, mid) => mid === "pooled" ? a.blocks.pooled : a.blocks.local;
const label = mid => mid === "pooled" ? "Pooled model" : `${AREA[mid].name} · local model`;
const hex = h => [parseInt(h.slice(1,3),16), parseInt(h.slice(3,5),16), parseInt(h.slice(5,7),16)];
const mix = (h, a) => hex(h).map(v => Math.round(v*a + 255*(1-a)));     // tint toward white
const css = c => `rgb(${c.join(",")})`;
const HCLASS = [[0,"< 10 m","#f1e4c3"],[10,"10–25 m","#e9a96b"],[25,"25–40 m","#c2557a"],[40,"≥ 40 m","#4b3370"]];
const GCLASS = [[0,"GSI < 0.25 (open)","#e4eedd"],[0.25,"0.25–0.40","#a9c99a"],[0.40,"≥ 0.40 (compact)","#5f8f5a"]];
const FORMS = ["#4269d0","#efb118","#ff725c","#6cc5b0","#a463f2","#3ca951","#ff8ab7","#56b3e6","#d5477a","#9c6b4e"];
const hcls = h => HCLASS.filter(c => h >= c[0]).pop();
const gcls = g => GCLASS.filter(c => g >= c[0]).pop();

// ---------------------------------------------------------------- geometry
for (const a of D.areas) {
  const P = r => { const o = []; for (let i = 0; i < r.length; i += 2) o.push([r[i]/10, r[i+1]/10]); return o; };
  a.B = a.blocks.g.map(ps => ps.map(rings => rings.map(P)));
  a.G = a.bld.g.map(P);
  a.S = a.st.g.map(P);
  a.cen = a.G.map(r => { let x=0,y=0; for (const p of r){x+=p[0];y+=p[1];} return [x/r.length, y/r.length]; });
}

// ---------------------------------------------------------------- header + tabs
const nb = D.areas.reduce((s,a) => s + a.blocks.key.length, 0);
$("#sub").textContent = `${D.areas.map(a=>a.name).join(" · ")} — ${nb.toLocaleString("en")} blocks · ` +
  `${D.run.descriptors.length} descriptors${D.run.descriptor_set !== "full57" ? " (morphometric only)" : ""} · blockmorph ${D.run.version}`;
$("#tabs").onclick = e => { const b = e.target.closest("button"); if (!b) return; st.tab = b.dataset.v;
  [...$("#tabs").children].forEach(x => x.classList.toggle("on", x === b)); render(); };

// ---------------------------------------------------------------- colours
function blockColour(a, i, mid) {
  if (st.colour === "gsi") { const g = a.blocks.gsi[i]; return g == null ? [236,236,236] : hex(gcls(g)[2]); }
  const t = lab(a, mid)[i], c = D.models[mid].colours[t];
  if (st.focus != null && t !== st.focus) return [244,244,244];
  return st.colour === "type" ? mix(c, .28) : [238,238,238];
}
function bldColour(a, j, mid) {
  const bi = a.bld.b[j];
  if (st.colour === "type") {
    if (bi < 0) return [205,205,205];
    const t = lab(a, mid)[bi];
    if (st.focus != null && t !== st.focus) return [226,226,226];
    return hex(D.models[mid].colours[t]);
  }
  if (st.colour === "height") return hex(hcls(a.bld.h[j]/10)[2]);
  if (st.colour === "known") return a.bld.imp[j] ? [232,179,75] : [91,143,203];
  if (st.colour === "form") { const f = a.bld.f ? a.bld.f[j] : -1; return f < 0 ? [222,222,222] : hex(FORMS[f]); }
  if (st.colour === "gsi") return [120,120,120];
}

// ---------------------------------------------------------------- side column
function chips(opts, cur, multi) {
  return `<div class="chips">${opts.map(([v,l]) => `<button class="chip ${(multi ? cur.includes(v) : cur === v) ? "on" : ""}" data-v="${esc(v)}">${esc(l)}</button>`).join("")}</div>`;
}
function legendHTML() {
  if (st.colour === "type") {
    const mids = st.model === "pooled" ? ["pooled"] : st.areas;
    return mids.map(mid => { const m = D.models[mid], cnt = {};
      for (const a of D.areas) if (st.areas.includes(a.id) && (mid === "pooled" || a.id === mid)) lab(a, mid).forEach(t => cnt[t] = (cnt[t]||0)+1);
      return `${mids.length > 1 ? `<div class="small" style="margin:8px 0 3px">${esc(AREA[mid].name)}${m.status!=="rule"?' <span class="fb">· fallback (exploratory)</span>':""}</div>` : ""}` +
        Object.keys(m.names).map(Number).sort((x,y)=>x-y).filter(t => cnt[t]).map(t =>
        `<div class="row ${st.focus!=null && st.focus!==t && mids.length===1 ? "dim" : ""}" data-t="${t}"><span class="sw" style="background:${m.colours[t]}"></span>
        <span><span class="code">${code(mid,t)}</span>${esc(m.names[t])}</span><span class="n">${cnt[t]}</span></div>`).join(""); }).join("") +
      `<div class="small" style="margin-top:6px">${st.model==="pooled" ? "click a type to isolate it · " : ""}blocks tinted, buildings in full colour</div>`;
  }
  const L = st.colour === "height" ? HCLASS.map(c => [c[2], c[1]]) : st.colour === "gsi" ? GCLASS.map(c => [c[2], c[1]])
    : st.colour === "known" ? [["#5b8fcb","height known"],["#e8b34b","height imputed"]]
    : D.forms.map((f,i) => [FORMS[i], f]).concat([["#dedede","no prediction"]]);
  return L.map(([c,l]) => `<div class="row"><span class="sw" style="background:${c}"></span><span>${esc(l)}</span></div>`).join("");
}
function side() {
  const A = D.areas.map(a => [a.id, a.name]);
  let h = "";
  if (st.tab === "map") {
    h += `<h4>Typology</h4>` + chips([...(hasPooled ? [["pooled","Pooled"]] : []), ["local","Local"]], st.model) +
      `<div class="small" style="margin-top:6px">${st.model==="pooled" ? "one model over all areas — types comparable across areas" : "each area clustered on its own — types are area-specific"}</div>`;
    h += `<h4>Areas</h4>` + chips(A, st.areas, true);
    const CO = [["type","Block type"],["height","Building height"],["known","Height known / imputed"],
      ...(D.areas[0].bld.f ? [["form","Building form (R-CNN)"]] : []), ["gsi","Block coverage (GSI)"]];
    h += `<h4>Colour by</h4>` + CO.map(([v,l]) => `<label class="opt"><input type="radio" name="col" value="${v}" ${st.colour===v?"checked":""}>${l}</label>`).join("");
    h += `<h4>Display</h4>` + [["three","3D buildings"],["ground","Block ground"],["streets","Streets"]].map(([v,l]) =>
      `<label class="opt"><input type="checkbox" data-d="${v}" ${st[v]?"checked":""} ${v==="three"&&!GL?"disabled":""}>${l}</label>`).join("") +
      (!GL ? `<div class="small" style="margin-top:4px">3D view needs WebGL and deck.gl — showing the 2D map.</div>` : "");
    h += `<h4>Legend</h4><div class="legend" id="legend">${legendHTML()}</div>`;
  } else {
    h += `<h4>Model</h4>` + chips(D.order.map(m => [m, m === "pooled" ? "Pooled" : AREA[m].name + " · local"]), st.tmodel);
  }
  const fb = D.order.filter(m => D.models[m].status !== "rule");
  if (fb.length) h += `<div class="note">${fb.map(m => `<b>${esc(m==="pooled"?"Pooled":AREA[m].name)}</b>: no K met both selection conditions — K = ${D.models[m].k} is a fallback partition (exploratory).`).join("<br>")}</div>`;
  if (D.run.descriptor_set !== "full57") h += `<div class="note">Morphometric-only model (46 descriptors, no building-form shares) — not the 57-descriptor model.</div>`;
  $("#side").innerHTML = h;
  $("#side").querySelectorAll(".chips").forEach((el, gi) => el.onclick = e => {
    const b = e.target.closest(".chip"); if (!b) return; const v = b.dataset.v;
    if (st.tab !== "map") { st.tmodel = v; return render(); }
    if (gi === 0) { st.model = v; st.focus = null; }
    else { st.areas = st.areas.includes(v) ? st.areas.filter(x => x !== v) : D.areas.map(a=>a.id).filter(x => x === v || st.areas.includes(x));
           if (!st.areas.length) st.areas = [v]; }
    render(); });
  $("#side").querySelectorAll("input[name=col]").forEach(r => r.onchange = () => { st.colour = r.value; st.focus = null; side(); restyle(); });
  $("#side").querySelectorAll("input[data-d]").forEach(r => r.onchange = () => { st[r.dataset.d] = r.checked; restyle(true); });
  const lg = $("#legend");
  if (lg) lg.onclick = e => { const r = e.target.closest(".row[data-t]"); if (!r || st.model !== "pooled") return;
    const t = +r.dataset.t; st.focus = st.focus === t ? null : t; side(); restyle(); };
}

// ---------------------------------------------------------------- hover cards
function cards() {
  const hb = st.hover.block, hg = st.hover.bld;
  let b = `<div class="card"><h5>BLOCK</h5><div class="hint">hover a block</div></div>`;
  if (hb) { const a = AREA[hb.area], i = hb.i, mid = modelOf(a.id), t = lab(a, mid)[i], m = D.models[mid];
    b = `<div class="card"><h5>BLOCK · ${esc(a.blocks.key[i])}</h5><div style="display:flex;gap:7px"><span class="sw" style="background:${m.colours[t]}"></span>
    <span><span class="code">${code(mid,t)}</span>${esc(m.names[t])}</span></div><table>
    <tr><td>buildings</td><td>${a.blocks.n[i]}</td></tr><tr><td>block area</td><td>${fmt(a.blocks.ha[i],2)} ha</td></tr>
    <tr><td>coverage GSI</td><td>${fmt(a.blocks.gsi[i],2)}</td></tr><tr><td>FSI*</td><td>${fmt(a.blocks.fsi[i],2)}</td></tr>
    <tr><td>height*</td><td>${fmt(a.blocks.h[i],1)} m</td></tr><tr><td>heights known</td><td>${fmt(100*a.blocks.known[i],0)} %</td></tr></table>
    <div class="small" style="margin-top:4px">* from heights available before imputation</div></div>`; }
  let g = `<div class="card"><h5>BUILDING</h5><div class="hint">hover a building</div></div>`;
  if (hg) { const a = AREA[hg.area], j = hg.j, bi = a.bld.b[j], f = a.bld.f ? a.bld.f[j] : -1;
    g = `<div class="card"><h5>BUILDING</h5><table><tr><td>height</td><td>${fmt(a.bld.h[j]/10,1)} m${a.bld.imp[j] ? " (imputed)" : ""}</td></tr>
    ${a.bld.f ? `<tr><td>form</td><td>${f < 0 ? "–" : esc(D.forms[f])}</td></tr>` : ""}<tr><td>block</td><td>${bi < 0 ? "outside blocks" : esc(a.blocks.key[bi])}</td></tr></table></div>`; }
  $("#cards").innerHTML = b + g;
}

// ---------------------------------------------------------------- maps (deck.gl, 2D canvas fallback)
let GL = false;
try { GL = !!(window.deck && document.createElement("canvas").getContext("webgl2")); } catch (e) { GL = false; }
let PANELS = [];
function fitZoom(a, w, h) {
  const ew = a.ext[2]-a.ext[0], eh = a.ext[3]-a.ext[1], lat = a.origin[1]*Math.PI/180;
  const mpp = Math.max(ew/(w*0.86), eh/(h*0.82));
  return Math.log2(40075016*Math.cos(lat)/(512*mpp));
}
function viewOf(a, el) {
  return {longitude: a.origin[0], latitude: a.origin[1], zoom: fitZoom(a, el.clientWidth, el.clientHeight) - (st.three ? .3 : 0),
    pitch: st.three ? 48 : 0, bearing: st.three ? -18 : 0, maxPitch: 75, minZoom: 10, maxZoom: 20};
}
function layers(a) {
  const mid = modelOf(a.id), O = {coordinateSystem: deck.COORDINATE_SYSTEM.METER_OFFSETS, coordinateOrigin: [a.origin[0], a.origin[1], 0]};
  const trig = [st.model, st.colour, st.focus, st.ground].join("|");
  const L = [];
  if (st.ground || st.colour === "gsi") L.push(new deck.PolygonLayer({id: "blk-"+a.id, ...O, data: a.B.flatMap((ps,i) => ps.map(r => ({r, i}))),
    getPolygon: d => d.r, getFillColor: d => (st.ground || st.colour==="gsi") ? blockColour(a, d.i, mid) : [0,0,0,0], filled: true, stroked: true,
    getLineColor: [255,255,255], lineWidthUnits: "pixels", getLineWidth: .6, pickable: true, updateTriggers: {getFillColor: trig},
    onHover: o => { st.hover.block = o.object ? {area: a.id, i: o.object.i} : null; cards(); }}));
  if (st.streets) L.push(new deck.PathLayer({id: "st-"+a.id, ...O, data: a.S.map((p,i) => ({p, c: a.st.c[i]})), getPath: d => d.p,
    getColor: d => d.c === 0 ? [214,214,214] : d.c === 1 ? [226,226,226] : [236,236,236], widthUnits: "pixels",
    getWidth: d => d.c === 0 ? 1.6 : d.c === 1 ? 1 : .6}));
  L.push(new deck.SolidPolygonLayer({id: "bd-"+a.id, ...O, data: a.G.map((p,j) => ({p, j})), getPolygon: d => d.p,
    extruded: st.three, getElevation: d => a.bld.h[d.j]/10, getFillColor: d => bldColour(a, d.j, mid), pickable: true,
    material: {ambient: .62, diffuse: .55, shininess: 8, specularColor: [40,40,40]},
    updateTriggers: {getFillColor: trig, extruded: st.three},
    onHover: o => { st.hover.bld = o.object ? {area: a.id, j: o.object.j} : null; cards(); }}));
  return L;
}
function canvasMap(a, el) {
  const cv = document.createElement("canvas"), dpr = window.devicePixelRatio || 1, W = el.clientWidth, H = el.clientHeight;
  cv.width = W*dpr; cv.height = H*dpr; cv.style.width = W+"px"; cv.style.height = H+"px"; cv.style.position = "absolute"; cv.style.inset = "0"; el.appendChild(cv);
  const ctx = cv.getContext("2d"), mid = modelOf(a.id);
  let s = Math.min(W*.86/(a.ext[2]-a.ext[0]), H*.82/(a.ext[3]-a.ext[1])), ox = W/2, oy = H/2;
  const X = x => ox + x*s, Y = y => oy - y*s;
  const paths = a.B.map(ps => { const p = new Path2D(); ps.forEach(rs => rs.forEach(r => { r.forEach(([x,y],k) => k ? p.lineTo(X(x),Y(y)) : p.moveTo(X(x),Y(y))); p.closePath(); })); return p; });
  ctx.scale(dpr, dpr);
  a.B.forEach((ps,i) => { ctx.fillStyle = css(st.ground || st.colour==="gsi" ? blockColour(a,i,mid) : [250,250,250]); ctx.fill(paths[i], "evenodd"); });
  if (st.streets) { ctx.strokeStyle = "#dcdcdc"; ctx.lineWidth = .6; for (const p of a.S) { ctx.beginPath(); p.forEach(([x,y],k) => k ? ctx.lineTo(X(x),Y(y)) : ctx.moveTo(X(x),Y(y))); ctx.stroke(); } }
  a.G.forEach((r,j) => { ctx.beginPath(); r.forEach(([x,y],k) => k ? ctx.lineTo(X(x),Y(y)) : ctx.moveTo(X(x),Y(y))); ctx.closePath(); ctx.fillStyle = css(bldColour(a,j,mid)); ctx.fill(); });
  cv.onmousemove = e => { const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    const i = paths.findIndex(p => ctx.isPointInPath(p, x*dpr, y*dpr, "evenodd"));
    st.hover.block = i >= 0 ? {area: a.id, i} : null; cards(); };
}
function drawMaps() {
  PANELS.forEach(p => p.deck && p.deck.finalize()); PANELS = [];
  const mv = $("#mapview"); mv.innerHTML = "";
  for (const aid of st.areas) { const a = AREA[aid];      // lay out every panel first, then draw
    const el = document.createElement("div"); el.className = "panel";
    const mid = modelOf(aid);
    el.innerHTML = `<div class="ptitle"><b>${esc(a.name)}</b><div>${a.bld.h.length.toLocaleString("en")} buildings · ${a.blocks.key.length.toLocaleString("en")} blocks${D.models[mid] && D.models[mid].status!=="rule" ? ' · <span class="fb">fallback K — exploratory</span>' : ""}</div></div>` +
      (GL ? `<button class="reset">⟲ view</button>` : "");
    mv.appendChild(el); PANELS.push({a, el});
  }
  for (const P of PANELS) { const {a, el} = P;
    if (GL) {
      P.deck = new deck.Deck({parent: el, views: new deck.MapView({repeat: false}), initialViewState: viewOf(a, el),
        controller: {dragRotate: true, touchRotate: true}, layers: layers(a), style: {background: "#ffffff"},
        getCursor: ({isHovering}) => isHovering ? "pointer" : "grab"});
      el.querySelector(".reset").onclick = () => P.deck.setProps({initialViewState: {...viewOf(a, el), transitionDuration: 400}});
    } else canvasMap(a, el);
  }
}
function restyle(view) {
  if (!GL) return drawMaps();
  PANELS.forEach(p => p.deck.setProps({layers: layers(p.a), ...(view ? {initialViewState: {...viewOf(p.a, p.el), transitionDuration: 500}} : {})}));
}

// ---------------------------------------------------------------- Types page
function shade(rgb, f) { return css(rgb.map(v => Math.round(f < 0 ? v*(1+f) : v + (255-v)*f))); }
function axo(cv, a, key, colour, scale) {
  const W = 420, H = 260, dpr = 2; cv.width = W*dpr; cv.height = H*dpr;
  const ctx = cv.getContext("2d"); ctx.scale(dpr, dpr);
  const i = a.blocks.key.indexOf(key); if (i < 0) return;
  const rings = a.B[i].flat(); let x0=1e9,y0=1e9,x1=-1e9,y1=-1e9;
  rings.forEach(r => r.forEach(([x,y]) => { x0=Math.min(x0,x); x1=Math.max(x1,x); y0=Math.min(y0,y); y1=Math.max(y1,y); }));
  const cx = (x0+x1)/2, cy = (y0+y1)/2, th = -0.55, c = Math.cos(th), s = Math.sin(th), k = .56;
  const P = (x, y, z=0) => { const dx = x-cx, dy = y-cy, rx = dx*c - dy*s, ry = dx*s + dy*c; return [W/2 + rx*scale, H*0.56 - ry*scale*k - z*scale*0.82]; };
  const RX = W/2 - 6, RY = H/2 - 8, R = RX/scale*1.25;
  ctx.save(); ctx.beginPath(); ctx.ellipse(W/2, H/2+2, RX, RY, 0, 0, 2*Math.PI); ctx.clip();
  ctx.fillStyle = "#ececec"; ctx.fillRect(0,0,W,H);
  const poly = (pts, z) => { ctx.beginPath(); pts.forEach(([x,y],q) => { const p = P(x,y,z); q ? ctx.lineTo(...p) : ctx.moveTo(...p); }); ctx.closePath(); };
  ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 4.5; ctx.lineCap = "round";
  a.S.forEach(p => { if (!p.some(([x,y]) => Math.hypot(x-cx,y-cy) < R)) return; ctx.beginPath(); p.forEach(([x,y],q) => { const pt = P(x,y); q ? ctx.lineTo(...pt) : ctx.moveTo(...pt); }); ctx.stroke(); });
  ctx.strokeStyle = "#cfcfcf"; ctx.lineWidth = .7;
  a.S.forEach(p => { if (!p.some(([x,y]) => Math.hypot(x-cx,y-cy) < R)) return; ctx.beginPath(); p.forEach(([x,y],q) => { const pt = P(x,y); q ? ctx.lineTo(...pt) : ctx.moveTo(...pt); }); ctx.stroke(); });
  a.G.forEach((r,j) => { if (a.bld.b[j] === i) return; const [gx,gy] = a.cen[j]; if (Math.hypot(gx-cx,gy-cy) > R) return;
    poly(r, 0); ctx.fillStyle = "#d6d6d6"; ctx.fill(); });
  const rgb = hex(colour);
  ctx.beginPath(); rings.forEach(r => r.forEach(([x,y],q) => { const p = P(x,y); q ? ctx.lineTo(...p) : ctx.moveTo(...p); }));
  ctx.fillStyle = css(mix(colour, .62)); ctx.fill("evenodd"); ctx.strokeStyle = "#1e1e1e"; ctx.lineWidth = 1.3; ctx.stroke();
  const bs = []; a.G.forEach((r,j) => { if (a.bld.b[j] === i) bs.push(j); });
  const depth = j => { const [gx,gy] = a.cen[j]; const dx=gx-cx, dy=gy-cy; return dx*s + dy*c; };
  bs.sort((u,v) => depth(v) - depth(u));
  for (const j of bs) { const r = a.G[j], h = a.bld.h[j]/10, imp = a.bld.imp[j];
    ctx.setLineDash(imp ? [2,1.6] : []); ctx.lineWidth = .55; ctx.strokeStyle = imp ? "#8a8a8a" : "#333";
    const walls = r.map((p,q) => [p, r[(q+1)%r.length]]).map(([p,q]) => { const [ax,ay] = P(p[0],p[1]), [bx,by] = P(q[0],q[1]);
      return {p, q, d: (depth2(p)+depth2(q))/2, face: (bx-ax)}; });
    function depth2(p){ const dx=p[0]-cx, dy=p[1]-cy; return dx*s + dy*c; }
    walls.sort((u,v) => v.d - u.d);
    for (const w of walls) { ctx.beginPath(); [P(...w.p,0),P(...w.q,0),P(...w.q,h),P(...w.p,h)].forEach((p,q) => q ? ctx.lineTo(...p) : ctx.moveTo(...p));
      ctx.closePath(); ctx.fillStyle = imp ? (w.face > 0 ? "#f1f1f1" : "#e4e4e4") : (w.face > 0 ? "#cfcfcf" : "#9d9d9d"); ctx.fill(); ctx.stroke(); }
    poly(r, h); ctx.fillStyle = imp ? "#fbfbfb" : "#f4f4f4"; ctx.fill(); ctx.stroke(); }
  ctx.setLineDash([]); ctx.restore();
}
function typesPage() {
  const mid = st.tmodel, m = D.models[mid], ids = Object.keys(m.names).map(Number).sort((x,y)=>x-y);
  const areas = mid === "pooled" ? D.areas : [AREA[mid]];
  const counts = {}; areas.forEach(a => { counts[a.id] = {}; lab(a, mid).forEach(t => counts[a.id][t] = (counts[a.id][t]||0)+1); });
  let h = `<div class="page"><h2>${esc(label(mid))} — ${m.k} types${m.status!=="rule" ? ' <span class="fb">(fallback, exploratory)</span>' : ""}</h2>
    <p class="lede">${mid === "pooled" ? "One model over all areas: the share of each area's blocks in each type." : "Types of this area's own model."}</p>`;
  if (mid === "pooled") {
    const tot = Object.fromEntries(areas.map(a => [a.id, a.blocks.key.length]));
    const mx = Math.max(...ids.flatMap(t => areas.map(a => (counts[a.id][t]||0)/tot[a.id])));
    h += `<table class="comp"><tr><th class="l" style="color:var(--muted)">Pooled type (blocks)</th>${areas.map(a => `<th>${esc(a.name)}<small>% of ${tot[a.id].toLocaleString("en")} blocks</small></th>`).join("")}</tr>` +
      ids.map(t => { const n = areas.reduce((s,a) => s + (counts[a.id][t]||0), 0);
        return `<tr><td><span class="sw" style="display:inline-block;vertical-align:-2px;background:${m.colours[t]}"></span>&nbsp; ${code(mid,t)} · ${esc(m.names[t])} (${n})</td>` +
        areas.map(a => { const v = 100*(counts[a.id][t]||0)/tot[a.id]; return `<td><div class="bar"><i style="width:${Math.max(2, 130*v/100/mx)}px;background:${m.colours[t]}"></i>${v.toFixed(0)}</div></td>`; }).join("") + "</tr>"; }).join("") + "</table>";
  }
  h += `<h2 style="margin-top:34px">Representative blocks</h2><p class="lede">Blocks closest to each type's median profile. All blocks at one scale; ground colour: type; grey: neighbouring buildings and streets.</p>`;
  const allKeys = ids.flatMap(t => m.reps[t] || []);
  const sizes = allKeys.map(k => { for (const a of D.areas) { const i = a.blocks.key.indexOf(k); if (i >= 0) { let x0=1e9,y0=1e9,x1=-1e9,y1=-1e9;
    a.B[i].flat().forEach(r => r.forEach(([x,y]) => { x0=Math.min(x0,x); x1=Math.max(x1,x); y0=Math.min(y0,y); y1=Math.max(y1,y); })); return Math.hypot(x1-x0, y1-y0)/2; } } return 50; }).sort((x,y)=>x-y);
  const r75 = sizes[Math.floor(sizes.length*.75)] || 60, scale = (420/2 - 30) / r75 * .8;
  for (const t of ids) { const p = m.profiles.find(x => x.type === t) || {};
    const by = mid === "pooled" ? " · " + D.areas.map(a => `${esc(a.id)} ${counts[a.id][t]||0}`).join(" · ") : "";
    h += `<div class="trow"><div class="tx"><div><span class="sw" style="display:inline-block;width:22px;height:22px;vertical-align:-5px;background:${m.colours[t]}"></span>&nbsp;<span class="code">${code(mid,t)}</span></div>
      <div class="nm">${esc(m.names[t])}</div><div class="meta">${p.blocks} blocks${by}<br>height* ${fmt(p.height_measured_m,0)} m · GSI ${fmt(p.gsi,2)} · ${fmt(p.footprint_mean_m2,0)} m²<br>FSI* ${fmt(p.fsi_measured,1)} · ${fmt(p.n_buildings,0)} buildings · ${fmt(p.blk_area_ha,2)} ha</div></div>` +
      (m.reps[t] || []).slice(0,3).map(k => `<div><canvas data-k="${esc(k)}" data-c="${m.colours[t]}"></canvas><div class="lab">${esc(k)}</div></div>`).join("") + "</div>";
  }
  h += `<div class="foot">Buildings: height available before imputation solid; imputed height lighter with dashed outline.<br>* median of block values from heights available before imputation; GSI: building coverage; m²: mean building footprint.</div></div>`;
  $("#pageview").innerHTML = h;
  $("#pageview").querySelectorAll("canvas[data-k]").forEach(cv => { const k = cv.dataset.k, a = D.areas.find(a => a.blocks.key.includes(k)); if (a) axo(cv, a, k, cv.dataset.c, scale); });
}

// ---------------------------------------------------------------- Model quality page
const KC = {sil: "#48cfad", ch: "#5b8fcb", db: "#e8b34b", comb: "#ac92ec"};
function kchart(mid) {
  const m = D.models[mid], s = m.sweep, W = 520, H = 230, L = 18, R = 12, T = 22, B = 30;
  const nm = f => { const v = s.map(f), lo = Math.min(...v), hi = Math.max(...v); return v.map(x => hi > lo ? (x-lo)/(hi-lo) : .5); };
  const ser = {sil: nm(r => r.silhouette), ch: nm(r => r.calinski), db: nm(r => -r.davies), comb: nm(r => r.combined)};
  const ks = s.map(r => r.k), k0 = Math.min(...ks), k1 = Math.max(...ks);
  const X = k => L + (W-L-R)*(k-k0)/Math.max(1,k1-k0), Y = v => T + (H-T-B)*(1-v), bw = (W-L-R)/Math.max(1,k1-k0);
  let g = `<svg viewBox="0 0 ${W} ${H}" font-size="11">`;
  s.forEach(r => { if (!r.size_ok) g += `<rect x="${X(r.k)-bw*.22}" y="${T}" width="${bw*.44}" height="${H-T-B}" fill="#ecedf0"/>`; });
  g += `<line x1="${L}" x2="${W-R}" y1="${H-B}" y2="${H-B}" stroke="#222" stroke-width="1.2"/>` + ks.map(k => `<text x="${X(k)}" y="${H-B+16}" text-anchor="middle" fill="#6b6b6b">${k}</text>`).join("");
  if (m.elbow) g += `<line x1="${X(m.elbow)}" x2="${X(m.elbow)}" y1="${T-6}" y2="${H-B}" stroke="#555" stroke-dasharray="2 3"/><text x="${X(m.elbow)+5}" y="${T+2}" fill="#6b6b6b">elbow K = ${m.elbow}</text>`;
  g += `<line x1="${X(m.k)}" x2="${X(m.k)}" y1="${T-6}" y2="${H-B}" stroke="#333" stroke-width="1.6" stroke-dasharray="6 3"/>` +
    `<text x="${X(m.k)+5}" y="${H-B-6}" fill="${m.status==="rule" ? "#333" : "#c4456f"}">${m.status==="rule" ? "reported K = "+m.k : "K = "+m.k+": fallback"}</text>`;
  const line = (v, c, w) => `<polyline fill="none" stroke="${c}" stroke-width="${w}" stroke-linejoin="round" points="${ks.map((k,i) => X(k)+","+Y(v[i])).join(" ")}"/>`;
  g += line(ser.sil, KC.sil, 1.4) + line(ser.ch, KC.ch, 1.4) + line(ser.db, KC.db, 1.4) + line(ser.comb, KC.comb, 3.6);
  ks.forEach((k,i) => { g += `<circle cx="${X(k)}" cy="${Y(ser.sil[i])}" r="3" fill="${KC.sil}"/><rect x="${X(k)-3}" y="${Y(ser.ch[i])-3}" width="6" height="6" fill="${KC.ch}"/>` +
    `<path d="M${X(k)} ${Y(ser.db[i])-3.6}l3.4 6h-6.8z" fill="${KC.db}"/>` +
    `<circle cx="${X(k)}" cy="${Y(ser.comb[i])}" r="9" fill="transparent"><title>K ${k}: silhouette ${s[i].silhouette.toFixed(3)}, smallest type ${s[i].min_type_size}${s[i].size_ok ? "" : " (< min)"}${s[i].admissible ? ", admissible" : ""}</title></circle>`; });
  g += "</svg>";
  return `<div class="kchart"><h3>${esc(label(mid))}${m.status!=="rule" ? " (exploratory)" : ""}</h3><div class="st">${m.status==="rule" ? "K selected by the rule" : `<span class="fb">${esc(m.reason)}</span>`}</div>${g}</div>`;
}
function qualityPage() {
  let h = `<div class="page"><h2>Models</h2><p class="lede">Whitened PCA, K-means; K chosen by the rule (every type ≥ minimum size and K at or above the inertia elbow; best combined score). A fallback is reported, never presented as a rule result.</p><div class="mcards">`;
  for (const mid of D.order) { const m = D.models[mid], sb = m.stability;
    h += `<div class="mcard"><h3>${esc(label(mid))}</h3><div><span class="k">K = ${m.k}</span> ${m.status==="rule" ? "rule" : '<span class="fb">fallback</span>'}</div>
      <div>${m.blocks.toLocaleString("en")} blocks · PCA variance ${fmt(100*m.pca_var,1)} %</div>
      <div>silhouette ${fmt(m.silhouette,3)} (+${fmt(m.sil_gap,3)} over noise)</div>
      <div>stability ARI ${sb ? `${fmt(sb.median_ari,2)} <span class="small">(IQR ${fmt(sb.q25,2)}–${fmt(sb.q75,2)}, ${sb.reps} refits)</span>` : "–"}</div></div>`; }
  h += `</div><h2>Selection of K</h2><p class="lede">Metrics min–max normalised within each model.</p><div class="grid2">${D.order.map(kchart).join("")}</div>
    <div class="klegend"><span><svg width="26" height="10"><line x1="0" x2="26" y1="5" y2="5" stroke="${KC.sil}" stroke-width="1.6"/><circle cx="13" cy="5" r="3" fill="${KC.sil}"/></svg>silhouette</span>
    <span><svg width="26" height="10"><line x1="0" x2="26" y1="5" y2="5" stroke="${KC.ch}" stroke-width="1.6"/><rect x="10" y="2" width="6" height="6" fill="${KC.ch}"/></svg>Calinski–Harabasz</span>
    <span><svg width="26" height="10"><line x1="0" x2="26" y1="5" y2="5" stroke="${KC.db}" stroke-width="1.6"/><path d="M13 1.5l3.4 6h-6.8z" fill="${KC.db}"/></svg>Davies–Bouldin (inverted)</span>
    <span><svg width="26" height="10"><line x1="0" x2="26" y1="5" y2="5" stroke="${KC.comb}" stroke-width="4"/></svg>combined score (decision)</span>
    <span><svg width="26" height="10"><line x1="0" x2="26" y1="5" y2="5" stroke="#555" stroke-dasharray="2 3"/></svg>inertia elbow</span>
    <span><svg width="26" height="10"><line x1="0" x2="26" y1="5" y2="5" stroke="#333" stroke-width="1.6" stroke-dasharray="6 3"/></svg>reported K</span>
    <span><svg width="14" height="12"><rect width="14" height="12" fill="#ecedf0"/></svg>a type below the minimum size</span></div>`;
  if (hasPooled) {
    const P = D.models.pooled, pids = Object.keys(P.names).map(Number).sort((x,y)=>x-y);
    h += `<h2 style="margin-top:14px">Local types within the pooled types</h2><p class="lede">Blocks of each local type (rows) by pooled type (columns); shading = share of the local type.</p>`;
    for (const a of D.areas) { const L = D.models[a.id]; if (!L) continue;
      const ct = {}; a.blocks.local.forEach((l,i) => { const p = a.blocks.pooled[i]; ct[l] = ct[l] || {}; ct[l][p] = (ct[l][p]||0)+1; });
      const lids = Object.keys(L.names).map(Number).sort((x,y)=>x-y);
      h += `<div class="ctwrap"><table class="ct"><tr><th class="l"><b>${esc(a.name)}</b>${L.status!=="rule" ? ' <span class="fb">(fallback)</span>' : ""}</th>${pids.map(p => `<th title="${esc(P.names[p])}"><span class="sw" style="display:inline-block;vertical-align:-2px;background:${P.colours[p]}"></span> P${p}</th>`).join("")}</tr>` +
        lids.map(l => { const row = ct[l] || {}, tot = Object.values(row).reduce((s,v)=>s+v,0) || 1;
          return `<tr><td class="l"><span class="sw" style="display:inline-block;vertical-align:-2px;background:${L.colours[l]}"></span> <span class="code">${a.id}${l}</span>${esc(L.names[l])}</td>` +
          pids.map(p => { const v = row[p] || 0; return `<td style="background:${v ? css(mix(P.colours[p], .12 + .7*v/tot)) : "transparent"}">${v || ""}</td>`; }).join("") + "</tr>"; }).join("") + "</table></div>"; }
  }
  $("#pageview").innerHTML = h + "</div>";
}

// ---------------------------------------------------------------- render
function render() {
  side(); const map = st.tab === "map";
  $("#mapview").hidden = !map; $("#cards").hidden = !map; $("#pageview").hidden = map;
  if (map) { drawMaps(); cards(); }
  else { PANELS.forEach(p => p.deck && p.deck.finalize()); PANELS = []; $("#mapview").innerHTML = "";
    st.tab === "types" ? typesPage() : qualityPage(); $("#main").scrollTop = 0; }
}
let rt; window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => st.tab === "map" && drawMaps(), 200); });
render();
</script></body></html>
"""
