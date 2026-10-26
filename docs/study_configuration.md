# Exact configuration of the study (Cities A–C)

This is the configuration behind the manuscript's results. It is encoded in
`configs/study_reproduction.yaml`.

## Inputs, shared by all three cities
- **Buildings:** OpenStreetMap footprints; heights from the OSM `height`
  tag, else 3D-GloBFP, else floors × 3.2 m.
- **Streets:** OpenStreetMap, fetched with
  `osmnx.graph_from_polygon(network_type="all", retain_all=True,
  truncate_by_edge=True)`. This returns every way with a `highway` tag
  except those osmnx excludes for `"all"`:
  - `area=yes`;
  - `highway=abandoned`, `construction`, `no`, `planned`, `platform`,
    `proposed`, `raceway` or `razed`;
  - `service=private`.
- **Analysis CRS:** EPSG:2039 (Israeli TM Grid).
- **Study area:** city boundary ∩ imagery extent.
- **Building-form predictions:** `building-typology-rcnn` v1.0.0, checkpoint
  `fasterrcnnv100-12.pth`. Inference used confidence 0.3, NMS 0.5 per
  window and 0.2 globally, and **edge windows resized** (`--edge-mode
  resize`, see that repository's `configs/study.yaml`). The repository
  default for new work is `pad`. Assignment to footprints uses 20 %
  coverage.

## Road categories that cut blocks
Only the filter applied to the fetched ways differs between cities.

| city | rule | OSM `highway` values that cut blocks |
|---|---|---|
| City A | default cutting set (`cut_all_roads: false`) | motorway, motorway_link, trunk, trunk_link, primary, primary_link, secondary, secondary_link, tertiary, tertiary_link, residential, living_street, pedestrian, footway |
| City B | default cutting set (`cut_all_roads: false`) | the same 14 values |
| City C | **all mapped ways** (`cut_all_roads: true`) | every value in the fetched network. In the study data (9,387 segments) these were: service 2,767; residential 2,363; footway 1,936; tertiary 956; secondary 511; track 140; cycleway 124; pedestrian 96; trunk 86; path 82; unclassified 79; living_street 66; trunk_link 50; secondary_link 39; primary 30; motorway_link 19; primary_link 17; tertiary_link 14; motorway 6; corridor 4; steps 2 |

So in City C, beyond the default set, **service, track, cycleway, path,
unclassified, corridor and steps** ways also cut blocks. This is a per-city
rule carried over from the earlier pipeline. Its documented reason is that
City C's OSM data tags many ordinary streets as `service`. The block counts were 1,590 (A),
1,666 (B) and 594 (C).

The default set excludes service, track, path, cycleway, unclassified,
steps and corridor ways in Cities A and B.

## Blocks, descriptors and model
All parameters are the defaults of `params.py`:
- **Blocks:** GSI ≥ 0.03, area ≥ 1,500 m², at least one building.
- **Model:** 57 descriptors and the explicit missing-data treatment;
  whitened PCA (5); K-means with `n_init` 10 and seed 42; K from 5 to 15,
  minimum type size 12, elbow floor.
- **Pooling order:** City A, City C, City B (row order of the pooled
  model).

| model | K | selection |
|---|---|---|
| pooled | 7 | rule |
| City A | 10 | rule |
| City B | 8 | rule |
| City C | 6 | **fallback, exploratory**: no K met both conditions; the elbow condition was relaxed |

City C's whole local partition (all six types, C0–C5) is exploratory.
Its blocks are typed by the pooled model in the comparative results.
