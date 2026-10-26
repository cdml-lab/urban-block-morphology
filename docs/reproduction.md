# Reproducing the manuscript's results

The saved results are in the results archive v1.0.0. Its `inputs/` folder
holds the buildings and streets of Cities A–C, and `samples/` a sample of the
building-form predictions. The complete predictions and the study-area
polygons are available on request (open an issue at https://github.com/cdml-lab/building-typology-rcnn/issues (or contact the corresponding author of the article); requests are subject to approval). With them placed as
`inputs/<A|B|C>/typology_predictions.gpkg` and `inputs/<A|B|C>/study_area.gpkg`,
the commands below reproduce the results.

```bash
# unpack the package next to this repository as ../study-results, then
blockmorph run configs/study_reproduction.yaml
python scripts/check_reproduction.py study_run ../study-results
```

## Tolerances
| quantity | tolerance | verification run (2026-10-26) |
|---|---|---|
| blocks per area | exact | 1,590 / 1,666 / 594 |
| block keys | identical | identical |
| K and selection status | exact | pooled 7 (rule); A 10 (rule); B 8 (rule); C 6 (fallback) |
| type ids, local and pooled | identical (ARI = 1.0) | identical, ARI 1.0000 for all four models |
| descriptors | ≤ 1e-4 of each column's range | max 8.4e-6 (`car_mean`) |
| subsample stability (100 refits) | exact for the same library versions | pooled median ARI 0.9213, as published |

`car_mean` depends on the enclosed tessellation, whose cell geometry
varies at about 1e-5 between runs. That variation does not reach the
partitions.

## Conditions for exact partitions
- Use the building-form predictions shipped in the data package
  (`inputs/*/typology_predictions.gpkg`). To regenerate them from imagery
  with `building-typology-rcnn` v1.0.0, run
  `buildingtypology infer <ortho.tif> --checkpoint fasterrcnnv100-12.pth --edge-mode resize --out detections.gpkg`
  and then `buildingtypology assign`. `resize` is the study setting for
  edge windows; the repository default `pad` gives slightly different boxes
  along image borders.
- Use the same row order. Areas are pooled in the study's order (A, C, B),
  as in `configs/study_reproduction.yaml`, because K-means initialisation
  depends on row order.
- Use the same projected CRS (EPSG:2039).
- City C uses `cut_all_roads: true`.
- Keep the default parameters (`params.py`).

Library versions are in `requirements-lock.txt`. A clean environment with
newer releases gave the same selected K on the example data, but exact
identity of the study partitions is guaranteed only with the locked
versions.
