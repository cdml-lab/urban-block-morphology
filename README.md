# urban-block-morphology

Urban-block typologies from building footprints and streets: block
delineation, 57 block descriptors, an explicit missing-data treatment,
whitened PCA, rule-based selection of the number of types (K), K-means,
physical type profiles and an HTML results dashboard. It runs on one area
(local model) or on several areas together (pooled model), or both.

This is the code of the block-morphology analysis (Track A) of the
manuscript. With the study inputs it reproduces the published partitions
exactly (`docs/reproduction.md`; exact study settings, including the road
categories per city: `docs/study_configuration.md`).

## Results dashboard

The study's dashboard (maps in 2D/3D, type profiles, representative blocks, model quality):
**[view in the browser](https://cdml-lab.github.io/urban-block-morphology/)** · [download v1.0.0](https://github.com/cdml-lab/urban-block-morphology/releases/download/v1.0.0/urban-block-types_dashboard.html) · `dashboard/`.
It needs an internet connection and WebGL 2 for the 3D map (`dashboard/README.md`).

## Install

```bash
conda env create -f environment.yml
conda activate urban-block-morphology
# or: python -m venv .venv && pip install -e .
```

Exact versions of the verification run: `requirements-lock.txt`.

## Run the example

```bash
blockmorph run examples/example_config.yaml
# -> examples/example_output/  (tables, GeoPackages, dashboard.html)
blockmorph run examples/example_config_morphometric.yaml   # without typology predictions
```

The example uses three small sub-areas (City A, B, C) of the study areas,
about 140 to 190 blocks each (`examples/data/README.md`). With so few
blocks, the local models of the example are fallback partitions; the
pooled model is selected by the rule. This is how the dashboard reports
both cases.

## Your own areas

1. One boundary per area (any vector format, any CRS). You choose the area
   ids and display names.
2. Buildings: `osm` (downloaded) or your own footprints with optional
   `height_m` / `floors`. Optional extra height layers (e.g. 3D-GloBFP).
3. Streets: `osm` or your own centrelines with an OSM `highway` class.
4. Building-typology predictions (for the 57-descriptor model): the
   output of [`building-typology-rcnn`](https://github.com/cdml-lab/building-typology-rcnn)
   **v1.0.0** (`assign` command, output schema 1.0, column `class_name`).
   Without them, set `descriptors: morphometric46`. This is a different,
   morphometric-only model and is labelled as such in every output. No
   typology share is ever filled with zeros (`docs/typology_interface.md`).
5. CRS: `auto` picks the UTM zone of your areas, or you can give any
   projected CRS in metres.

See `docs/inputs.md` for the full configuration reference.

## What you get

| output | content |
|---|---|
| `assignments.csv` | `block_key`, area, local type and name, pooled type and name |
| `descriptors.csv` | the model input (57 or 46 columns) per block |
| `block_stats.csv` | physical statistics per block (heights before imputation) |
| `areas/<id>/*.gpkg` | blocks, buildings (with imputation flag), streets, boundary |
| `models/<model>/report.json` | K, **rule or fallback** with the reason, quality, stability |
| `models/<model>/k_selection.csv` | K sweep with the selection conditions |
| `models/<model>/profiles.csv` | type profiles; `representatives.json` |
| `run.json` | configuration, CRS, descriptor set, library versions |
| `dashboard.html` | maps, profiles, representative blocks (2D / 3D), distributions, quality |

`docs/outputs.md` documents every column.

## K selection, reported honestly

K is chosen for *your* data, from the range 5–15. The rule is that every
type holds at least 12 blocks, K is at or above the inertia elbow, and,
among those K, the best combined quality score wins. When no K meets both
conditions, the partition with the best score after relaxing one condition
is produced. It is marked `fallback` with the reason in `report.json`,
the log and the dashboard, and should be treated as exploratory. See
`docs/method.md`.

## Licence and citation

Code: MIT (`LICENSE`). Example data: OpenStreetMap contributors (ODbL),
3D-GloBFP heights (CC BY 4.0) and building-form predictions
(`examples/data/README.md`). Please cite with `CITATION.cff`.
