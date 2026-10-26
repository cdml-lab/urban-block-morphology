# Inputs and configuration

A run is described by one YAML file. Relative paths are resolved from the
file's folder.

```yaml
crs: auto                  # or any projected CRS in metres, e.g. EPSG:2039, EPSG:25832
descriptors: full57        # full57 (needs typology_predictions) | morphometric46
output_dir: outputs
names: my_names.json       # optional: your own type names (see below)
analysis: {local: true, pooled: true}   # pooled needs >= 2 areas
parameters: {}             # optional overrides of docs/method.md defaults
areas:
  - id: A                  # short id, used in block keys (A-0, A-1, ...)
    name: City A           # display name in tables and the dashboard
    boundary: data/A/boundary.gpkg
    buildings: osm         # or a vector file
    heights:               # optional extra height layers, in this order
      - {path: globfp/tile.shp, column: Height}
    streets: osm           # or a vector file with a 'highway' column
    cut_all_roads: false   # true: every mapped way cuts blocks
    typology_predictions: data/A/footprints_typology.gpkg
```

## Boundary
Any vector file (GeoPackage, GeoJSON, shapefile...) with a CRS. All
polygons are merged into one study area. Blocks are clipped to it.

## Buildings
`osm` downloads `building=*` polygons through OSMnx. Roofs, carports, sheds,
garages, greenhouses, huts, tents, ruins and construction sites are dropped.
Heights come from the `height` tag and floors from `building:levels`.

A file can be any vector format or GeoParquet. Optional columns are
`height_m` (or `height`) in metres and `floors` (or `levels`). The cleaning
steps are:
- make geometries valid and keep the largest part;
- drop footprints under 25 m²;
- for stacked duplicates, keep the tallest;
- keep heights only between 2 and 260 m.

Heights are taken in this order:
1. the source's own height;
2. the `heights` layers, joined by representative point;
3. floors × 3.2 m.

Anything still missing is imputed later (`method.md`). The input heights
are kept as `height_measured_m`.

## Streets
`osm` uses `graph_from_polygon(network_type="all", retain_all=True,
truncate_by_edge=True)`. A file needs centrelines and an OSM-style
`highway` value. By default only these classes cut blocks: motorway, trunk,
primary, secondary, tertiary (and their links), residential, living_street,
pedestrian and footway. `cut_all_roads: true` makes every mapped way cut,
including service, track, path, cycleway, unclassified, steps and corridor.
The study used it for City C (`study_configuration.md`).

## Building-typology predictions
Footprints labelled by `building-typology-rcnn` v1.0.0 (`assign`), output
schema 1.0. Only the `class_name` column is read. See
`typology_interface.md`.

## CRS
Every layer is reprojected to one analysis CRS. `auto` uses the UTM zone
estimated from the union of all areas (`GeoSeries.estimate_utm_crs`). For
a local projected grid (as the study did with EPSG:2039, the Israeli TM
Grid), give it explicitly. Descriptors are metric, so a geographic CRS
(degrees) is not accepted as the analysis CRS.

## Names
Types are named automatically from their profiles, using a fixed
vocabulary of height, coverage, grain, block size and arrangement. Treat
these names as drafts. To publish your own names, give a JSON file:
`{"A": {"0": "Courtyard blocks", ...}, "pooled": {"3": "..."}}`. Type ids are
those in `assignments.csv`. `report.json` records which names were
supplied (`names_approved`).

## Parameters
Any field of `blockmorph.params.Params` can be overridden under
`parameters:`. The defaults are the study's values. Changing them changes
the procedure, so `run.json` records the values used.
