# Outputs

All tables are UTF-8 CSV. All layers are GeoPackage in the analysis CRS
(`run.json` → `crs`). Blocks are identified everywhere by **`block_key`**
= `<area id>-<block_id>`.

## `assignments.csv`
| column | meaning |
|---|---|
| block_key, area | block and area id |
| local_type, local_name | type id (0..K-1) and name in the area's own model |
| pooled_type, pooled_name | type id and name in the pooled model (≥ 2 areas) |

## `descriptors.csv`
The model input, one row per block: 57 columns (`full57`) or 46
(`morphometric46`), defined in `descriptors.csv` of this folder. Values
come after height imputation and before the model preprocessing. NaN means
missing (for example typology shares of a block without classified
buildings).

## `block_stats.csv`
Physical statistics for profiles and the dashboard. Heights are those
known before imputation.

| column | meaning |
|---|---|
| blk_area_ha, n_buildings, gsi | block area (ha), buildings, coverage |
| fsi_measured | GSI × footprint-weighted storeys of buildings with a known height |
| height_measured_m, height_p90_m | footprint-weighted mean and 90th percentile of known heights |
| footprint_mean_m2, spacing_m, shared_walls, edge_share | mean footprint, mean distance to 5 nearest buildings, shared-wall ratio, street-edge share |
| heights_measured | share of the block's buildings with a known height |

## `areas/<id>/`
- `blocks.gpkg`: block_id, block_key, area, geometry.
- `buildings.gpkg`: block_id, block_key (empty outside blocks), height_m
  (after imputation), height_measured_m, height_imputed (0/1),
  footprint_m2, and class_name with the full57 descriptor set.
- `streets.gpkg`: the block-cutting streets used.
- `boundary.gpkg`.

## `models/<model>/` (`<model>` = area id or `pooled`)
- `report.json`:
  - `selected_k`, `status` (`rule` | `fallback`), `reason`, `elbow_k`,
    `admissible_k`;
  - `pca_components`, `pca_variance`;
  - `silhouette`, `silhouette_gap`, `calinski`, `davies`;
  - `stability` {reps, median_ari, q25, q75};
  - `type_order` (Ward order of type centroids, used for colours and tables);
  - `names`, `names_approved`.
- `k_selection.csv`: one row per K, with the columns described in `method.md`
  and `selected`.
- `profiles.csv`: type, name, blocks, `blocks_<area>`, median profile
  columns (as `block_stats.csv`), heights_measured (mean), order.
- `representatives.json`: {type: [block_key, ...]}.

## `run.json`
Package version, Python and library versions, CRS, descriptor set and
note, the list of descriptors, areas (id, name, cut_all_roads), height
imputation counts per area, all parameters, and per model `selected_k`,
`status` and `reason`.

## `dashboard.html`
A self-contained page with no external resources, built only from the
files above (`blockmorph dashboard <output_dir>`). It shows:
- model cards with rule/fallback status;
- maps for the local and pooled models;
- type profiles;
- representative blocks as 2D plans, with an optional 3D view on a canvas
  (falls back to 2D);
- pooled types by area and local × pooled cross-tabulations;
- K-selection charts.
