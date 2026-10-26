# Method (study procedure, defaults in `params.py`)

## 1. Blocks
1. Enclosures: `momepy.enclosures(streets, limit=study_area, clip=True)`.
2. Each building is placed in an enclosure by its representative point.
3. An enclosure is a **block** when it meets all three conditions:
   - building coverage (GSI) ≥ 0.03;
   - area ≥ 1,500 m²;
   - it holds at least one building.

   Other enclosures are open space and are not classified.
4. Enclosed tessellation (`momepy.enclosed_tessellation`) gives one cell
   per building, used for the coverage area ratio.
5. `block_id` numbers blocks in enclosure order. `block_key` =
   `<area id>-<block_id>`.

## 2. Descriptors
There are 57 per block (`docs/descriptors.csv`):
- 36 shape, intensity, height, grain and configuration descriptors;
- 10 building-typology shares and 1 classified-building count;
- 4 street-interface and 6 neighbouring-block context descriptors.

Building-level characters come from momepy 1.0 with libpysal graphs:
fuzzy contiguity at 0.25 m and 5 nearest neighbours. Context is the mean
over blocks contiguous within 25 m.

## 3. Missing data (explicit treatment)
- **Height.** A building without a height takes the median known height of
  its own block, else the median of its 10 nearest buildings with a known
  height. Floors fall back to round(height / 3.2), and floor area and
  volume are recomputed. `buildings.gpkg` keeps `height_measured_m` and
  `height_imputed`, and `run.json` the imputation counts.
- **Typology.** No classified building in a block gives NaN shares (see
  `typology_interface.md`).
- **Profiles** use only heights known before imputation
  (`block_stats.csv`).

## 4. Model
1. Per descriptor:
   - fill missing values with the median;
   - apply log1p when the column is non-negative and |skew| > 1.5;
   - winsorise at 0.5 / 99.5 %;
   - standardise.

   All descriptors are equally weighted.
2. PCA, 5 components, whitened (`random_state` 42).
3. K-means for K = 5..15 (`n_init` 10, `random_state` 42).

The same procedure is fitted separately for each area (**local**) and once
for all areas together (**pooled**). The descriptors are identical in both
cases.

## 5. K selection
For each K the sweep (`k_selection.csv`) records:
- silhouette, plus the silhouette of K-means on Gaussian noise of the same
  size (`silhouette_gap`);
- Calinski–Harabasz and Davies–Bouldin;
- inertia and the smallest type.

`combined` is the mean of the min–max-normalised silhouette and
Calinski–Harabasz scores and the inverted, normalised Davies–Bouldin score.

A K is **admissible** when both conditions hold:
- every type holds ≥ 12 blocks (`size_ok`);
- K ≥ the inertia elbow (`kneed`, convex and decreasing; `above_elbow`).

**Rule:** the admissible K with the highest `combined` score.

**Fallback (reported):** if no K is admissible, one condition is relaxed:
- the elbow condition, when some K satisfies the size condition;
- otherwise the size condition (best `combined` at or above the elbow).

The result has `status: fallback` with the reason. It is flagged in the
log, `report.json`, `run.json` and the dashboard, and should be treated as
exploratory. In the study, City C had no admissible K and its K = 6 local
partition is such a fallback.

## 6. Profiles, representatives, names, stability
- **Profiles** are medians over each type's blocks of: block area,
  buildings, GSI, FSI and height from measured heights, 90th-percentile
  height, mean footprint, spacing, shared walls and street-edge share.
- **Representatives** are the three blocks nearest the type's median
  profile in robust z-units. They are drawn from blocks with ≥ 80 %
  measured heights and an area inside the type's 10–90 % range, with one
  per area holding ≥ 10 % of the type.
- **Names:** see `inputs.md`.
- **Stability:** the median pairwise ARI of refits on random 80 %
  subsamples. Preprocessing, PCA and K-means are all refitted, at the
  selected K. The default is 20 refits, while the study used 100
  (`parameters: {stability_reps: 100}`).
