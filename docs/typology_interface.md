# Building-typology predictions (interface to building-typology-rcnn)

**Pinned version:** `building-typology-rcnn` **v1.0.0** (Git tag `v1.0.0`),
output schema **1.0**, checkpoint `fasterrcnnv100-12.pth` (SHA-256
`87727651374c5ae9e2b652f4c01487d1565b9971bff670044b86733604c1dc78`).

## What is read
A polygon layer of building footprints, in any vector format or
GeoParquet and any CRS, with a column **`class_name`**. Each value is one
of the 10 analysis categories or `None`:

`Carpet Houses, Geometric Building, Geometric Shape, H Shape, L Shape,
Long Rectangle, Private House_A, Private House_B, Rectangle Building, U Shape`

The network has 11 classes. *Plus Shape* is merged into *H Shape*, and that
merge is done by the typology repository (`class_name`), not here.

The footprints need not be the same polygons as the `buildings` input.
Each labelled footprint is placed in a block by its representative point.

## Block descriptors derived
- `rcnn_<class>` (10 columns): the share of each category among the
  block's footprints with a predicted category.
- `rcnn_labelled`: the number of such footprints.

## Missing predictions: never zeros
- **A block without any classified building** gets `rcnn_labelled = 0` and
  NaN shares, not an all-zero composition. In the model these NaNs take the
  column median, like every other missing descriptor (study procedure).
  The log reports how many blocks this affects.
- **An area without predictions** cannot run the 57-descriptor model. The
  run stops and asks you to choose between two options:
  - supply the predictions;
  - set `descriptors: morphometric46`. That is a **morphometric-only model**
    with the 46 morphometric and contextual descriptors and no typology
    columns at all. It is a different model from the manuscript's. `run.json`
    (`descriptor_set`, `descriptor_note`) and a banner in the dashboard state
    this.

Mixing is not allowed. All areas of one run use the same descriptor set,
so a pooled model never combines blocks with and without typology
information.
