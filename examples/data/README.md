# Example data

These are three rectangular sub-areas (1.4–1.5 km wide) of the study areas
City A, City B and City C, in EPSG:2039 (Israeli TM Grid). They exist only
to run and test the code. They are not a sample of the study's results.

| file | content | source / licence |
|---|---|---|
| `boundary.gpkg` | the rectangle | generated |
| `buildings.gpkg` | footprints, `height_m`, `floors` | footprints © OpenStreetMap contributors (ODbL 1.0); heights from 3D-GloBFP (Che et al. 2024, CC BY 4.0), else OSM tags |
| `streets.gpkg` | centrelines with OSM `highway` | © OpenStreetMap contributors (ODbL 1.0) |
| `typology_predictions.gpkg` | OSM footprints with `class_name` (schema 1.0) | predictions of building-typology-rcnn v1.0.0 on aerial imagery (Esri World Imagery); a sample (these three sub-areas); the imagery itself is not included |

City C uses `cut_all_roads: true`, the study's rule for that area.

Distribution under ODbL requires attribution ("© OpenStreetMap
contributors") and share-alike for derived databases.
