"""Analysis parameters. The defaults are the values of the study (manuscript
model); change them only knowingly - the K-selection rule, the descriptor
set and the missing-data treatment define the published procedure."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Params:
    # buildings
    min_building_area_m2: float = 25.0       # smaller footprints are dropped
    floor_height_m: float = 3.2              # floors <-> height conversion
    max_height_m: float = 260.0              # heights outside 2 m .. this are discarded
    # blocks
    cutting_highways: tuple = ("motorway", "trunk", "primary", "secondary", "tertiary",
                               "motorway_link", "trunk_link", "primary_link",
                               "secondary_link", "tertiary_link", "residential",
                               "living_street", "pedestrian", "footway")
    min_block_area_m2: float = 1500.0        # smaller enclosures are not blocks
    min_block_gsi: float = 0.03              # enclosures with lower coverage are open space
    # descriptors
    edge_band_m: float = 15.0                # building centroid this close to the block edge = edge building
    street_search_m: float = 120.0           # nearest-street search radius
    context_buffer_m: float = 25.0           # neighbouring-block contiguity buffer
    # missing data
    impute_knn: int = 10                     # rule 2: median of the k nearest known heights
    # clustering
    pca_components: int = 5
    pca_whiten: bool = True
    k_range: tuple = tuple(range(5, 16))
    min_type_size: int = 12                  # every type must hold at least this many blocks
    kmeans_n_init: int = 10
    random_state: int = 42
    # stability (subsample refits; the study used 100)
    stability_reps: int = 20
    stability_fraction: float = 0.8
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict | None) -> "Params":
        d = dict(d or {})
        for k in ("cutting_highways", "k_range"):
            if k in d and d[k] is not None:
                v = d[k]
                if isinstance(v, dict):                       # {from: 5, to: 15}
                    v = range(int(v["from"]), int(v["to"]) + 1)
                d[k] = tuple(v)
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)
