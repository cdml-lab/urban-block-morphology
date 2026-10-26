import numpy as np
import pandas as pd
import pytest

from blockmorph.descriptors import FULL57, MORPHO_COLUMNS, TYPOLOGY_COLUMNS
from blockmorph.inputs import choose_crs
from blockmorph.model import pca_space, select_k
from blockmorph.params import Params


def test_descriptor_sets():
    assert len(FULL57) == 57 and len(MORPHO_COLUMNS) == 46
    assert set(FULL57) - set(MORPHO_COLUMNS) == set(TYPOLOGY_COLUMNS) | {"rcnn_labelled"}


def test_crs_must_be_metric():
    with pytest.raises(ValueError):
        choose_crs([], "EPSG:4326")
    assert choose_crs([], "EPSG:2039") == "EPSG:2039"


def _blobs(sizes, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(i * 6, 1, size=(n, 8)) for i, n in enumerate(sizes)])
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])


def test_rule_selection():
    f = _blobs([60, 60, 60, 60, 60, 60])
    z, _ = pca_space(f, Params())
    labels, sweep, st = select_k(z, Params())
    assert st["status"] in ("rule", "fallback")
    assert st["selected_k"] == len(np.unique(labels))
    assert sweep["selected"].sum() == 1


def test_fallback_is_reported():
    # too few blocks for any K >= 5 with 12 blocks per type and an elbow
    f = _blobs([8, 8, 8, 8, 8, 8, 8])
    p = Params(k_range=tuple(range(5, 9)))
    z, _ = pca_space(f, p)
    _, sweep, st = select_k(z, p)
    assert not sweep["admissible"].any()
    assert st["status"] == "fallback" and "relaxed" in st["reason"]


def test_missing_shares_not_zero():
    f = _blobs([40, 40]).iloc[:, :3]
    f["rcnn_x"] = np.where(np.arange(80) < 10, np.nan, 0.5)
    z, _ = pca_space(f, Params(pca_components=2))
    assert np.isfinite(z).all()
