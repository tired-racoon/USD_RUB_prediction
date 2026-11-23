from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_shap_rank_excludes_future_observations_and_stale_features(monkeypatch):
    pytest.importorskip("xgboost")
    pytest.importorskip("sklearn")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    rank = import_module("shap_holdout").rank

    rng = np.random.default_rng(191)
    cutoff = 480
    prices = 80 * np.exp(np.cumsum(rng.normal(0, 0.01, 600)))
    panel = pd.DataFrame(rng.normal(size=(600, 3)), columns=["basis", "returns", "obsolete"])
    panel.loc[cutoff - 120 : cutoff - 1, "obsolete"] = np.nan

    initial = rank(panel, prices, cutoff)
    panel.loc[cutoff:, :] *= 500
    prices[cutoff:] *= 2

    assert set(initial.index) == {"basis", "returns"}
    pd.testing.assert_frame_equal(initial, rank(panel, prices, cutoff))
