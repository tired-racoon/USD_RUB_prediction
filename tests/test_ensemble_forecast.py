import numpy as np
import pandas as pd
import pytest

from usd_rub.ensemble_forecast import blend_predictions
from usd_rub.market_correction import MARKET_FEATURES, load_market_bundle, market_forecast


def test_blend_applies_weights_before_pooling_a_shared_fixing():
    classical = np.arange(101.0, 108.0)[None]
    gamma = classical + 20
    changes = np.array([[False, True, False, False, True, False, True]])

    forecast = blend_predictions(classical, gamma, np.array([100.0]), changes)

    np.testing.assert_allclose(forecast, [[100, 104, 104, 104, 108.5, 108.5, 110]])


def test_market_forecast_adds_a_quarter_of_the_level_correction():
    class FixedCorrection:
        def inplace_predict(self, features):
            assert features.columns.tolist() == [*MARKET_FEATURES, "anchor_return"]
            return np.full(len(features), 100 * np.log(1.04))

    panel = pd.DataFrame([[0.0] * len(MARKET_FEATURES)], columns=MARKET_FEATURES)
    baseline = np.full((1, 7), 100.0)
    changes = np.array([[False, False, True, True, True, True, True]])
    forecast = market_forecast([FixedCorrection()] * 7, panel, baseline, np.array([100.0]), changes)

    np.testing.assert_allclose(forecast, [[100, 100, 101, 101, 101, 101, 101]])


def test_packaged_market_bundle_loads():
    pytest.importorskip("xgboost")

    models, metadata = load_market_bundle()

    assert len(models) == 7
    assert metadata["calibration_origins"] == 1433
    assert metadata["trained_before"] == "2025-01-01"
