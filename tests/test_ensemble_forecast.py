from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from usd_rub import ensemble_forecast
from usd_rub.ensemble_forecast import blend_predictions
from usd_rub.market_correction import (
    COMPACT_MACRO_BUNDLE,
    MACRO_BUNDLE,
    MARKET_FEATURES,
    load_correction_bundle,
    load_market_bundle,
    market_forecast,
)
from usd_rub.theory_features import (
    COMPACT_FUNDAMENTAL_SIGNS,
    COMPACT_MACRO_FEATURES,
    FUNDAMENTAL_SIGNS,
    MACRO_CORRECTION_FEATURES,
)


def test_default_forecast_uses_classical_anchor_and_half_correction(monkeypatch, tmp_path):
    pytest.importorskip("threadpoolctl")
    dates = pd.date_range(end="2026-11-16", periods=20)
    target = pd.Series(100.0, index=dates)
    panel = pd.DataFrame(0.0, index=dates, columns=COMPACT_MACRO_FEATURES)
    calendar = tmp_path / "calendar.json"
    calendar.write_text('{"fixed_days_off": [], "years": {}}')
    args = SimpleNamespace(
        base_only=False,
        classical_only=False,
        market_only=False,
        legacy_t0_gamma=False,
        checkpoint=None,
        train_before=None,
        calendar=calendar,
    )

    class FixedCorrection:
        def inplace_predict(self, features):
            assert features.columns.tolist() == [*COMPACT_MACRO_FEATURES, "anchor_return"]
            return np.full(len(features), 100 * np.log(1.04))

    def load_bundle(bundle, columns, signs):
        assert (bundle, columns, signs) == (
            COMPACT_MACRO_BUNDLE,
            COMPACT_MACRO_FEATURES,
            COMPACT_FUNDAMENTAL_SIGNS,
        )
        return [FixedCorrection()] * 7, {"trained_before": "2025-01-01"}

    def legacy_prediction(*args):
        pytest.fail("Основная модель загрузила t0-gamma")

    monkeypatch.setattr(
        ensemble_forecast,
        "forecast_inputs",
        lambda *args, **kwargs: (target, {"extended": panel}, []),
    )
    monkeypatch.setattr(
        ensemble_forecast, "classical_predictions", lambda *args: np.full((1, 7), 100.0)
    )
    monkeypatch.setattr(ensemble_forecast, "legacy_prediction", legacy_prediction)
    monkeypatch.setattr(ensemble_forecast, "load_correction_bundle", load_bundle)

    forecast, metadata = ensemble_forecast.run_forecast(args, dates[-1])

    np.testing.assert_allclose(forecast["prediction"], 102.0)
    assert metadata["model"] == "ensemble_classical_xgb_shap8"
    assert metadata["macro_correction"]["fraction"] == 0.5


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


def test_packaged_macro_correction_respects_fundamental_signs():
    pytest.importorskip("xgboost")

    models, metadata = load_correction_bundle(
        MACRO_BUNDLE, MACRO_CORRECTION_FEATURES, FUNDAMENTAL_SIGNS
    )
    columns = [*MACRO_CORRECTION_FEATURES, "anchor_return"]
    features = pd.DataFrame(0.0, index=range(41), columns=columns)
    for key, sign in FUNDAMENTAL_SIGNS.items():
        values = np.linspace(-20, 20, len(features))
        varied = features.assign(**{key: values})
        for model in models:
            assert np.all(sign * np.diff(model.inplace_predict(varied)) >= 0)

    assert len(models) == 7
    assert metadata["trained_before"] == "2025-01-01"


def test_classical_ensemble_excludes_features_of_macro_correction(monkeypatch):
    def fit(x, levels, candidate, split, origins, horizons, *, min_hit_rate):
        assert x.shape == (10, 1)
        return np.zeros((len(origins), len(horizons))), np.array([0])

    monkeypatch.setattr(ensemble_forecast, "fit_predict", fit)
    panel = pd.DataFrame({"own__return_1": np.ones(10), "fx__theory_ppp_value_gap": np.ones(10)})

    prediction = ensemble_forecast.classical_predictions(
        {"strict": panel}, np.full(10, 100.0), 8, np.array([9])
    )

    np.testing.assert_allclose(prediction, np.full((1, 7), 100.0))
