import numpy as np

from usd_rub.ensemble_forecast import blend_predictions


def test_blend_applies_weights_before_pooling_a_shared_fixing():
    classical = np.arange(101.0, 108.0)[None]
    gamma = classical + 20
    changes = np.array([[False, True, False, False, True, False, True]])

    forecast = blend_predictions(classical, gamma, np.array([100.0]), changes)

    np.testing.assert_allclose(forecast, [[100, 104, 104, 104, 108.5, 108.5, 110]])
