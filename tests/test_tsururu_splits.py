import numpy as np
import pandas as pd
import pytest

from usd_rub.tsururu_splits import matured, temporal_folds


def test_tsururu_folds_keep_weekly_targets_inside_the_block():
    pytest.importorskip("tsururu")
    dates = pd.date_range("2011-11-17", "2026-11-18")
    folds, start = temporal_folds(dates, "2025-01-01")

    assert dates[start] == pd.Timestamp("2018-01-03")
    assert [fold.warmup for fold in folds] == [True, True, False, False, False, False]
    assert dates[folds[0].cutoff] == pd.Timestamp("2019-01-03")
    assert dates[folds[-1].stop - 1] == pd.Timestamp("2024-12-31")
    for fold in folds:
        assert fold.stop - fold.cutoff == 365
        assert len(fold.origins) == 358
        assert matured(fold.origins, fold.stop).all()
        assert not matured(np.array([fold.origins[-1] + 1]), fold.stop).item()
    for left, right in zip(folds[:-1], folds[1:], strict=True):
        assert left.stop == right.cutoff


def test_maturity_depends_on_each_horizon():
    origins = np.arange(12, 20)
    np.testing.assert_array_equal(origins[matured(origins, 20, 7)], [12])
    np.testing.assert_array_equal(origins[matured(origins, 20, 1)], np.arange(12, 19))
