import json
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_calendar_uses_fixing_day_and_working_saturday(monkeypatch):
    pytest.importorskip("sklearn")
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    analysis = import_module("correlation_holdout")
    calendar = json.loads((root / "config" / "cbr_fixing_calendar.json").read_text())
    dates = pd.DatetimeIndex(["2025-10-31", "2025-11-01", "2025-11-02"])
    changes = analysis.fixing_mask(dates, calendar, True)
    np.testing.assert_array_equal(changes[0, :5], [True, True, False, False, False])
    np.testing.assert_array_equal(changes[1, :4], [True, False, False, False])
    np.testing.assert_array_equal(changes[2, :4], [False, False, False, True])
    forecast = np.tile(np.arange(101.0, 108.0), (3, 1))
    prior = np.array([100.0, 100.0, 100.0])
    pooled = analysis.apply_calendar(forecast, prior, changes, True)
    carried = analysis.apply_calendar(forecast, prior, changes, False)
    np.testing.assert_array_equal(pooled[0, :5], [101, 103.5, 103.5, 103.5, 103.5])
    np.testing.assert_array_equal(carried[0, :5], [101, 102, 102, 102, 102])
    np.testing.assert_array_equal(pooled[2, :3], [100, 100, 100])


def test_calendar_transfer_is_used_after_availability(monkeypatch):
    pytest.importorskip("sklearn")
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    analysis = import_module("correlation_holdout")
    calendar = {
        "fixed_days_off": [],
        "years": {"2025": {"available_from": "2025-10-31", "days_off": [], "workdays": ["11-01"]}},
    }
    dates = pd.DatetimeIndex(["2025-10-30", "2025-10-31"])
    changes = analysis.fixing_mask(dates, calendar, True)
    assert not changes[0, 2]
    assert changes[1, 1]
