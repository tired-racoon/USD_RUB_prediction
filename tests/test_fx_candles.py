import numpy as np
import pandas as pd

from usd_rub.fx_candles import session_prices


def test_candles_precede_cutoff_and_fill_only_missing_sessions():
    dates = pd.date_range("2025-01-02", "2025-01-10")
    begin = pd.to_datetime(
        ["2025-01-02 12:00", "2025-01-02 22:00", "2025-01-02 23:00", "2025-01-04 17:00"]
    )
    close = [80.0, 82.0, 90.0, 83.0]
    bars = pd.DataFrame(
        {
            "instrument": "USDRUBF",
            "begin": begin,
            "end": begin + pd.Timedelta(minutes=59, seconds=59),
            "open": close,
            "close": close,
            "high": close,
            "low": close,
            "volume": 10,
        }
    )

    result = session_prices(bars, "USDRUBF", dates)

    assert np.isnan(result.loc["2025-01-02", "close"])
    assert result.loc["2025-01-03", "close"] == 82
    assert result.loc["2025-01-03", "morning"] == 80
    assert result.loc["2025-01-04", "close"] == 82
    assert result.loc["2025-01-04", "age"] == 2
    assert result.loc["2025-01-05", "close"] == 83
    assert np.isnan(result.loc["2025-01-05", "morning"])
    assert np.isnan(result.loc["2025-01-09", "close"])
