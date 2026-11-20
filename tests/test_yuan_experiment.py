from io import BytesIO
from zipfile import ZipFile

import pandas as pd
import pytest

from usd_rub.storage import Payload, observations
from usd_rub.yuan_experiment import align_inputs, ecb_usd_cny


def test_cross_uses_ecb_quote_only_after_publication_day():
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr(
            "eurofxref-hist.csv",
            "Date,USD,CNY\n2024-01-01,1.25,8.75\n2024-01-02,1.25,10.00\n",
        )
    payload = Payload(buffer.getvalue(), "https://example.com", "hash", "2026-01-01T00:00:00Z")
    dates = pd.date_range("2024-01-02", periods=2)
    target = pd.Series([80.0, 88.0], index=dates)
    local = {
        name: observations(dates, values, name, "RUB/CNY", payload, "market_close")
        for name, values in {"cny_tom_vwap": [10, 11], "cny_tom_close": [11, 12]}.items()
    }
    panel = align_inputs(local, ecb_usd_cny(payload), target)

    assert panel.ecb_usd_cny.tolist() == [7.0, 8.0]
    assert panel.cross_vwap.tolist() == [70.0, 88.0]
    assert panel.cross_close.tolist() == [77.0, 96.0]
    assert panel.basis_factor.tolist() == pytest.approx([80 / 70, 1])
