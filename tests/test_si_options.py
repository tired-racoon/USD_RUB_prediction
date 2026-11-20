"""Даты опционов и интерполяция волатильности по сроку."""

from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from usd_rub.sources.si_options import constant_maturity, contracts


def test_si_expiry_comes_from_the_full_contract_code():
    content = BytesIO()
    with ZipFile(content, "w") as archive:
        archive.writestr(
            "options_coefficients_archive.csv",
            "optionscoefficients\n\n"
            "tradedate;isin;option_type;expiration_date;delta;volat;openposition\n"
            "01.06.2015;Si-6.15M150615PA 50000;P;15.06.2005;-0,30;20,00;20\n"
            "01.06.2015;Si-6.15M010615CA 50000;C;;0,30;20,00;20\n"
            "01.06.2015;Eu-6.15M150615CA 50000;C;;0,30;20,00;20\n",
        )
    frame = contracts(content.getvalue())
    assert frame.days.tolist() == [14]
    assert frame.call_delta.iloc[0] == pytest.approx(0.70)


def test_constant_maturity_uses_total_variance_and_no_extrapolation():
    expirations = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02"] * 3),
            "days": [20, 40, 45],
            "oi": [100, 100, 100],
            "atm": [20.0, 30.0, 31.0],
            "call25": [24.0, 34.0, np.nan],
            "put25": [20.0, 30.0, 40.0],
        }
    )
    result = constant_maturity(expirations).iloc[0]
    atm = np.sqrt((20 * 20**2 + 40 * 30**2) / 60)
    call = np.sqrt((20 * 24**2 + 40 * 34**2) / 60)
    assert result.si_iv_30d == pytest.approx(atm)
    assert result.si_rr25_30d == pytest.approx(call - atm)
    assert np.isnan(result.si_iv_7d)
