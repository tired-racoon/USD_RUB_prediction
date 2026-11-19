"""Индексы геополитического риска Caldara–Iacoviello."""

from io import BytesIO

import pandas as pd

from usd_rub.storage import observations


def parse_gpr(payload, daily=False):
    table = pd.read_excel(BytesIO(payload.content), engine="xlrd")
    columns = (
        {"GPRD": "gpr_daily", "GPRD_ACT": "gpr_acts_daily", "GPRD_THREAT": "gpr_threats_daily"}
        if daily
        else {
            "GPR": "gpr_monthly",
            "GPRT": "gpr_threats_monthly",
            "GPRA": "gpr_acts_monthly",
            "GPRC_RUS": "gpr_russia_monthly",
        }
    )
    date_column = "date" if daily else "month"
    if date_column not in table or not set(columns).issubset(table):
        raise ValueError("GPR: изменился формат таблицы")
    return pd.concat(
        [
            observations(table[date_column], table[column], name, "index points", payload)
            for column, name in columns.items()
        ],
        ignore_index=True,
    )


def fetch_gpr(client, start, end, daily=False):
    filename = "data_gpr_daily_recent.xls" if daily else "data_gpr_export.xls"
    return parse_gpr(client.get("https://www.matteoiacoviello.com/gpr_files/" + filename), daily)
