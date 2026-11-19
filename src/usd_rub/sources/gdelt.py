"""Дневные архивы событий GDELT 1.0 с российскими участниками."""

from io import BytesIO
from zipfile import ZipFile

import pandas as pd

from usd_rub.storage import observations


def parse_gdelt(payload, archive_date):
    count, tone_sum, tone_count, goldstein_sum, goldstein_count = 0, 0.0, 0, 0.0, 0
    with ZipFile(BytesIO(payload.content)) as archive:
        files = [item for item in archive.infolist() if not item.is_dir()]
        if len(files) != 1 or files[0].file_size > 1_000_000_000:
            raise ValueError("GDELT: неожиданный состав дневного ZIP")
        with archive.open(files[0]) as data:
            for chunk in pd.read_csv(data, sep="\t", header=None, dtype=str, chunksize=100_000):
                if chunk.shape[1] != 58:
                    raise ValueError(f"GDELT 1.0: ожидалось 58 столбцов, получено {chunk.shape[1]}")
                russian = chunk.loc[
                    (chunk[7] == "RUS") | (chunk[17] == "RUS") | (chunk[51] == "RS")
                ]
                count += len(russian)
                tone = pd.to_numeric(russian[34], errors="raise")
                goldstein = pd.to_numeric(russian[30], errors="raise")
                tone_sum += tone.sum()
                tone_count += tone.notna().sum()
                goldstein_sum += goldstein.sum()
                goldstein_count += goldstein.notna().sum()
    values = [
        ("gdelt_ru_event_count", count, "events"),
        ("gdelt_ru_avg_tone", tone_sum / tone_count if tone_count else None, "score"),
        (
            "gdelt_ru_goldstein",
            goldstein_sum / goldstein_count if goldstein_count else None,
            "score",
        ),
    ]
    return pd.concat(
        [
            observations(
                [archive_date], [value], name, unit, payload, "conservative_daily_batch", lag_days=2
            )
            for name, value, unit in values
        ],
        ignore_index=True,
    )


def fetch_gdelt_day(client, archive_date):
    raise ValueError("GDELT исключён по требованию пользователя; загрузка отключена")
