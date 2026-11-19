"""Панель наблюдений по датам периода с доступностью на момент прогноза."""

import numpy as np
import pandas as pd

from usd_rub.freshness import ensure_allowed


class VintagePanel:
    def __init__(self, frames, specs, start, end, mode="strict", fill_limit=15):
        if mode not in {"strict", "assumed"} or fill_limit < 1:
            raise ValueError("Некорректный режим доступности панели")
        self.index = pd.date_range(pd.Timestamp(start) - pd.Timedelta(days=1), end, freq="D")
        self.columns = list(specs)
        self.specs, self.mode, self.fill_limit = specs, mode, fill_limit
        self.values = np.full((len(self.index), len(specs)), np.nan)
        self.assumed_rows = {}
        events = []
        for column, (name, spec) in enumerate(specs.items()):
            ensure_allowed(name)
            if spec["transform"] not in {"diff", "logdiff"}:
                raise ValueError(f"Неизвестное преобразование {name}")
            frame = frames[name].copy()
            frame["date"] = pd.to_datetime(frame.date)
            frame["available_at"] = pd.to_datetime(frame.available_at, utc=True, format="mixed")
            frame["ingested_at"] = pd.to_datetime(frame.ingested_at, utc=True, format="mixed")
            assumed = frame.availability_basis.fillna("unknown").str.startswith("assumed")
            if mode == "strict":
                frame = frame.loc[~assumed].copy()
            unknown = frame.availability_basis.eq("unknown")
            count = 0
            if mode == "assumed" and "assumed_lag_days" in spec:
                lag = spec["assumed_lag_days"]
                if not isinstance(lag, int) or lag < 1:
                    raise ValueError(f"{name}: исследовательский лаг должен быть целым и >=1")
                dates = pd.DatetimeIndex(frame.loc[unknown, "date"]).tz_localize("Europe/Moscow")
                frame.loc[unknown, "available_at"] = (dates + pd.Timedelta(days=lag)).tz_convert(
                    "UTC"
                )
                count = int(unknown.sum())
            self.assumed_rows[name] = count
            frame = frame.drop_duplicates(["date", "available_at", "value"])
            positions = self.index.get_indexer(frame.date)
            for position, row in zip(positions, frame.itertuples(), strict=True):
                if position < 0:
                    continue
                if not np.isfinite(row.value) or pd.isna(row.available_at):
                    raise ValueError(f"{name}: некорректное наблюдение или доступность")
                if spec["transform"] == "logdiff" and row.value <= 0:
                    raise ValueError(f"{name}: логарифм неположительного значения")
                events.append(
                    (row.available_at.value, row.ingested_at.value, position, column, row.value)
                )
        self.events = sorted(events)
        self.pointer = 0
        self.cutoff = None

    def snapshot(self, cutoff):
        """Возврат к более раннему cutoff запрещён, чтобы не сохранить будущую ревизию."""
        cutoff = pd.Timestamp(cutoff)
        if cutoff.tz is None:
            raise ValueError("Для cutoff требуется часовой пояс")
        if self.cutoff is not None and cutoff < self.cutoff:
            raise ValueError("Моменты прогноза должны идти по возрастанию")
        self.cutoff = cutoff
        while self.pointer < len(self.events) and self.events[self.pointer][0] <= cutoff.value:
            _, _, position, column, value = self.events[self.pointer]
            self.values[position, column] = value
            self.pointer += 1
        last_date = cutoff.tz_convert("Europe/Moscow").tz_localize(None).normalize()
        stop = self.index.searchsorted(last_date, side="right")
        raw = pd.DataFrame(self.values[:stop], index=self.index[:stop], columns=self.columns)
        filled = raw.ffill(limit=self.fill_limit)
        for name, spec in self.specs.items():
            last = raw[name].last_valid_index()
            if last is not None:
                filled.loc[filled.index > last, name] = np.nan
            if spec["transform"] == "logdiff":
                filled[name] = np.log(filled[name])
        return filled.diff()
