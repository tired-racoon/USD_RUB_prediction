"""Проверка качества, покрытия, доступности и допуска сохранённых рядов."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from usd_rub.freshness import coverage_gaps, ensure_allowed, freshness
from usd_rub.storage import read_series, utc_now


def audit_data(root="data", year=None, as_of=None):
    root = Path(root)
    reference = pd.Timestamp(
        as_of or (f"{year}-12-31" if year else pd.Timestamp.now(tz="Europe/Moscow").date())
    )
    rows = []
    for path in sorted((root / "normalized").glob("*.csv")):
        frame = read_series(root, path.stem)
        first, last = frame.date.min(), frame.date.max()
        gaps = coverage_gaps(path.stem, frame.date)
        current = freshness(frame, reference)
        reason = ""
        duplicate = int(frame.duplicated(["date", "vintage"]).sum())
        invalid = int((~np.isfinite(frame.value)).sum())
        missing_raw = sorted(
            d for d in frame.raw_sha256.unique() if not (root / "raw" / "objects" / d).exists()
        )
        unknown = int(frame.availability_basis.eq("unknown").sum())
        assumed = int(frame.availability_basis.str.startswith("assumed").sum())
        try:
            ensure_allowed(path.stem)
            if duplicate or invalid or missing_raw:
                status, reason = (
                    "invalid",
                    "Повторы, неконечные значения или отсутствующие исходники",
                )
            elif current["status"] in {"stale", "no_observations"}:
                status, reason = "stale", "Превышен допустимый возраст наблюдения"
            else:
                long = first <= last - pd.DateOffset(years=10)
                status = "active_long_history" if long else "active_short_history"
                if long and (gaps["missing_calendar_years"] or gaps["missing_periods"]):
                    status = "active_long_history_with_gaps"
        except ValueError as exc:
            status, reason = "excluded", str(exc)
        rows.append(
            {
                "series": path.stem,
                "status": status,
                "reason": reason,
                "first": str(first.date()),
                "last": str(last.date()),
                "dates": frame.date.nunique(),
                "rows_with_vintages": len(frame),
                "unit": frame.unit.iloc[0],
                **gaps,
                "freshness": current,
                "unknown_availability_rows": unknown,
                "assumed_availability_rows": assumed,
                "availability_basis": sorted(frame.availability_basis.unique().tolist()),
                "point_in_time_limitation": "Без исторической публикации: только после загрузки"
                if unknown
                else "Требуются допущения"
                if assumed
                else "См. основание доступности и версии источника",
                "duplicate_date_vintage": duplicate,
                "invalid_values": invalid,
                "missing_raw_sha256": missing_raw,
                "source_urls": sorted(frame.source_url.dropna().unique().tolist()),
                "max_gap_days": int(frame.date.drop_duplicates().sort_values().diff().dt.days.max())
                if frame.date.nunique() > 1
                else 0,
            }
        )
    statuses = sorted({row["status"] for row in rows})
    report = {
        "checked_at": utc_now(),
        "reference_date": str(reference.date()),
        "reference_year": reference.year,
        "series": rows,
        "counts": {status: sum(row["status"] == status for row in rows) for status in statuses},
        "warning": "Допуск источника не означает доказанную прогнозную пользу или полную историческую доступность",
    }
    folder = root / "reports"
    folder.mkdir(parents=True, exist_ok=True)
    content = json.dumps(report, ensure_ascii=False, indent=2)
    (folder / f"active_sources_{reference.year}.json").write_text(content)
    (folder / f"source_audit_{reference.date()}.json").write_text(content)
    return report
