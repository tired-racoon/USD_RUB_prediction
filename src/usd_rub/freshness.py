"""Допуск источников, возраст наблюдений и проверяемые пропуски."""

import pandas as pd

BLOCKED_SERIES = {
    "gdelt_": "GDELT исключён по требованию пользователя",
    "otc_": "Старый SOAP VKO заканчивается в 2024 году; нужен действующий источник",
    "liquidity_operations": "SOAP SaldoXML заканчивается в 2025 году",
    "cbr_fx_net_purchases": "Старая выгрузка неполна и имела неверную единицу измерения",
    "zenodo_": "Исследовательский набор не является проверенным действующим источником",
    "fnb_inflows_rub_bn": "Движение ФНБ не согласовано с изменением объёма фонда",
    "fnb_withdrawals_rub_bn": "Движение ФНБ не согласовано с изменением объёма фонда",
    "fnb_revaluation_rub_bn": "Движение ФНБ не согласовано с изменением объёма фонда",
    "fnb_unreconciled_change_rub_bn": "Необъяснённый остаток сверки ФНБ, не валютная операция",
}

MONTHLY_PREFIXES = (
    "budget_",
    "fnb_",
    "china_",
    "household_",
    "corporate_",
    "business_",
    "export_settlement_",
    "import_settlement_",
    "industrial_production_",
    "oil_gas_",
    "ndpi_",
    "ndd_",
)
MONTHLY_SERIES = {
    "cpi_us",
    "cpi_ru_mom_index",
    "m2",
    "m2x",
    "exports",
    "imports",
    "trade_balance",
    "gas_eu",
    "gold_monthly",
    "gpr_russia_monthly",
    "gpr_russia_ru_press_monthly",
    "lng_japan_monthly",
    "urals_official_monthly",
    "nonfinancial_net_fx_sales",
    "exporter_net_fx_sales",
    "minfin_fx_gold_operations_rub_bn",
}


def ensure_allowed(name):
    for prefix, reason in BLOCKED_SERIES.items():
        if name.startswith(prefix):
            raise ValueError(f"Ряд {name} исключён: {reason}")


def freshness(frame, as_of=None):
    """Оценка возраста после конца периода, а не после его первого дня."""
    reference = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="Europe/Moscow")
    reference = reference.tz_localize(None).normalize()
    dates = pd.DatetimeIndex(pd.to_datetime(frame["date"]).drop_duplicates()).sort_values()
    dates = dates[dates <= reference]
    if dates.empty:
        return {"status": "no_observations", "as_of": str(reference.date())}
    name = frame["series"].iloc[0]
    gap = dates.to_series().diff().dt.days.median() if len(dates) > 1 else float("nan")
    if name == "cbr_meeting":
        frequency, period_end, limit = "event", dates.max(), None
    elif name in {"m2", "m2x"}:
        frequency, period_end, limit = "monthly_stock", dates.max(), 60
    elif "quarterly" in name or pd.notna(gap) and 80 <= gap <= 100:
        frequency, period_end, limit = (
            "quarterly",
            dates.max().to_period("Q").end_time.normalize(),
            120,
        )
    elif (
        name in MONTHLY_SERIES
        or name.startswith(MONTHLY_PREFIXES)
        or pd.notna(gap)
        and 25 <= gap <= 35
    ):
        frequency, period_end, limit = (
            "monthly",
            dates.max().to_period("M").end_time.normalize(),
            60,
        )
    elif pd.notna(gap) and 5 <= gap <= 10:
        frequency, period_end, limit = "weekly", dates.max(), 21
    else:
        frequency, period_end, limit = "daily_or_sparse", dates.max(), 15
    age = max(0, (reference - period_end).days)
    return {
        "status": "review_schedule"
        if limit is None
        else "stale"
        if age > limit
        else "within_lag_budget",
        "as_of": str(reference.date()),
        "last": str(dates.max().date()),
        "period_end": str(period_end.date()),
        "age_days_after_period_end": age,
        "frequency": frequency,
        "max_age_days": limit,
        "warning": "Порог не доказывает наличие последнего опубликованного выпуска",
    }


def coverage_gaps(name, dates):
    """Пропуски, которые можно проверить без торгового календаря."""
    dates = pd.DatetimeIndex(pd.to_datetime(dates).drop_duplicates()).sort_values()
    if dates.empty:
        return {"missing_calendar_years": [], "missing_periods": [], "documented_absences": []}
    years = sorted(set(range(dates.min().year, dates.max().year + 1)) - set(dates.year))
    missing, documented = [], []
    if name.startswith("fns_") or name in {"ru_holiday_fns", "gpr_russia_ru_press_daily"}:
        missing = (
            pd.date_range(dates.min(), dates.max()).difference(dates).strftime("%Y-%m-%d").tolist()
        )
    elif (
        name in MONTHLY_SERIES
        or name.startswith(MONTHLY_PREFIXES)
        or len(dates) > 1
        and 25 <= dates.to_series().diff().dt.days.median() <= 35
    ):
        missing = (
            pd.period_range(dates.min(), dates.max(), freq="M")
            .difference(dates.to_period("M"))
            .astype(str)
            .tolist()
        )
        if name == "cpi_us" and "2025-10" in missing:
            missing.remove("2025-10")
            documented.append(
                {
                    "period": "2025-10",
                    "reason": "BLS не выпускало ИПЦ из-за остановки сбора",
                    "source_url": "https://www.bls.gov/cpi/additional-resources/2025-federal-government-shutdown-impact-cpi-faq.htm",
                }
            )
    return {
        "missing_calendar_years": years,
        "missing_periods": missing,
        "documented_absences": documented,
    }


def require_current(frame, year=None, as_of=None):
    stale = []
    for name, part in frame.groupby("series"):
        ensure_allowed(name)
        status = freshness(part, as_of if as_of is not None else f"{year}-12-31" if year else None)
        if status["status"] in {"stale", "no_observations"}:
            stale.append(f"{name}: {status.get('last', 'нет данных')}")
    if stale:
        raise ValueError("Превышен допустимый возраст данных: " + "; ".join(stale))
