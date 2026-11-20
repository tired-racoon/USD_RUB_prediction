"""Команды загрузки данных и запуска эталонных моделей."""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from usd_rub.backtest import daily_target, regime_scores, score, walk_forward
from usd_rub.catalog import FRED_SERIES, SOURCES, fetch
from usd_rub.features import build_features
from usd_rub.freshness import ensure_allowed, freshness, require_current
from usd_rub.meeting_features import CALENDAR_PATH
from usd_rub.models import MODEL_FACTORIES
from usd_rub.sources.cbr import validate_range
from usd_rub.sources.events import extract_document, parse_events_csv
from usd_rub.sources.markets import fetch_alfred, fetch_moex
from usd_rub.sources.tables import (
    discover_table_links,
    load_payload,
    parse_long_table,
    parse_month_matrix,
    parse_period_rows,
    parse_row_timeseries,
)
from usd_rub.storage import Client, coverage, input_fingerprints, read_series, save_series, utc_now


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def fetch_command(args):
    validate_range(args.start, args.end)
    names = (
        args.sources.split(",")
        if args.sources
        else [
            name
            for name, source in SOURCES.items()
            if source.enabled
            and (
                source.profile == args.profile
                or (args.profile == "extended" and source.profile == "core")
            )
        ]
    )
    unknown = set(names) - set(SOURCES)
    if unknown:
        raise ValueError(f"Неизвестные источники: {sorted(unknown)}")
    report = {"started_at": utc_now(), "start": args.start, "end": args.end, "sources": {}}

    def download(name):
        with Client(
            args.data_dir, refresh=args.refresh, browser_fallback=args.browser_fallback
        ) as client:
            return fetch(SOURCES[name], client, args.start, args.end)

    failed = False
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download, name): name for name in names}
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                if result.empty:
                    raise ValueError("Нет числовых наблюдений в запрошенном интервале")
                excluded_series = {}
                for series, part in result.groupby("series"):
                    try:
                        ensure_allowed(series)
                        require_current(part, as_of=args.end)
                    except ValueError as exc:
                        excluded_series[series] = {
                            "reason": str(exc),
                            "last": str(part.date.max().date()),
                        }
                result = result.loc[~result["series"].isin(excluded_series)]
                if result.empty:
                    raise ValueError(f"Все ряды исключены: {excluded_series}")
                save_series(result, args.data_dir)
                series_reports = {
                    s: coverage(part, args.start, args.end) for s, part in result.groupby("series")
                }
                complete = all(
                    r["history_at_least_10y"]
                    and r["covers_requested_bounds"]
                    and r["continuous_checked_history"]
                    for r in series_reports.values()
                )
                status = "ok" if complete else "partial_history"
                if args.require_10y and not complete:
                    failed = True
                report["sources"][name] = {
                    "status": status,
                    "series": series_reports,
                    "limitation": SOURCES[name].limitation,
                    "excluded_series": excluded_series,
                }
                print(f"{name}: {status}, {len(result)} наблюдений", flush=True)
            except Exception as exc:
                failed = True
                report["sources"][name] = {
                    "status": "error",
                    "error": str(exc),
                    "limitation": SOURCES[name].limitation,
                }
                print(f"{name}: ошибка: {exc}", file=sys.stderr, flush=True)
    report["finished_at"] = utc_now()
    report_path = Path(args.data_dir) / "reports" / "latest_fetch.json"
    write_json(report_path, report)
    stamp = datetime.now(ZoneInfo("Europe/Moscow")).strftime("%Y%m%d_%H%M%S")
    write_json(report_path.with_name(f"fetch_{stamp}.json"), report)
    print(f"Отчёт: {report_path}")
    return int(failed)


def models_command(args):
    frame = read_series(args.data_dir, args.target)
    require_current(frame, as_of=args.end or frame.date.max())
    y = daily_target(frame, args.start, args.end)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    predictions = walk_forward(
        y,
        args.models.split(",") if args.models else None,
        args.min_train,
        args.window,
        args.horizon,
        args.step,
        args.alpha,
        args.test_start,
    )
    metrics = score(predictions, args.step)
    predictions.to_csv(output / "predictions.csv", index=False)
    metrics.to_csv(output / "metrics.csv", index=False)
    regime_scores(predictions, args.step).to_csv(output / "regime_metrics.csv", index=False)
    write_json(
        output / "run.json",
        {
            "created_at": utc_now(),
            "parameters": vars(args),
            "n_target": len(y),
            "target_first": str(y.index.min()),
            "target_last": str(y.index.max()),
            "target_raw_sha256": sorted(frame["raw_sha256"].unique()),
            "origin_time": "00:00 Europe/Moscow on origin date",
            "horizon_unit": "calendar days",
            "inputs": input_fingerprints(
                [
                    Path(args.data_dir) / "normalized" / f"{args.target}.csv",
                    Path(__file__).parent / "models.py",
                    Path(__file__).parent / "backtest.py",
                ]
            ),
        },
    )
    print(
        metrics.loc[
            metrics["horizon"].isin([1, args.horizon]),
            [
                "model",
                "horizon",
                "n",
                "mae",
                "rmse",
                "mape_pct",
                "smape_pct",
                "rmspe_pct",
                "rmse_skill_vs_rw",
            ],
        ].to_string(index=False)
    )
    print(f"Результаты: {output}")
    return 0


def forecast_command(args):
    frame = read_series(args.data_dir, args.target)
    require_current(frame, as_of=args.end)
    y = daily_target(frame, args.start, args.end)
    history = y.iloc[-args.window :] if args.window else y
    rows = []
    names = args.models.split(",") if args.models else list(MODEL_FACTORIES)
    for name in names:
        prediction = MODEL_FACTORIES[name]().fit(history).predict(args.horizon, args.alpha)
        for h in range(args.horizon):
            rows.append(
                {
                    "model": name,
                    "origin": y.index[-1],
                    "target_date": y.index[-1] + pd.Timedelta(days=h + 1),
                    "horizon": h + 1,
                    "prediction": prediction.mean[h],
                    "lower": prediction.lower[h],
                    "upper": prediction.upper[h],
                }
            )
    frame = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    write_json(
        Path(args.output).with_suffix(".json"),
        {
            "created_at": utc_now(),
            "target": args.target,
            "origin": str(y.index[-1].date()),
            "horizon_unit": "calendar days",
            "inputs": input_fingerprints(
                [
                    Path(args.data_dir) / "normalized" / f"{args.target}.csv",
                    Path(__file__).parent / "models.py",
                ]
            ),
            "parameters": vars(args),
        },
    )
    print(frame.to_string(index=False))
    return 0


def import_command(args):
    spec = json.loads(Path(args.spec).read_text())
    with Client(args.data_dir, refresh=args.refresh) as client:
        payload = load_payload(client, args.input)
    parser_name = spec.pop("parser")
    transform = spec.pop("transform", None)
    if parser_name == "long":
        result = parse_long_table(payload, spec)
    elif parser_name == "month_matrix":
        result = parse_month_matrix(payload, **spec)
    elif parser_name == "row_timeseries":
        result = parse_row_timeseries(payload, **spec)
    elif parser_name == "period_rows":
        result = parse_period_rows(payload, **spec)
    elif parser_name == "events":
        events = parse_events_csv(payload)
        folder = Path(args.data_dir) / "events"
        folder.mkdir(parents=True, exist_ok=True)
        events.to_csv(folder / f"{payload.sha256}.csv", index=False)
        print(f"Импортировано событий: {len(events)}")
        return 0
    else:
        raise ValueError(f"Неизвестный формат: {parser_name}")
    if transform == "ytd_to_monthly":
        from usd_rub.sources.minfin import ytd_to_monthly

        result = ytd_to_monthly(result)
    elif transform is not None:
        raise ValueError(f"Неизвестное преобразование: {transform}")
    save_series(result, args.data_dir)
    print(f"Импортировано наблюдений: {len(result)}, ряды: {', '.join(result['series'].unique())}")
    return 0


def features_command(args):
    names = args.series.split(",")
    frames = {name: read_series(args.data_dir, name) for name in names}
    for frame in frames.values():
        require_current(frame, as_of=args.end)
    lags = json.loads(Path(args.assumed_lags).read_text()) if args.assumed_lags else None
    event_files = sorted((Path(args.data_dir) / "events").glob("*.csv"))
    event_files.extend(sorted((Path(args.data_dir) / "calendars").glob("*_events.csv")))
    events = (
        pd.concat([pd.read_csv(path) for path in event_files], ignore_index=True)
        if event_files
        else None
    )
    if events is not None:
        events = events.drop_duplicates()
        if events["event_id"].duplicated().any():
            comparable = events.drop(columns=["ingested_at"], errors="ignore").drop_duplicates()
            if comparable.event_id.duplicated().any():
                raise ValueError("Конфликтующие event_id в импортированных событиях")
            events = events.drop_duplicates("event_id")
    dependencies = [
        Path(args.data_dir) / "normalized" / f"{name}.csv" for name in names
    ] + event_files
    dependencies.extend(
        Path(__file__).parent / name
        for name in ["features.py", "freshness.py", "cli.py", "storage.py", "meeting_features.py"]
    )
    dependencies.append(CALENDAR_PATH)
    if args.assumed_lags:
        dependencies.append(Path(args.assumed_lags))
    fingerprints = input_fingerprints(dependencies)
    frame = build_features(frames, args.start, args.end, args.cutoff, lags, events)
    if fingerprints != input_fingerprints(dependencies):
        raise ValueError("Входные файлы изменились во время сборки признаков")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output)
    write_json(
        Path(args.output).with_suffix(".json"),
        {
            "mode": "assumed_lags_latest_vintage" if lags else "strict_availability",
            "assumed_lags_days": lags,
            "cutoff_moscow": args.cutoff,
            "missing_share": frame.isna().mean().to_dict(),
            "warning": "Лаг не устраняет ревизии текущего винтажа" if lags else None,
            "created_at": utc_now(),
            "series": names,
            "events_files": [str(p) for p in event_files],
            "cbr_meeting_calendar": str(CALENDAR_PATH),
            "inputs": fingerprints,
            "freshness": {name: freshness(part, args.end) for name, part in frames.items()},
            "all_null_columns": frame.columns[frame.isna().all()].tolist(),
        },
    )
    print(f"Датасет: {args.output}, {len(frame)} строк")
    return 0


def main(argv=None):
    today = datetime.now(ZoneInfo("Europe/Moscow")).date().isoformat()
    parser = argparse.ArgumentParser(prog="usd-rub")
    parser.add_argument("--data-dir", default="data")
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("sources", help="Реестр источников")
    listing.add_argument("--json", action="store_true")
    dl = sub.add_parser("fetch", help="Загрузить исторические данные")
    dl.add_argument("--sources")
    dl.add_argument("--profile", choices=["core", "extended", "research"], default="core")
    dl.add_argument("--start", default="2015-01-01")
    dl.add_argument("--end", default=today)
    dl.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    dl.add_argument("--refresh", action="store_true")
    dl.add_argument("--require-10y", action="store_true")
    dl.add_argument("--browser-fallback", action="store_true")
    factor = sub.add_parser("factor-backtest", help="Сравнить DFM Giannone с бейзлайнами")
    factor.add_argument("--config", default="config/giannone_daily.json")
    factor.add_argument("--target", default="usd_rub")
    factor.add_argument("--start", default="2015-01-01")
    factor.add_argument("--end", default=today)
    factor.add_argument("--mode", choices=["strict", "assumed", "both"], default="both")
    factor.add_argument("--min-train", type=int, default=730)
    factor.add_argument("--window", type=int, default=730)
    factor.add_argument("--test-start", default="2017-01-01")
    factor.add_argument("--horizon", type=int, choices=range(1, 8), default=7)
    factor.add_argument("--step", type=int, default=1)
    factor.add_argument("--alpha", type=float, default=0.05)
    factor.add_argument("--output", default="data/reports/giannone_20261118")
    ensemble = sub.add_parser("forecast-ensemble", help="Основной прогноз USD/RUB на 1–7 дней")
    ensemble.add_argument("--dataset", type=Path, help="Снимок признаков для исторического запуска")
    ensemble.add_argument("--config", type=Path, default=Path("config/ml_holdout.json"))
    ensemble.add_argument("--calendar", type=Path, default=Path("config/cbr_fixing_calendar.json"))
    ensemble.add_argument("--checkpoint", type=Path, help="Каталог локального чекпойнта t0-gamma")
    ensemble.add_argument("--end", help="Дата прогноза в формате ГГГГ-ММ-ДД")
    ensemble.add_argument("--train-before", help="Граница обучающих целей для повтора holdout")
    ensemble.add_argument("--output", type=Path, default=Path("data/reports/forecast_ensemble.csv"))
    for command in ("backtest", "forecast"):
        model = sub.add_parser(command)
        model.add_argument("--target", default="usd_rub")
        model.add_argument("--start")
        model.add_argument("--end")
        model.add_argument("--models")
        model.add_argument("--window", type=int)
        model.add_argument("--horizon", type=int, choices=range(1, 8), default=7)
        model.add_argument("--alpha", type=float, default=0.05)
        model.add_argument(
            "--output",
            default="data/reports/baselines"
            if command == "backtest"
            else "data/reports/forecast.csv",
        )
        if command == "backtest":
            model.add_argument("--min-train", type=int, default=365)
            model.add_argument("--test-start")
            model.add_argument("--step", type=int, default=1)
    imp = sub.add_parser("import-table")
    imp.add_argument("--input", required=True)
    imp.add_argument("--spec", required=True)
    imp.add_argument("--refresh", action="store_true")
    feature = sub.add_parser("features")
    feature.add_argument(
        "--series", default="usd_rub,cny_rub,cny_tom_close,si_front_settle,imoex,vix,key_rate"
    )
    feature.add_argument("--start", default="2015-01-01")
    feature.add_argument("--end", default=today)
    feature.add_argument("--cutoff", default="15:00")
    feature.add_argument("--assumed-lags")
    feature.add_argument("--output", default="data/features/daily.csv")
    discover = sub.add_parser("discover-tables")
    discover.add_argument("--url", required=True)
    discover.add_argument("--pattern", required=True)
    document = sub.add_parser("extract-document")
    document.add_argument("--input", required=True)
    document.add_argument("--output", required=True)
    alfred = sub.add_parser("alfred")
    alfred.add_argument("--series", choices=list(FRED_SERIES), required=True)
    alfred.add_argument("--vintage", required=True)
    alfred.add_argument("--start", default="2015-01-01")
    alfred.add_argument("--end", default=today)
    iss = sub.add_parser("moex")
    iss.add_argument("--spec", required=True)
    iss.add_argument("--start", default="2015-01-01")
    iss.add_argument("--end", default=today)
    gdelt = sub.add_parser("gdelt")
    gdelt.add_argument("--start", required=True)
    gdelt.add_argument("--end", required=True)
    ofac = sub.add_parser("ofac")
    ofac.add_argument("--start", required=True)
    ofac.add_argument("--end", required=True)
    ofac.add_argument("--documents", action="store_true")
    ofac.add_argument("--output", default="data/documents/ofac.json")
    sanctions = sub.add_parser("sanctions")
    sanctions.add_argument("--sources", default="ofac_sdn,ofac_non_sdn,uk,eu,bis")
    sanctions.add_argument("--refresh", action="store_true")
    releases = sub.add_parser("cbr-releases")
    releases.add_argument("--start", default="2015-01-01")
    releases.add_argument("--end", default=today)
    audit = sub.add_parser("audit")
    audit.add_argument("--as-of", default=today)
    news = sub.add_parser("news-fetch", help="Собрать архив российских новостных заголовков")
    news.add_argument("--start", default="2016-10-01")
    news.add_argument("--end", default=today)
    news.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    news.add_argument("--output")
    news = sub.add_parser("news-features", help="Рассчитать дневные эмбеддинги и тональность")
    news.add_argument("--input")
    news.add_argument("--output")
    news.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    news.add_argument("--batch-size", type=int, default=128)
    news.add_argument("--smartlab-parquet", action="append", default=[])
    news.add_argument("--rbc-ids", help="Диапазон постов канала РБК START:STOP")
    repair = sub.add_parser("repair-data")
    repair.add_argument("--as-of", default=today)
    args = parser.parse_args(argv)
    try:
        if args.command == "sources":
            if args.json:
                print(
                    json.dumps([asdict(s) for s in SOURCES.values()], ensure_ascii=False, indent=2)
                )
            else:
                for source in SOURCES.values():
                    status = " [отключён]" if not source.enabled else ""
                    print(f"{source.name:28s} {source.profile:9s} {source.description}{status}")
            return 0
        if args.command == "fetch":
            return fetch_command(args)
        if args.command == "backtest":
            return models_command(args)
        if args.command == "factor-backtest":
            from usd_rub.factor_backtest import run_comparison

            return run_comparison(args)
        if args.command == "forecast-ensemble":
            from usd_rub.ensemble_forecast import run_forecast
            from usd_rub.sources.news_archive import write_csv

            forecast, metadata = run_forecast(args, today)
            write_csv(forecast, args.output)
            write_json(args.output.with_suffix(".json"), {"created_at": utc_now(), **metadata})
            print(f"{args.output}: прогноз на 1–7 дней от {metadata['origin']}")
            return 0
        if args.command == "forecast":
            return forecast_command(args)
        if args.command == "import-table":
            return import_command(args)
        if args.command == "features":
            return features_command(args)
        if args.command == "news-fetch":
            from usd_rub.sources.news_archive import fetch_news

            return fetch_news(args)
        if args.command == "news-features":
            from usd_rub.news_features import build_news

            return build_news(args)
        if args.command == "audit":
            from usd_rub.audit import audit_data

            report = audit_data(args.data_dir, as_of=args.as_of)
            print(json.dumps(report["counts"], ensure_ascii=False, indent=2))
            return 0
        if args.command == "repair-data":
            from usd_rub.repairs import repair_saved_data

            report = repair_saved_data(args.data_dir, args.as_of)
            print(
                json.dumps(
                    {
                        key: value
                        for key, value in report.items()
                        if key not in {"inputs_after_repair", "changes"}
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        with Client(args.data_dir, refresh=getattr(args, "refresh", False)) as client:
            if args.command == "discover-tables":
                print(
                    json.dumps(
                        discover_table_links(client.get(args.url), args.pattern),
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            elif args.command == "extract-document":
                payload = load_payload(client, args.input)
                write_json(
                    args.output,
                    {
                        "source_url": payload.url,
                        "raw_sha256": payload.sha256,
                        "ingested_at": payload.ingested_at,
                        "pages": extract_document(payload),
                    },
                )
            elif args.command == "alfred":
                import os

                if not os.environ.get("FRED_API_KEY"):
                    raise ValueError("Нужен бесплатный FRED_API_KEY; инструкция: docs/access.md")
                series_id, unit, _ = FRED_SERIES[args.series]
                result = fetch_alfred(
                    client, series_id, args.series, unit, args.start, args.end, args.vintage
                )
                save_series(result, args.data_dir)
                print(f"ALFRED {args.series}: {len(result)} наблюдений")
            elif args.command == "moex":
                spec = json.loads(Path(args.spec).read_text())
                result = fetch_moex(client, spec, args.start, args.end)
                if result.empty:
                    raise ValueError("Нет доступных числовых данных")
                save_series(result, args.data_dir)
            elif args.command == "gdelt":
                raise ValueError("GDELT исключён по требованию пользователя; загрузка отключена")
            elif args.command == "ofac":
                from usd_rub.sources.ofac import fetch_actions

                rows = fetch_actions(client, args.start, args.end, args.documents)
                write_json(
                    args.output,
                    {
                        "start": args.start,
                        "end": args.end,
                        "ingested_at": utc_now(),
                        "actions": rows,
                    },
                )
                print(f"OFAC: {len(rows)} документов, {args.output}")
                return int(any(row.get("document_error") for row in rows))
            elif args.command == "sanctions":
                from usd_rub.sources.sanctions import URLS, fetch_snapshots, fetch_uk_changes

                names = args.sources.split(",")
                if set(names) - set(URLS):
                    raise ValueError("Допустимые списки: " + ",".join(URLS))
                report = fetch_snapshots(client, names)
                print(json.dumps(report, ensure_ascii=False, indent=2))
                if "uk" in names:
                    try:
                        result = fetch_uk_changes(client)
                        print(
                            f"UK: {len(result['changes'])} записей официального журнала изменений"
                        )
                    except (ValueError, RuntimeError, OSError) as exc:
                        print(f"UK journal: {exc}", file=sys.stderr)
                return int(any(row["status"] == "error" for row in report.values()))
            elif args.command == "cbr-releases":
                from usd_rub.sources.reports import fetch_cbr_decisions

                records = fetch_cbr_decisions(client, args.start, args.end)
                print(
                    f"ЦБ: документов {sum(item['status'] == 'downloaded' for item in records)}, ошибок {sum(item['status'] == 'error' for item in records)}"
                )
                return int(not records or any(item["status"] == "error" for item in records))
        return 0
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
