"""Аудит сохранённого снимка без изменения исходных данных и сетевых запросов."""

import argparse
import hashlib
import json
from collections import Counter
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

from usd_rub.backtest import daily_target, dm_test
from usd_rub.features import asof_series, build_features
from usd_rub.freshness import ensure_allowed
from usd_rub.sources.cbr import SOAP_TABLES
from usd_rub.storage import read_series


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, output = Path(args.data_dir).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    reference = pd.Timestamp(args.as_of)
    files = [p for p in root.rglob("*") if p.is_file() and output not in p.parents]
    manifest = [
        {
            "file": str(p.relative_to(root)),
            "size": p.stat().st_size,
            "mtime_ns": p.stat().st_mtime_ns,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        for p in sorted(files)
    ]
    pd.DataFrame(manifest).to_csv(output / "input_manifest.csv", index=False)
    inventory = Counter(str(p.relative_to(root).parts[0]) for p in files)
    raw = root / "raw" / "objects"
    hashes = {p.name for p in raw.iterdir() if p.is_file()}
    bad_hashes = [
        p.name
        for p in raw.iterdir()
        if p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() != p.name
    ]
    json_errors, missing_links = [], []
    checked_links = 0

    def links(value, path):
        nonlocal checked_links
        if isinstance(value, dict):
            for key, item in value.items():
                raw_reference = (
                    key in {"raw_sha256", "document_raw_sha256", "contract_metadata_sha256"}
                    or key == "sha256"
                    and "url" in value
                )
                if raw_reference and isinstance(item, str) and len(item) == 64:
                    checked_links += 1
                    if item not in hashes:
                        missing_links.append({"file": str(path), "field": key, "hash": item})
                elif isinstance(item, (list, dict)):
                    links(item, path)
        elif isinstance(value, list):
            for item in value:
                links(item, path)

    for path in files:
        if path.suffix not in {".json", ".jsonl"}:
            continue
        try:
            if path.suffix == ".jsonl":
                for line in path.read_text().splitlines():
                    links(json.loads(line), path)
            else:
                links(json.loads(path.read_text()), path)
        except (ValueError, UnicodeError) as exc:
            json_errors.append({"file": str(path), "error": str(exc)})

    frames, series_rows, gaps = {}, [], []
    required = {
        "date",
        "value",
        "series",
        "unit",
        "source_url",
        "raw_sha256",
        "ingested_at",
        "vintage",
        "availability_basis",
        "available_at",
    }
    for path in sorted((root / "normalized").glob("*.csv")):
        frame = pd.read_csv(path, float_precision="round_trip")
        dates = pd.to_datetime(frame.date, errors="coerce")
        available = pd.to_datetime(frame.available_at, utc=True, format="mixed", errors="coerce")
        ingested = pd.to_datetime(frame.ingested_at, utc=True, format="mixed", errors="coerce")
        values = pd.to_numeric(frame.value, errors="coerce")
        unique = (
            frame.assign(date=dates)
            .sort_values(["date", "ingested_at"])
            .drop_duplicates("date", keep="last")
        )
        timeline = pd.DatetimeIndex(unique.date)
        frequency = float(timeline.to_series().diff().dt.days.median()) if len(timeline) > 1 else 0
        blocked = ""
        try:
            ensure_allowed(path.stem)
        except ValueError as exc:
            blocked = str(exc)
        tolerance = (
            10 if frequency <= 3 else 21 if frequency <= 10 else 75 if frequency <= 35 else 200
        )
        age = (reference - timeline.max()).days
        if 25 <= frequency <= 35:
            expected = pd.period_range(timeline.min(), timeline.max(), freq="M")
            missing_periods = expected.difference(timeline.to_period("M"))
            gaps.extend(
                {"series": path.stem, "kind": "missing_month", "date": str(d)}
                for d in missing_periods
            )
        if path.stem in {"fns_tax_payment_count", "fns_tax_filing_count", "ru_holiday_fns"}:
            missing_dates = pd.date_range(timeline.min(), timeline.max()).difference(timeline)
            gaps.extend(
                {"series": path.stem, "kind": "missing_calendar_day", "date": str(d.date())}
                for d in missing_dates
            )
        row = {
            "series": path.stem,
            "rows": len(frame),
            "dates": len(timeline),
            "first": str(timeline.min().date()),
            "last": str(timeline.max().date()),
            "age_days": age,
            "median_gap_days": frequency,
            "max_gap_days": int(timeline.to_series().diff().dt.days.max())
            if len(timeline) > 1
            else 0,
            "freshness_review_threshold_days": tolerance,
            "age_exceeds_threshold": age > tolerance,
            "blocked_reason": blocked,
            "unit": "|".join(frame.unit.unique()),
            "availability_basis": "|".join(sorted(frame.availability_basis.unique())),
            "unknown_availability_rows": int(frame.availability_basis.eq("unknown").sum()),
            "invalid_dates": int(dates.isna().sum()),
            "invalid_values": int((~np.isfinite(values)).sum()),
            "invalid_available_at": int(available.isna().sum()),
            "invalid_ingested_at": int(ingested.isna().sum()),
            "missing_columns": "|".join(sorted(required - set(frame.columns))),
            "wrong_series_rows": int(frame.series.ne(path.stem).sum()),
            "duplicate_date_vintage": int(frame.duplicated(["date", "vintage"]).sum()),
            "conflicting_date_vintage": int(
                frame.groupby(["date", "vintage"]).value.nunique().gt(1).sum()
            ),
            "revision_dates": int(frame.groupby("date").value.nunique().gt(1).sum()),
            "missing_raw_rows": int((~frame.raw_sha256.isin(hashes)).sum()),
            "future_observation_rows": int((dates > reference).sum()),
            "min": float(values.min()),
            "max": float(values.max()),
            "zero_rows": int(values.eq(0).sum()),
            "negative_rows": int(values.lt(0).sum()),
            "label": "|".join(frame.series_label.dropna().unique())
            if "series_label" in frame
            else "",
        }
        series_rows.append(row)
        frames[path.stem] = read_series(root, path.stem)
    pd.DataFrame(series_rows).to_csv(output / "series_inventory.csv", index=False)
    pd.DataFrame(gaps, columns=["series", "kind", "date"]).to_csv(output / "gaps.csv", index=False)

    def latest(name):
        return (
            frames[name]
            .sort_values(["date", "ingested_at"])
            .drop_duplicates("date", keep="last")
            .set_index("date")
            .value
        )

    consistency = []

    def identity(label, operands, residual, tolerance):
        table = pd.concat({name: latest(name) for name in operands}, axis=1).dropna()
        error = residual(table).abs()
        consistency.append(
            {
                "check": label,
                "dates": len(table),
                "tolerance": tolerance,
                "violations": int(error.gt(tolerance).sum()),
                "max_abs_error": float(error.max()),
            }
        )

    identity(
        "trade_balance=exports-imports",
        ["trade_balance", "exports", "imports"],
        lambda t: t.trade_balance - t.exports + t.imports,
        1.01,
    )
    if "budget_balance" in frames:
        for suffix in ["", "_ytd"]:
            names = ["budget_" + s + suffix for s in ["balance", "revenue", "expenditure"]]
            identity(
                "budget_balance" + suffix,
                names,
                lambda t: t.iloc[:, 0] - t.iloc[:, 1] + t.iloc[:, 2],
                0.021,
            )
        for name in [
            "budget_revenue",
            "budget_expenditure",
            "budget_balance",
            "budget_non_oil_revenue",
            "budget_domestic_vat",
            "budget_profit_tax",
        ]:
            part = latest(name)
            difference = (part.groupby(part.index.year).cumsum() - latest(name + "_ytd")).abs()
            consistency.append(
                {
                    "check": name + "_cumsum=ytd",
                    "dates": len(difference),
                    "violations": int(difference.gt(1e-6).sum()),
                    "max_abs_error": float(difference.max()),
                }
            )
    for direction in ["export", "import"]:
        names = [
            f"{direction}_settlement_share_{suffix}" for suffix in ["rub", "other", "listed_states"]
        ]
        identity(direction + "_currency_shares_sum=100", names, lambda t: t.sum(axis=1) - 100, 1e-6)
    for stem in ["goods", "services"]:
        names = [f"bop_{stem}_{suffix}_quarterly" for suffix in ["balance", "exports", "imports"]]
        identity(
            stem + "_quarterly_balance",
            names,
            lambda t: t.iloc[:, 0] - t.iloc[:, 1] + t.iloc[:, 2],
            0.021,
        )
    names = [
        f"fnb_{suffix}_rub_bn"
        for suffix in ["closing", "opening", "inflows", "withdrawals", "revaluation"]
    ]
    identity(
        "fnb_stock_flow",
        names,
        lambda t: t.iloc[:, 0] - t.iloc[:, 1] - t.iloc[:, 2] + t.iloc[:, 3] - t.iloc[:, 4],
        0.031,
    )
    identity("vix_cboe_vs_fred", ["vix", "vix_fred"], lambda t: t.vix - t.vix_fred, 0.011)
    identity(
        "ust_spread",
        ["ust_10y", "ust_2y", "ust_10y_2y"],
        lambda t: t.ust_10y - t.ust_2y - t.ust_10y_2y,
        0.011,
    )
    table = pd.concat(
        {s: latest("cny_tom_" + s) for s in ["open", "high", "low", "close", "vwap"]}, axis=1
    ).dropna()
    bad = (table.low > table[["open", "close", "vwap"]].min(axis=1) + 1e-6) | (
        table.high < table[["open", "close", "vwap"]].max(axis=1) - 1e-6
    )
    consistency.append(
        {
            "check": "cny_ohlc_vwap_bounds",
            "dates": len(table),
            "violations": int(bad.sum()),
            "examples": table.loc[bad].head(10).to_dict("index"),
        }
    )
    rev = latest("oil_gas_revenue")
    ytd = latest("oil_gas_revenue_ytd")
    error = (rev.groupby(rev.index.year).cumsum() - ytd).abs()
    consistency.append(
        {
            "check": "monthly_revenue_cumsum=ytd",
            "dates": len(error),
            "violations": int(error.gt(1e-6).sum()),
            "max_abs_error": float(error.max()),
        }
    )
    for name in ["usd_rub", "cny_rub", "eur_rub"]:
        frame = frames[name]
        errors = []
        checked = 0
        for digest, part in frame.groupby("raw_sha256"):
            records = ET.fromstring((raw / digest).read_bytes()).findall("Record")
            lookup = {
                pd.to_datetime(r.attrib["Date"], format="%d.%m.%Y"): float(
                    r.findtext("Value").replace(",", ".")
                )
                / float(r.findtext("Nominal"))
                for r in records
            }
            for r in part.itertuples():
                checked += 1
                if r.date not in lookup or not np.isclose(
                    r.value, lookup[r.date], rtol=0, atol=1e-10
                ):
                    errors.append(str(r.date))
        consistency.append(
            {
                "check": name + "_vs_raw_xml",
                "rows": checked,
                "violations": len(errors),
                "examples": errors[:10],
            }
        )
    original_checks = verify_original_values(frames, raw)
    pd.DataFrame(original_checks).to_csv(output / "original_value_checks.csv", index=False)

    feature_reports, feature_differences = [], []
    for path in sorted((root / "features").glob("*.csv")):
        meta = json.loads(path.with_suffix(".json").read_text())
        saved = pd.read_csv(path, index_col=0)
        index = pd.to_datetime(saved.index, utc=True).tz_convert("Europe/Moscow")
        saved.index = index
        names = [name for name in frames if name in saved.columns]
        event_paths = [
            root / ("calendars" if "calendars" in Path(p).parts else "events") / Path(p).name
            for p in meta.get("events_files", [])
        ]
        events = (
            pd.concat([pd.read_csv(p) for p in event_paths], ignore_index=True).drop_duplicates(
                "event_id"
            )
            if event_paths
            else None
        )
        fresh = build_features(
            {name: frames[name] for name in names},
            index.min().date(),
            index.max().date(),
            meta["cutoff_moscow"],
            meta.get("assumed_lags_days"),
            events,
        )
        differences = {}
        for col in saved:
            if col not in fresh:
                differences[col] = "missing_on_rebuild"
            elif pd.api.types.is_numeric_dtype(saved[col]):
                n = int(
                    (
                        ~np.isclose(saved[col], fresh[col], equal_nan=True, rtol=1e-10, atol=1e-10)
                    ).sum()
                )
                if n:
                    differences[col] = n
                    mask = ~np.isclose(
                        saved[col], fresh[col], equal_nan=True, rtol=1e-10, atol=1e-10
                    )
                    for cutoff in saved.index[mask]:
                        feature_differences.append(
                            {
                                "file": path.name,
                                "cutoff": str(cutoff),
                                "column": col,
                                "stored": saved.loc[cutoff, col],
                                "rebuilt": fresh.loc[cutoff, col],
                            }
                        )
            elif not saved[col].equals(fresh[col]):
                differences[col] = "different"
        feature_reports.append(
            {
                "file": str(path),
                "rows": len(saved),
                "columns": len(saved.columns),
                "first": str(index.min()),
                "last": str(index.max()),
                "all_null_columns": saved.columns[saved.isna().all()].tolist(),
                "rebuild_differences": differences,
                "dependency_mismatches": [
                    item["path"]
                    for item in meta.get("inputs", [])
                    if not Path(item["path"]).exists()
                    or hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() != item["sha256"]
                ],
            }
        )

    pd.DataFrame(
        feature_differences, columns=["file", "cutoff", "column", "stored", "rebuilt"]
    ).to_csv(output / "feature_differences.csv", index=False)
    household = frames["household_net_fx_purchases"]
    household[["date", "value", "evidence", "source_url", "document_page"]].to_csv(
        output / "household_fx_evidence.csv", index=False
    )
    events = pd.read_csv(root / "calendars" / "fns_events.csv")
    calendar_report = {
        "rows": len(events),
        "duplicate_event_ids": int(events.event_id.duplicated().sum()),
        "duplicate_logical_events": int(
            events.duplicated(["effective_time", "event_type", "tax_name", "evidence"]).sum()
        ),
        "events_by_year": events.effective_time.str[:4].value_counts().sort_index().to_dict(),
        "loaded_by_features_cli": True,
        "warning": "Архивные версии выбираются по времени доступности; допущения ФНС исключены из строгой сборки",
    }
    entity_reports = []
    for path in sorted((root / "entities").glob("*current.csv")):
        table = pd.read_csv(path)
        identifier = (
            "Unique ID" if "Unique ID" in table else "entity_id" if "entity_id" in table else "_id"
        )
        entity_reports.append(
            {
                "file": str(path),
                "rows": len(table),
                "identifier": identifier,
                "unique_ids": int(table[identifier].nunique()),
                "missing_id_rows": int(table[identifier].isna().sum()),
                "missing_raw_rows": int((~table.raw_sha256.isin(hashes)).sum()),
                "snapshot_dates": table.snapshot_date.unique().tolist(),
            }
        )
    document_reports = []
    for folder in sorted((root / "documents").iterdir()):
        if not folder.is_dir():
            continue
        documents = [
            json.loads(p.read_text()) for p in folder.glob("*.json") if p.stem != "manifest"
        ]
        periods = [d.get("period", d.get("date")) for d in documents]
        periods = [p for p in periods if p]
        document_reports.append(
            {
                "folder": str(folder),
                "documents": len(documents),
                "first_period": min(periods) if periods else None,
                "last_period": max(periods) if periods else None,
                "missing_publication_time": sum(not d.get("publication_time") for d in documents),
                "empty_texts": sum(
                    not any(p.get("text", "").strip() for p in d.get("pages", []))
                    for d in documents
                ),
            }
        )
    ofac = json.loads((root / "documents" / "ofac_2015_2026.json").read_text())["actions"]
    ofac_summary = {
        "actions": len(ofac),
        "unique_urls": len({d["source_url"] for d in ofac}),
        "with_text": sum(bool(d.get("pages")) for d in ofac),
        "errors": sum("document_error" in d for d in ofac),
        "title_russia_ukraine_matches": sum(
            any(word in d["title"].lower() for word in ["russia", "ukraine"]) for d in ofac
        ),
        "by_year": dict(sorted(Counter(d["date"][:4] for d in ofac).items())),
    }
    report = {
        "reference_date_moscow": args.as_of,
        "inventory_files": dict(inventory),
        "raw_objects": len(hashes),
        "bad_raw_hashes": bad_hashes,
        "json_errors": json_errors,
        "json_sha256_references_checked": checked_links,
        "missing_json_raw_references": missing_links,
        "normalized_series": len(series_rows),
        "normalized_rows": sum(r["rows"] for r in series_rows),
        "availability_by_series": dict(Counter(r["availability_basis"] for r in series_rows)),
        "unknown_availability_rows": sum(r["unknown_availability_rows"] for r in series_rows),
        "blocked_series": sum(bool(r["blocked_reason"]) for r in series_rows),
        "consistency": consistency,
        "original_value_checks": original_checks,
        "features": feature_reports,
        "documents": document_reports,
        "ofac": ofac_summary,
        "calendar": calendar_report,
        "entities": entity_reports,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    predictions = pd.read_csv(root / "reports" / "baselines" / "predictions.csv")
    target = daily_target(frames["usd_rub"])
    actual = target.reindex(pd.to_datetime(predictions.target_date)).to_numpy()
    last = target.reindex(pd.to_datetime(predictions.origin)).to_numpy()
    metrics = pd.read_csv(root / "reports" / "baselines" / "metrics.csv").set_index(
        ["model", "horizon"]
    )
    metric_errors = []
    for key, part in predictions.groupby(["model", "horizon"]):
        error = part.prediction - part.actual
        if not np.isclose(
            error.abs().mean(), metrics.loc[key, "mae"], atol=1e-10, rtol=1e-10
        ) or not np.isclose(
            np.sqrt(np.mean(error**2)), metrics.loc[key, "rmse"], atol=1e-10, rtol=1e-10
        ):
            metric_errors.append(str(key))
    report["saved_baselines"] = {
        "rows": len(predictions),
        "actual_mismatches": int((~np.isclose(actual, predictions.actual)).sum()),
        "origin_value_mismatches": int((~np.isclose(last, predictions["last"])).sum()),
        "duplicate_forecasts": int(predictions.duplicated(["model", "origin", "horizon"]).sum()),
        "wrong_horizons": int(
            (
                (
                    pd.to_datetime(predictions.target_date) - pd.to_datetime(predictions.origin)
                ).dt.days
                != predictions.horizon
            ).sum()
        ),
        "mae_rmse_mismatch_groups": metric_errors,
        "one_day_actual_unchanged_fraction": float(
            predictions.loc[(predictions.model == "rw") & (predictions.horizon == 1), "actual"]
            .eq(predictions.loc[(predictions.model == "rw") & (predictions.horizon == 1), "last"])
            .mean()
        ),
    }
    screen_features(frames, output)
    now_files = {p for p in root.rglob("*") if p.is_file() and output not in p.parents}
    changes = [
        {"file": row["file"], "reason": "modified_or_removed"}
        for row in manifest
        if not (root / row["file"]).exists()
        or (root / row["file"]).stat().st_size != row["size"]
        or (root / row["file"]).stat().st_mtime_ns != row["mtime_ns"]
    ]
    changes.extend(
        {"file": str(p.relative_to(root)), "reason": "added"} for p in now_files - set(files)
    )
    report["inputs_changed_during_run"] = changes
    dump(output / "audit.json", report)
    print(
        json.dumps(
            {
                "output": str(output),
                "series": len(series_rows),
                "raw_objects": len(hashes),
                "bad_hashes": len(bad_hashes),
                "json_errors": len(json_errors),
                "missing_json_raw_links": len(missing_links),
            },
            ensure_ascii=False,
        )
    )


def verify_original_values(frames, raw):
    """Сверка чисел FRED, MOEX и SOAP с полями сохранённых ответов."""
    soap_specs = {
        name: (row, dt, tag)
        for _, _, _, _, row, dt, columns in SOAP_TABLES.values()
        for tag, (name, _) in columns.items()
    }
    moex_columns = {
        "open": "OPEN",
        "high": "HIGH",
        "low": "LOW",
        "close": "CLOSE",
        "vwap": "WAPRICE",
        "trades": "NUMTRADES",
        "settle": "SETTLEPRICE",
        "oi": "OPENPOSITION",
        "volume": "VOLUME",
        "swaprate": "SWAPRATE",
    }
    checks = []
    cache = {}
    for name, frame in frames.items():
        url = frame.source_url.iloc[0]
        kind = (
            "fred"
            if "fred.stlouisfed.org/graph/" in url
            else "moex"
            if "iss.moex.com" in url
            else "soap"
            if name in soap_specs
            else None
        )
        if kind is None:
            continue
        failures, checked = [], 0
        for digest, part in frame.groupby("raw_sha256"):
            key = (digest, kind)
            if key not in cache:
                content = (raw / digest).read_bytes()
                if kind == "fred":
                    cache[key] = pd.read_csv(BytesIO(content))
                elif kind == "moex":
                    block = json.loads(content)["history"]
                    cache[key] = pd.DataFrame(block["data"], columns=block["columns"])
                else:
                    xml = ET.fromstring(content)
                    for node in xml.iter():
                        node.tag = node.tag.split("}")[-1]
                    cache[key] = xml
            original = cache[key]
            if kind == "fred":
                lookup = pd.Series(
                    pd.to_numeric(original.iloc[:, 1], errors="coerce").to_numpy(),
                    index=pd.to_datetime(original.iloc[:, 0]),
                ).to_dict()
            elif kind == "moex":
                column = (
                    "CLOSE"
                    if name in {"imoex", "moex_usd_fixing"}
                    else moex_columns[name.rsplit("_", 1)[1]]
                )
                scale = 1000 if name in {"si_front_close", "si_front_settle"} else 1
                lookup = pd.Series(
                    pd.to_numeric(original[column], errors="coerce").to_numpy() / scale,
                    index=pd.to_datetime(original.TRADEDATE),
                ).to_dict()
            else:
                row_tag, date_tag, value_tag = soap_specs[name]
                lookup = {
                    pd.Timestamp(row.findtext(date_tag)).tz_localize(None).normalize(): float(
                        row.findtext(value_tag).replace(",", ".")
                    )
                    for row in original.findall(".//" + row_tag)
                    if row.findtext(date_tag) and row.findtext(value_tag)
                }
            for row in part.itertuples():
                checked += 1
                if row.date not in lookup or not np.isclose(
                    row.value, lookup[row.date], rtol=1e-10, atol=1e-10
                ):
                    failures.append(str(row.date.date()))
        checks.append(
            {
                "series": name,
                "source_family": kind,
                "rows": checked,
                "mismatches": len(failures),
                "examples": "|".join(failures[:10]),
            }
        )
    return checks


def screen_features(frames, output):
    """Последовательная проверка шести признаков с известным консервативным лагом."""
    y = daily_target(frames["usd_rub"])
    cutoffs = y.index.tz_localize("Europe/Moscow") + pd.Timedelta(hours=15)
    names = ["cny_tom_close", "si_front_settle", "vix", "imoex"]
    aligned = pd.DataFrame(
        {
            name: asof_series(frames[name], cutoffs, max_age_days=10).value.to_numpy()
            for name in names
        },
        index=y.index,
    )
    x = pd.DataFrame(index=y.index)
    for name in ["cny_tom_close", "vix", "imoex"]:
        x[name + "_log_change_1d"] = np.log(aligned[name]).diff()
    x["si_spot_log_basis"] = np.log(aligned.si_front_settle / y)
    x["usd_rub_log_change_1d"] = np.log(y).diff()
    x["usd_rub_log_change_7d"] = np.log(y).diff(7)
    rows = []
    for horizon in [1, 7]:
        target = np.log(y.shift(-horizon) / y)
        for name in x:
            predictions = []
            values = x[name].to_numpy()
            returns = target.to_numpy()
            for i in range(730, len(y) - horizon):
                stop = i - horizon + 1
                first = max(0, stop - 730)
                a, b = values[first:stop], returns[first:stop]
                valid = np.isfinite(a) & np.isfinite(b)
                if valid.sum() < 365 or not np.isfinite(values[i]):
                    continue
                a, b = a[valid], b[valid]
                variance = np.dot(a - a.mean(), a - a.mean())
                slope = np.dot(a - a.mean(), b - b.mean()) / variance if variance > 1e-16 else 0
                prediction = float(y.iloc[i] * np.exp(b.mean() + slope * (values[i] - a.mean())))
                predictions.append((y.index[i], prediction, y.iloc[i + horizon], y.iloc[i]))
            table = pd.DataFrame(predictions, columns=["origin", "prediction", "actual", "rw"])
            for regime, mask in {
                "all": np.ones(len(table), dtype=bool),
                "otc_since_2024": table.origin.ge("2024-06-13"),
            }.items():
                part = table.loc[mask]
                errors, baseline = part.prediction - part.actual, part.rw - part.actual
                rmse, rw_rmse = np.sqrt(np.mean(errors**2)), np.sqrt(np.mean(baseline**2))
                statistic, pvalue = dm_test(errors**2, baseline**2, lag=max(14, horizon - 1))
                rows.append(
                    {
                        "feature": name,
                        "horizon": horizon,
                        "regime": regime,
                        "n": len(part),
                        "first_origin": str(part.origin.min().date()),
                        "last_origin": str(part.origin.max().date()),
                        "rmse": rmse,
                        "rw_rmse": rw_rmse,
                        "rmse_skill_vs_rw": 1 - rmse / rw_rmse,
                        "mae": errors.abs().mean(),
                        "rw_mae": baseline.abs().mean(),
                        "n_target_changed": int(part.actual.ne(part.rw).sum()),
                        "rmse_target_changed": float(
                            np.sqrt(np.mean(errors[part.actual.ne(part.rw)] ** 2))
                        ),
                        "rw_rmse_target_changed": float(
                            np.sqrt(np.mean(baseline[part.actual.ne(part.rw)] ** 2))
                        ),
                        "dm_stat_squared_loss_hac14": statistic,
                        "dm_pvalue_two_sided": pvalue,
                    }
                )
    result = pd.DataFrame(rows)
    result["dm_pvalue_bonferroni_24"] = (result.dm_pvalue_two_sided * 24).clip(upper=1)
    result.to_csv(output / "feature_screen.csv", index=False)
    dump(
        output / "feature_screen_method.json",
        {
            "features": list(x),
            "horizons": [1, 7],
            "cutoff_moscow": "15:00",
            "max_market_age_days": 10,
            "model": "univariate OLS with intercept on future log return; price=origin_price*exp(predicted_return)",
            "window_calendar_days": 730,
            "minimum_valid_training_rows": 365,
            "first_origin_index": 730,
            "last_training_origin": "forecast_origin - horizon; all training outcomes have occurred by forecast cutoff",
            "baseline": "random walk without drift on identical evaluation dates",
            "dm_hac_lag": 14,
            "multiple_testing": "Bonferroni factor 24 across six features, two horizons and two reported regimes",
            "warning": "Exploratory audit screen. Historical market availability follows stored conservative lag assumptions; data and features were not independently certified point-in-time. No causal interpretation or guarantee of future skill.",
        },
    )


if __name__ == "__main__":
    main()
