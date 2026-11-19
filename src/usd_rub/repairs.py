"""Повторная нормализация сохранённых ответов с резервной копией результатов."""

import json
import os
import shutil
from pathlib import Path

import pandas as pd

from usd_rub.availability import enrich_availability, restore_document_times
from usd_rub.sources.china import parse_pmi
from usd_rub.sources.ofac import action_events, classify_action
from usd_rub.sources.reports import parse_orfr
from usd_rub.sources.tax_calendar import parse_calendar
from usd_rub.sources.urals import parse_urals
from usd_rub.storage import Payload, input_fingerprints, read_series, utc_now


def repair_saved_data(root="data", as_of=None):
    root = Path(root)
    reference = pd.Timestamp(as_of or pd.Timestamp.now(tz="Europe/Moscow").date())
    backup = root / "repairs" / str(reference.date()) / "before"
    backup.mkdir(parents=True, exist_ok=True)
    for name in ["normalized", "features", "calendars", "documents", "entities"]:
        if (root / name).exists() and not (backup / name).exists():
            shutil.copytree(root / name, backup / name)
    report = {
        "started_at": utc_now(),
        "as_of": str(reference.date()),
        "changes": {},
        "unresolved": [],
    }

    def replace(path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        old = backup / path.relative_to(root)
        if path.exists() and not old.exists():
            old.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, old)
        if path.exists() and path.read_text() == content:
            return
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content)
        temporary.replace(path)

    def save(frame):
        for name, part in frame.groupby("series"):
            part = part.copy()
            part["date"] = pd.to_datetime(part.date).dt.strftime("%Y-%m-%d")
            part = part.sort_values(["date", "ingested_at"]).drop_duplicates(["date", "vintage"])
            replace(root / "normalized" / f"{name}.csv", part.to_csv(index=False))
            report["changes"][name] = {"rows": len(part), "dates": part.date.nunique()}

    metadata, first_metadata, by_url = {}, {}, {}
    for line in (root / "raw" / "history.jsonl").read_text().splitlines():
        item = json.loads(line)
        metadata[item["sha256"]] = item
        first_metadata.setdefault((item["sha256"], item["url"]), item)
        by_url[item["url"]] = item["sha256"]

    def payload(digest, url=None, ingested=None):
        item = metadata.get(digest, {})
        return Payload(
            (root / "raw" / "objects" / digest).read_bytes(),
            url or item["url"],
            digest,
            ingested or first_metadata.get((digest, url or item["url"]), item)["ingested_at"],
            item.get("headers", {}),
        )

    report["decision_publication_times"] = restore_document_times(root)
    key_rate = read_series(root, "key_rate")
    enriched = enrich_availability(key_rate, root)
    save(enriched)
    report["key_rate_unknown_dates"] = int(
        enriched.loc[enriched.availability_basis.eq("unknown"), "date"].nunique()
    )

    pieces = []
    for path in sorted((root / "documents" / "cbr_orfr").glob("*.json")):
        if path.name == "manifest.json":
            continue
        document = json.loads(path.read_text())
        part = parse_orfr(
            payload(document["raw_sha256"], document["source_url"], document["ingested_at"]),
            pd.Timestamp(document["period"]),
            document["pages"],
        )
        if not part.empty:
            pieces.append(part)
    if pieces:
        data = pd.concat(pieces, ignore_index=True)
        save(data)
        if "household_net_fx_purchases" in set(data.series):
            assert (
                not data.loc[data.series.eq("household_net_fx_purchases"), "evidence"]
                .str.contains("с начала года", case=False)
                .any()
            )

    for name, parser in [
        ("urals_official_monthly", parse_urals),
        ("china_manufacturing_pmi", parse_pmi),
    ]:
        current = read_series(root, name)
        parts = [
            parser(payload(row.raw_sha256, row.source_url, row.ingested_at))
            for row in current.drop_duplicates("raw_sha256").itertuples()
        ]
        data = pd.concat(parts, ignore_index=True)
        save(data.loc[data.date.between(current.date.min(), reference)])

    fns_manifest = json.loads((root / "calendars" / "fns_manifest.json").read_text())
    calendar_frames, events = [], []
    for url in fns_manifest["releases"]:
        if url not in by_url:
            report["unresolved"].append({"source": url, "reason": "Нет сохранённого ответа"})
            continue
        part, records = parse_calendar(payload(by_url[url], url))
        calendar_frames.append(part)
        events.extend(records)
    if calendar_frames:
        data = pd.concat(calendar_frames, ignore_index=True)
        save(data.loc[data.date.between("2015-01-01", reference)])
        table = (
            pd.DataFrame(events)
            .drop_duplicates("event_id")
            .sort_values(["publication_time", "event_id"])
        )
        replace(root / "calendars" / "fns_events.csv", table.to_csv(index=False))
        missing = sorted(set(range(2015, reference.year + 1)) - set(table.calendar_year))
        fns_manifest.update(
            events=len(table),
            missing_calendar_years=missing,
            publication_time_basis="assumed_archive_release_date_end_of_day",
            warning="Нет заполнения пропущенных лет; дата в имени файла требует допущения",
        )
        replace(
            root / "calendars" / "fns_manifest.json",
            json.dumps(fns_manifest, ensure_ascii=False, indent=2),
        )
        report["tax_calendar"] = {
            "events_and_coverage_markers": len(table),
            "missing_years": missing,
            "duplicate_event_ids": int(table.event_id.duplicated().sum()),
        }
        if missing:
            report["unresolved"].append(
                {
                    "source": "fns_calendar",
                    "missing_years": missing,
                    "reason": "В сохранённых официальных выпусках год отсутствует; признакам запрещено переносить соседние даты",
                }
            )

    all_actions = {}
    action_starts, action_ends = [], []
    for path in sorted((root / "documents").glob("ofac*.json")):
        document = json.loads(path.read_text())
        if document.get("start"):
            action_starts.append(document["start"])
        if document.get("ingested_at"):
            action_ends.append(document["ingested_at"])
        actions = document.get("actions", [])
        for action in actions:
            action["classification"] = classify_action(action)
            previous = all_actions.get(action["source_url"])
            if previous is None or len(str(action)) > len(str(previous)):
                all_actions[action["source_url"]] = action
        replace(path, json.dumps(document, ensure_ascii=False, indent=2))
    for path in (root / "documents" / "ofac_archive").glob("*.json"):
        action = json.loads(path.read_text())
        action["classification"] = classify_action(action)
        replace(path, json.dumps(action, ensure_ascii=False, indent=2))
    actions = list(all_actions.values())
    replace(
        root / "calendars" / "ofac_events.csv",
        action_events(
            actions,
            min(action_starts) if action_starts else None,
            max(action_ends) if action_ends else None,
        ).to_csv(index=False),
    )
    report["ofac"] = {
        "unique_documents": len(actions),
        "title_explicit_russia_documents": sum(
            classify_action(r)["russia_relevance"] == "title_explicit" for r in actions
        ),
        "warning": "Классификация по правилам требует проверки; число документов не является числом санкций или сущностей",
    }
    old_uk = root / "documents" / "uk_sanctions" / "russia_changes.json"
    if old_uk.exists():
        document = json.loads(old_uk.read_text())
        document.update(
            status="archived_discontinued_source",
            replacement="https://www.gov.uk/api/content/government/publications/the-uk-sanctions-list",
        )
        replace(old_uk, json.dumps(document, ensure_ascii=False, indent=2))

    names = [
        f"fnb_{name}_rub_bn"
        for name in ["opening", "closing", "inflows", "withdrawals", "revaluation"]
    ]
    values = {}
    for name in names:
        frame = (
            read_series(root, name).sort_values("ingested_at").drop_duplicates("date", keep="last")
        )
        values[name] = frame.set_index("date").value
    table = pd.concat(values, axis=1).dropna()
    table["unreconciled_residual_rub_bn"] = (
        table.fnb_closing_rub_bn
        - table.fnb_opening_rub_bn
        - table.fnb_inflows_rub_bn
        + table.fnb_withdrawals_rub_bn
        - table.fnb_revaluation_rub_bn
    )
    replace(root / "reports" / "fnb_reconciliation.csv", table.to_csv())
    failed = table.unreconciled_residual_rub_bn.abs().gt(0.031)
    report["fnb"] = {
        "months": len(table),
        "unreconciled_months": int(failed.sum()),
        "max_abs_residual_rub_bn": float(table.unreconciled_residual_rub_bn.abs().max()),
        "flow_series_status": "excluded_until_reconciled",
        "stock_series_status": "retained_as_source_values",
    }
    if failed.any():
        report["unresolved"].append(
            {
                "source": "minfin_fnb",
                "reason": "Баланс не сходится в исходной таблице; потоки исключены, исходные числа сохранены",
            }
        )
    report["unknown_publication_policy"] = (
        "Неизвестные публикации не переносятся в прошлое; допущения доступны только в исследовательской сборке"
    )
    report["unresolved"].append(
        {
            "source": "historical_publication_times_and_vintages",
            "fred_api_key_configured": bool(os.environ.get("FRED_API_KEY")),
            "reason": "Для части рядов нет подтверждённых исторических публикаций и винтажей. Неизвестные даты сохраняются и не допускаются задним числом",
        }
    )
    report["finished_at"] = utc_now()
    report["inputs_after_repair"] = input_fingerprints((root / "normalized").glob("*.csv"))
    path = root / "reports" / f"repairs_{reference.date()}.json"
    replace(path, json.dumps(report, ensure_ascii=False, indent=2))
    return report
