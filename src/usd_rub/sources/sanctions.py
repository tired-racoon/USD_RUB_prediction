"""Действующие санкционные списки и официальный журнал изменений UK."""

import hashlib
import json
import re
from io import BytesIO
from xml.etree import ElementTree as ET

import pandas as pd

URLS = {
    "ofac_sdn": "https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports/SDN.XML",
    "ofac_non_sdn": "https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports/CONSOLIDATED.XML",
    "uk": "https://sanctionslist.fcdo.gov.uk/docs/UK-Sanctions-List.csv",
    "eu": "https://webgate.ec.europa.eu/fsd/fsf/public/files/xmlFullSanctionsList_1_1/content?token=dG9rZW4tMjAxNw",
    "bis": "https://data.trade.gov/downloadable_consolidated_screening_list/v1/consolidated.csv",
}


def parse_ofac_list(payload):
    root = ET.fromstring(payload.content)
    for node in root.iter():
        node.tag = node.tag.split("}")[-1]
    published = pd.to_datetime(root.findtext("publshInformation/Publish_Date"), format="%m/%d/%Y")
    expected = int(root.findtext("publshInformation/Record_Count", "0"))
    entries = root.findall("sdnEntry")
    if len(entries) != expected or not entries:
        raise ValueError("OFAC: число записей не совпадает с заголовком XML")
    records = []
    for entry in entries:

        def full_name(node):
            return " ".join(filter(None, [node.findtext("firstName"), node.findtext("lastName")]))

        records.append(
            {
                "entity_id": entry.findtext("uid"),
                "name": full_name(entry),
                "subject_type": entry.findtext("sdnType"),
                "programmes": json.dumps([p.text for p in entry.findall("programList/program")]),
                "aliases": json.dumps(
                    [full_name(a) for a in entry.findall("akaList/aka")], ensure_ascii=False
                ),
                "vessel_type": entry.findtext("vesselInfo/vesselType"),
                "vessel_flag": entry.findtext("vesselInfo/vesselFlag"),
            }
        )
    frame = pd.DataFrame(records)
    if frame.entity_id.duplicated().any():
        raise ValueError("OFAC: повтор идентификатора сущности")
    return frame, published


def parse_uk(payload):
    report_date = payload.content.splitlines()[0].decode("utf-8-sig")
    if not report_date.startswith("Report Date:"):
        raise ValueError("UKSL: не найдена дата актуального отчёта")
    snapshot_date = pd.to_datetime(report_date.split(":", 1)[1].strip(), format="%d-%b-%Y")
    table = pd.read_csv(BytesIO(payload.content), skiprows=1, dtype=str, keep_default_na=False)
    columns = [
        "Unique ID",
        "Regime Name",
        "Designation Type",
        "Date Designated",
        "Last Updated",
        "Sanctions Imposed",
        "UK Statement of Reasons",
    ]
    if not {"Unique ID", "Regime Name"}.issubset(table):
        raise ValueError("UKSL: изменились ключевые столбцы")
    result = table[[column for column in columns if column in table]].copy()
    result["name"] = table[[f"Name {i}" for i in range(1, 7)]].apply(
        lambda row: " ".join(v for v in row if v), axis=1
    )
    result = result.drop_duplicates()
    grouped = []
    for identifier, part in result.groupby("Unique ID", sort=False):
        record = {"Unique ID": identifier}
        for column in columns[1:]:
            if column in part:
                values = list(dict.fromkeys(v.strip() for v in part[column] if v.strip()))
                if len(values) > 1:
                    raise ValueError(f"UKSL: конфликт {column} у {identifier}")
                record[column] = values[0] if values else ""
        names = list(dict.fromkeys(part.name))
        record["name"] = names[0]
        record["aliases"] = json.dumps(names[1:], ensure_ascii=False)
        grouped.append(record)
    result = pd.DataFrame(grouped)
    return result, snapshot_date


def parse_eu(payload):
    root = ET.fromstring(payload.content)
    date = pd.Timestamp(root.attrib["generationDate"])
    for node in root.iter():
        node.tag = node.tag.split("}")[-1]
    records = []
    for entity in root.findall(".//sanctionEntity"):
        regulation = entity.find("regulation")
        subject = entity.find("subjectType")
        records.append(
            {
                "entity_id": entity.attrib.get("euReferenceNumber"),
                "names": json.dumps(
                    [n.attrib.get("wholeName", "") for n in entity.findall("nameAlias")],
                    ensure_ascii=False,
                ),
                "subject_type": subject.attrib.get("code") if subject is not None else None,
                "programme": regulation.attrib.get("programme") if regulation is not None else None,
                "publication_date": regulation.attrib.get("publicationDate")
                if regulation is not None
                else None,
                "regulation_url": regulation.attrib.get("publicationUrl")
                if regulation is not None
                else None,
            }
        )
    if not records:
        raise ValueError("EU: в XML не найдены санкционные записи")
    return pd.DataFrame(records), date


def parse_bis(payload):
    table = pd.read_csv(BytesIO(payload.content), dtype=str, keep_default_na=False)
    if not {"source", "name", "start_date"}.issubset(table):
        raise ValueError("BIS/CSL: изменился формат")
    selected = table.loc[table["source"].str.contains("Entity List", case=False)]
    if selected.empty:
        raise ValueError("BIS: не найдена Entity List в действующем CSL")
    columns = [
        c
        for c in [
            "_id",
            "source",
            "entity_number",
            "name",
            "programs",
            "start_date",
            "end_date",
            "federal_register_notice",
            "license_requirement",
            "license_policy",
            "source_information_url",
        ]
        if c in selected
    ]
    stamp = pd.to_datetime(payload.headers.get("last-modified"), errors="coerce", utc=True)
    return selected[columns], stamp


def fetch_snapshots(client, names):
    folder = client.root / "entities"
    folder.mkdir(parents=True, exist_ok=True)
    reports = {}
    for name in names:
        try:
            payload = client.get(URLS[name])
            frame, stamp = {
                "uk": parse_uk,
                "eu": parse_eu,
                "bis": parse_bis,
                "ofac_sdn": parse_ofac_list,
                "ofac_non_sdn": parse_ofac_list,
            }[name](payload)
            reference = pd.Timestamp(payload.ingested_at).tz_localize(None)
            if pd.isna(stamp) or reference - pd.Timestamp(stamp).tz_localize(None) > pd.Timedelta(
                days=90
            ):
                raise ValueError(f"Дата снимка старше 90 дней или неизвестна: {stamp}")
            frame["source_url"] = payload.url
            frame["raw_sha256"] = payload.sha256
            frame["available_at"] = payload.ingested_at
            frame["snapshot_date"] = str(stamp)
            frame.to_csv(folder / f"{name}_{payload.sha256}.csv", index=False)
            frame.to_csv(folder / f"{name}_current.csv", index=False)
            reports[name] = {
                "status": "snapshot_only",
                "rows": len(frame),
                "snapshot_date": str(stamp),
                "raw_sha256": payload.sha256,
                "history_at_least_10y": False,
                "warning": "Текущий список не восстанавливает снятые санкции и прошлые версии",
            }
        except (ValueError, RuntimeError, OSError) as exc:
            reports[name] = {"status": "error", "error": str(exc)}
    (folder / "report.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2))
    return reports


def fetch_uk_changes(client):
    payload = client.get(
        "https://www.gov.uk/api/content/government/publications/the-uk-sanctions-list"
    )
    doc = json.loads(payload.content)
    updated = pd.Timestamp(doc["public_updated_at"])
    if pd.Timestamp(payload.ingested_at) - updated > pd.Timedelta(days=90):
        raise ValueError("Журнал UK не обновлялся более 90 дней")
    changes = doc.get("details", {}).get("change_history", [])
    folder = client.root / "documents" / "uk_sanctions"
    folder.mkdir(parents=True, exist_ok=True)
    result = {
        "source_url": payload.url,
        "raw_sha256": payload.sha256,
        "updated_at": str(updated),
        "ingested_at": payload.ingested_at,
        "changes": changes,
        "details": doc.get("details", {}),
        "scope": "all_uk_regimes",
        "russia_changes": [r for r in changes if re.search(r"Russia", r.get("note", ""), re.I)],
        "history_limitation": "Журнал изменений страницы не заменяет полный архив санкционных уведомлений",
    }
    (folder / "uksl_changes.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    events = []
    coverage_start = doc.get("first_published_at") or min(
        (r["public_timestamp"] for r in changes), default=str(updated)
    )
    for change in result["russia_changes"]:
        stamp = pd.Timestamp(change["public_timestamp"])
        key = hashlib.sha256((str(stamp) + change["note"]).encode()).hexdigest()
        events.append(
            {
                "event_id": "uk_" + key,
                "event_key": "uk_" + key,
                "event_type": "uk_russia_list_update",
                "event_time": str(stamp),
                "effective_time": str(stamp),
                "publication_time": str(stamp),
                "source_url": payload.url,
                "raw_sha256": payload.sha256,
                "value": 1,
                "unit": "change notice",
                "evidence": change["note"],
                "availability_basis": "reported_change_history_timestamp",
                "coverage_start": coverage_start,
                "coverage_end": payload.ingested_at,
            }
        )
    calendars = client.root / "calendars"
    calendars.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        events,
        columns=[
            "event_id",
            "event_key",
            "event_type",
            "event_time",
            "effective_time",
            "publication_time",
            "source_url",
            "raw_sha256",
            "value",
            "unit",
            "evidence",
            "availability_basis",
            "coverage_start",
            "coverage_end",
        ],
    ).to_csv(calendars / "uk_sanctions_events.csv", index=False)
    return result
