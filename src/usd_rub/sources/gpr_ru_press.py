"""Геополитический риск для России по заголовкам двух российских лент."""

import hashlib
import json
import re
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
from lxml import html

from usd_rub.storage import utc_now

DAILY = "gpr_russia_ru_press_daily"
MONTHLY = "gpr_russia_ru_press_monthly"
INTERFAX = "https://www.interfax.ru"
MK = "https://www.mk.ru"
COUNTS = ["interfax_articles", "interfax_hits", "mk_articles", "mk_hits"]

RUSSIA = re.compile(
    r"\b(?:(?:анти)?росси[а-я]*|рф|москв[а-я]*|кремл[а-я]*|путин[а-я]*|лавров[а-я]*|"
    r"петербург[а-я]*|белгород[а-я]*|курск[а-я]*|брянск[а-я]*|"
    r"севастопол[а-я]*|черноморск[а-я]* флот[а-я]*|вс рф|вкс рф)\b"
)
SVO = re.compile(r"\bсво\b|\b(?:военн\w* операци\w*|спецопераци\w*) на украин\w*\b")
RISK = re.compile(
    r"\b(?:войн(?:а|ы|е|у|ой|ою|ам|ами|ах)?|военн\w* конфликт\w*|вооруженн\w* конфликт\w*|"
    r"боевы\w* действи\w*|вторжени\w*|военн\w* (?:спец)?операци\w*|конфликт\w* на украин\w*|"
    r"обстрел\w*|авиаудар\w*|бомбардировк\w*|бомбежк\w*|мобилизаци\w*|"
    r"террорис\w*|теракт\w*|терроризм\w*|диверс\w*|эскалаци\w*|"
    r"перемири\w*|прекращени\w* огня|гонк\w* вооружени\w*|"
    r"ядерн\w* (?:оружи|угроз|удар|войн|испытани|сдерживани)\w*|"
    r"военн\w* (?:угроз|учени)\w*|(?:контр)?санкци(?:я\w*|и|й|онн\w*)|эмбарго)\b"
)
MILITARY = re.compile(
    r"\b(?:войск\w*|ракет\w*|дрон\w*|беспилотник\w*|бпла|пво|нато|всу|"
    r"военн\w*|арми[яиюей]\w*|армейск\w*|истребител\w*|вооружени\w*)\b"
)
ACTION = re.compile(
    r"\b(?:удар\w*|атак\w*|сби(?:л|т)\w*|сбив\w*|уничтож\w*|наступлени\w*|"
    r"вторг\w*|вторжени\w*|развер\w*|размест\w*|перебро\w*|стяну\w*|"
    r"наращ\w*|нараст\w*|перехват\w*|угро[зж]\w*|прорыв\w*|пора[зж]\w*)\b"
)
EXCLUDE = re.compile(
    r"\b(?:велик\w* отечественн\w*|втор\w* миров\w*|перв\w* миров\w*|"
    r"войн\w* цен|ценов\w* войн\w*|звездн\w* войн\w*|"
    r"кассов\w* сбор\w*|фильм\w*|сериал\w*|компьютерн\w* игр\w*|"
    r"футбол\w*|хокке\w*|допинг\w*|антидопинг\w*|"
    r"санкци\w* (?:суда|прокурор\w*)|с санкци\w* следств\w*)\b"
)


def risk_headline(title):
    """Совпадение российской привязки и темы риска в одном заголовке."""
    title = " ".join(title.lower().replace("ё", "е").split())
    svo = SVO.search(title)
    return bool(
        (svo or RUSSIA.search(title))
        and (svo or RISK.search(title) or (MILITARY.search(title) and ACTION.search(title)))
        and not EXCLUDE.search(title)
    )


def parse_mk(content, day):
    """Новостные карточки федеральной ленты за архивные сутки."""
    return {path: title for path, (_, title) in mk_archive(content, day).items()}


def mk_archive(content, day):
    """Заголовки и ссылки федеральной ленты за архивные сутки."""
    doc = html.fromstring(content.decode("utf-8"))
    expected = f"{MK}/news/{day:%Y/%m/%d}/"
    if doc.xpath('//link[@rel="canonical"]/@href') != [expected]:
        raise ValueError(f"МК: неверная архивная дата {day}")
    headlines = {}
    for node in doc.xpath('//a[contains(@class,"news-listing__item-link")]'):
        url = urlsplit(node.attrib["href"])
        if url.hostname != "www.mk.ru":
            continue
        title = node.xpath("string(./h3)").strip()
        if not title:
            raise ValueError(f"МК: отсутствует заголовок {url.path}")
        headlines[url.path] = (node.attrib["href"], title)
    if not headlines:
        raise ValueError(f"МК: пустая лента {day}")
    return headlines


def parse_interfax(content, day, page):
    """Заголовки и число страниц дневного архива."""
    headlines, pages = interfax_archive(content, day, page)
    return {key: title for key, (_, title) in headlines.items()}, pages


def interfax_archive(content, day, page):
    """Заголовки, ссылки и число страниц дневного архива."""
    doc = html.fromstring(content.decode("utf-8"))
    stamp = f'var data_date="{day:%Y-%m-%d}"'
    if not doc.xpath("//script[contains(text(), $stamp)]", stamp=stamp):
        raise ValueError(f"Интерфакс: неверная архивная дата {day}")
    pages = doc.xpath('//div[@class="pages"]/a')
    if [p.text for p in pages if p.get("class") == "active"] != [str(page)]:
        raise ValueError(f"Интерфакс: не подтверждена страница {page} за {day}")
    headlines = {}
    for node in doc.xpath('//div[@class="an"]/div[@data-id]'):
        title = node.xpath("string(./a/h3)").strip()
        if not title:
            raise ValueError(f"Интерфакс: отсутствует заголовок {node.get('data-id')}")
        headlines[node.attrib["data-id"]] = (node.xpath("string(./a/@href)"), title)
    if not headlines:
        raise ValueError(f"Интерфакс: пустая лента {day}, страница {page}")
    return headlines, max(int(p.text) for p in pages if (p.text or "").isdigit())


def collect_day(client, day):
    """Счётчики и исходные страницы обоих изданий."""
    suffix = f"/news/{day:%Y/%m/%d}/"
    payload = client.get(MK + suffix)
    mk = parse_mk(payload.content, day)
    inputs = {payload.url: payload.sha256}
    interfax = {}
    page, last = 1, 1
    while page <= last:
        url = INTERFAX + suffix + (f"page_{page}" if page > 1 else "")
        payload = client.get(url)
        titles, page_count = parse_interfax(payload.content, day, page)
        interfax.update(titles)
        inputs[payload.url] = payload.sha256
        last = max(last, page_count)
        page += 1
    counts = [
        len(interfax),
        sum(map(risk_headline, interfax.values())),
        len(mk),
        sum(map(risk_headline, mk.values())),
    ]
    return dict(zip(COUNTS, counts, strict=True)), inputs


def observation(root, series, day, counts, inputs, code_hash, ingested):
    """Наблюдение с воспроизводимым составом исходных данных."""
    manifest = {
        "series": series,
        "date": day.isoformat(),
        "counts": counts,
        "inputs": inputs,
        "code_sha256": code_hash,
    }
    content = json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()
    digest = hashlib.sha256(content).hexdigest()
    (root / digest).write_bytes(content)
    value = 50 * (
        counts["interfax_hits"] / counts["interfax_articles"]
        + counts["mk_hits"] / counts["mk_articles"]
    )
    return {
        "date": pd.Timestamp(day),
        "value": value,
        "series": series,
        "unit": "% of headlines, equal outlet weights",
        **counts,
        "source_url": f"raw/objects/{digest}",
        "raw_sha256": digest,
        "vintage": digest,
        "ingested_at": ingested,
        "availability_basis": "unknown",
        "available_at": ingested,
    }


def fetch_ru_press(client, start, end):
    """Дневные значения и полные месяцы до последней московской полуночи."""
    yesterday = pd.Timestamp.now(tz="Europe/Moscow").date() - timedelta(days=1)
    days = pd.date_range(start, min(pd.Timestamp(end).date(), yesterday))
    if days.empty:
        raise ValueError("В интервале нет завершённых московских суток")
    root = client.root / "raw" / "objects"
    root.mkdir(parents=True, exist_ok=True)
    code = Path(__file__).read_bytes()
    code_hash = hashlib.sha256(code).hexdigest()
    (root / code_hash).write_bytes(code)
    rows = []
    by_month = []
    for stamp in days:
        day = stamp.date()
        counts, inputs = collect_day(client, day)
        row = observation(root, DAILY, day, counts, inputs, code_hash, utc_now())
        rows.append(row)
        by_month.append(row)
        if stamp.is_month_end:
            if len(by_month) == stamp.days_in_month:
                totals = {key: sum(item[key] for item in by_month) for key in COUNTS}
                members = {str(item["date"].date()): item["raw_sha256"] for item in by_month}
                rows.append(
                    observation(
                        root, MONTHLY, day.replace(day=1), totals, members, code_hash, utc_now()
                    )
                )
            by_month.clear()
    return pd.DataFrame(rows)
