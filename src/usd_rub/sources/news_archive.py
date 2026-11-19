"""Архив политических и экономических заголовков российских изданий."""

import hashlib
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urljoin, urlsplit
from zoneinfo import ZoneInfo

import pandas as pd

from usd_rub.sources.gpr_ru_press import INTERFAX, MK, interfax_archive, mk_archive, risk_headline
from usd_rub.storage import Client, DownloadError

MOSCOW = ZoneInfo("Europe/Moscow")
PRESS = ("interfax", "mk")
COLUMNS = ["date", "source", "topic", "headline", "url", "archive_url", "archive_sha256"]
SMARTLAB_DATASET = "https://huggingface.co/datasets/Kasymkhan/RussianFinancialNews"
SMARTLAB_REVISION = "f83da5c05c82c9086806cf90ded4155052f9605c"
ECONOMY = re.compile(
    r"\b(?:экономик\w*|экономическ\w*|рубл\w*|доллар\w*|юан\w*|валют\w*|"
    r"ставк\w*|бирж\w*|акци[ияй]\w*|облигац\w*|офз|инфляц\w*|ввп|"
    r"нефт\w*|газпром\w*|роснефт\w*|опек\w*|экспорт\w*|импорт\w*|"
    r"банк\w*|минфин\w*|бюджет\w*|налог\w*|рынк\w*|торговл\w*)\b",
    re.IGNORECASE,
)
POLITICS = re.compile(
    r"\b(?:путин\w*|кремл\w*|лавров\w*|мид|госдум\w*|совфед\w*|"
    r"санкци\w*|эмбарго|переговор\w*|саммит\w*|нато|оон|геополит\w*|"
    r"внешн\w* политик\w*|миров\w* соглашени\w*|пошлин\w*)\b",
    re.IGNORECASE,
)


def press_topic(source, url, title):
    """Рубрика издания или совпадение по теме для общей ленты."""
    path = urlsplit(url)
    section = path.path.split("/")[1]
    if source == "mk":
        return {"economics": "economy", "politics": "politics"}.get(section)
    if path.hostname != "www.interfax.ru":
        return None
    if section in {"business", "economics"}:
        return "economy"
    if section not in {"russia", "world", "moscow"}:
        return None
    if POLITICS.search(title) or risk_headline(title):
        return "politics"
    if ECONOMY.search(title):
        return "economy"
    return None


def collect_press_day(client, source, day):
    """Дневная лента с проверкой даты и всех страниц Интерфакса."""
    suffix = f"/news/{day:%Y/%m/%d}/"
    rows = {}
    page, last, seen = 1, 1, set()
    while page <= last:
        base = MK if source == "mk" else INTERFAX
        url = base + suffix + (f"page_{page}" if page > 1 else "")
        payload = client.get(url)
        if source == "mk":
            articles = mk_archive(payload.content, day)
        else:
            articles, pages = interfax_archive(payload.content, day, page)
            last = max(last, pages)
        if not set(articles) - seen:
            raise ValueError(f"{source}: повтор страницы {page} за {day}")
        seen.update(articles)
        for article_id, (link, headline) in articles.items():
            if not link:
                raise ValueError(f"{source}: отсутствует ссылка {article_id}")
            link = urljoin(base, link)
            headline = " ".join(headline.split())
            topic = press_topic(source, link, headline)
            if topic:
                rows[article_id] = (day, source, topic, headline, link, url, payload.sha256)
        page += 1
    return list(rows.values())


def write_csv(frame, path, *, index=False):
    """Атомарно записать CSV или CSV.GZ."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False) as file:
        temporary = Path(file.name)
    try:
        frame.to_csv(temporary, index=index)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def collect_press(root, start, end, workers):
    """Собрать корпус и состояние каждого архива по дням."""
    dates = pd.date_range(start, end)

    def download_month(month_days):
        rows, coverage = [], []
        with Client(root) as client:
            for stamp in month_days:
                day = stamp.date()
                for source in PRESS:
                    try:
                        articles = collect_press_day(client, source, day)
                    except (DownloadError, ValueError) as exc:
                        coverage.append((day, source, "error", None, str(exc)))
                    else:
                        rows.extend(articles)
                        coverage.append((day, source, "ok", len(articles), None))
        return rows, coverage

    months = [part for _, part in dates.to_series().groupby(dates.to_period("M"))]
    rows, coverage = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for month_days, (articles, states) in zip(
            months, pool.map(download_month, months), strict=True
        ):
            rows.extend(articles)
            coverage.extend(states)
            if month_days.iloc[-1].month in {6, 12} or month_days is months[-1]:
                print(f"{month_days.iloc[-1].date()}: {len(rows):,} заголовков", flush=True)
    return (
        pd.DataFrame(rows, columns=COLUMNS),
        pd.DataFrame(coverage, columns=["date", "source", "status", "articles", "error"]),
    )


def read_smartlab(paths, dates):
    """Заголовки Смартлаба из опубликованного финансового корпуса."""
    frames = []
    for path in paths:
        frame = pd.read_parquet(path, columns=["source", "date", "title", "body"])
        frame = frame.loc[frame.source.eq("smart_lab")].copy()
        frame["date"] = pd.to_datetime(frame.date, format="mixed").dt.normalize()
        frame = frame.loc[frame.date.between(dates[0], dates[-1])]
        title = frame.title.fillna("").str.strip()
        frame["headline"] = title.where(~title.str.lower().isin(["", "no title"]), frame.body)
        frame["headline"] = frame.headline.fillna("").str.split().str.join(" ")
        frame["topic"] = "economy"
        frame["source"] = "smartlab"
        frame["url"] = ""
        frame["archive_url"] = f"{SMARTLAB_DATASET}/tree/{SMARTLAB_REVISION}"
        with Path(path).open("rb") as file:
            frame["archive_sha256"] = hashlib.file_digest(file, "sha256").hexdigest()
        frames.append(frame.loc[frame.headline.ne(""), COLUMNS])
    result = pd.concat(frames, ignore_index=True).drop_duplicates(["date", "headline"])
    if result.empty:
        raise ValueError("В переданных файлах нет заголовков Смартлаба за указанный период")
    return result.sort_values(["date", "headline"], ignore_index=True)


def fetch_news(args):
    """Обновить корпус и отчёт о покрытии."""
    start, end = datetime.fromisoformat(args.start).date(), datetime.fromisoformat(args.end).date()
    yesterday = datetime.now(MOSCOW).date() - timedelta(days=1)
    end = min(end, yesterday)
    if end < start:
        raise ValueError("В интервале нет завершённых московских суток")
    output = Path(args.output) if args.output else Path(args.data_dir) / "news"
    corpus, coverage = collect_press(args.data_dir, start, end, args.workers)
    write_csv(corpus, output / "headlines.csv.gz")
    write_csv(coverage, output / "coverage.csv")
    errors = int(coverage.status.eq("error").sum())
    print(f"{output}: {len(corpus):,} заголовков, ошибок архива {errors}", flush=True)
    return int(bool(errors))
