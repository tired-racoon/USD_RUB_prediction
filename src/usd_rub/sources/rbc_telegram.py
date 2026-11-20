"""Заголовки из публичного архива Telegram-канала РБК."""

import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
from lxml import html

from usd_rub.sources.gpr_ru_press import risk_headline
from usd_rub.sources.news_archive import COLUMNS, ECONOMY, MOSCOW, POLITICS, write_csv
from usd_rub.storage import Client

CHANNEL = "https://t.me/s/rbc_news"


def parse_rbc_page(content, before):
    """Прочитать посты и проверить, что страница предшествует курсору."""
    doc = html.fromstring(content.decode("utf-8"))
    posts = []
    for node in doc.xpath('//div[starts-with(@data-post,"rbc_news/")]'):
        number = int(node.attrib["data-post"].split("/")[1])
        stamps = node.xpath('.//a[contains(@class,"tgme_widget_message_date")]/time/@datetime')
        if not stamps:
            continue
        first = node.xpath('.//div[contains(@class,"tgme_widget_message_text")]')
        headline = ""
        if first:
            intro = re.split(r"<br\b", html.tostring(first[0], encoding="unicode"), maxsplit=1)[0]
            headline = " ".join(html.fromstring(intro).text_content().split())[:240].strip()
        posted = datetime.fromisoformat(stamps[-1]).astimezone(MOSCOW)
        posts.append((number, posted, headline))
    if not posts or any(number >= before for number, _, _ in posts):
        raise ValueError(f"РБК: не подтверждена страница перед постом {before}")
    return posts


def collect_rbc(root, start, stop):
    """Пройти диапазон идентификаторов по ссылкам старых страниц."""

    def segment(first, last):
        selected = {}
        with Client(root) as client:
            cursor = last
            while cursor > first:
                url = f"{CHANNEL}?{urlencode({'before': cursor})}"
                payload = client.get(url)
                posts = parse_rbc_page(payload.content, cursor)
                for number, published, headline in posts:
                    if first <= number < last:
                        selected[number] = (published, headline, url, payload.sha256)
                cursor = min(number for number, _, _ in posts)
                time.sleep(0.75)
        return selected

    segments = [(n, min(n + 1000, stop)) for n in range(start, stop, 1000)]
    posts = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for part in pool.map(lambda limits: segment(*limits), segments):
            posts.update(part)
            print(f"РБК: {len(posts):,} постов", flush=True)
    if not posts:
        raise ValueError("РБК: в заданном диапазоне нет постов")
    stamps = [published.date() for published, _, _, _ in posts.values()]
    first, last = min(stamps), max(stamps)
    rows = []
    for number, (published, headline, archive, digest) in sorted(posts.items()):
        day = published.date()
        if not first < day < last:
            continue
        if POLITICS.search(headline) or risk_headline(headline):
            topic = "politics"
        elif ECONOMY.search(headline):
            topic = "economy"
        else:
            continue
        rows.append(
            (
                pd.Timestamp(day),
                "rbc",
                topic,
                headline,
                f"https://t.me/rbc_news/{number}",
                archive,
                digest,
            )
        )
    if not rows:
        raise ValueError("В диапазоне РБК нет полных суток с подходящими постами")
    return pd.DataFrame(rows, columns=COLUMNS)


def read_rbc(root, bounds, dates):
    """Использовать сохранённую подборку публикаций из указанного диапазона."""
    start, stop = (int(x) for x in bounds.split(":"))
    if not 1 <= start < stop:
        raise ValueError("--rbc-ids ожидает два возрастающих идентификатора START:STOP")
    path = Path(root) / "news" / "inputs" / f"rbc_{start}_{stop}.csv.gz"
    if not path.exists():
        write_csv(collect_rbc(root, start, stop), path)
    frame = pd.read_csv(path, parse_dates=["date"], keep_default_na=False)
    frame = frame.loc[frame.date.between(dates[0], dates[-1])]
    if frame.empty:
        raise ValueError("Нет публикаций РБК в запрошенные даты")
    return frame, path
