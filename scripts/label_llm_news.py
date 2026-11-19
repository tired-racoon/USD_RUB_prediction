"""Разметить валютные события в архиве новостей и сохранить проверяемый кэш."""

import argparse
import asyncio
import hashlib
import json
import os
import re
from pathlib import Path

import httpx
import pandas as pd

from usd_rub.news_features import headline_key
from usd_rub.storage import input_fingerprints, utc_now

MODEL = "qwen/qwen3.5-flash"
INPUT_RUB, OUTPUT_RUB = 4, 44
MAX_TITLES = 24
MAX_OUTPUT = 450
STAGES = ("implemented", "announced", "proposed", "observation")
CHANNELS = {
    "payments": re.compile(
        r"санкци\w*|\bswift\b|вторичн\w* огранич|трансгранич\w*|"
        r"расчет\w* в юан|корреспондентск\w* счет|замороз\w* актив|платеж\w* за рубеж"
    ),
    "export": re.compile(
        r"российск\w* нефт|нефт\w* из россии|\b(?:urals|опек)\b|"
        r"нефт\w* эмбарго|потол\w* цен|нефтегазов\w* доход|экспорт\w* (?:нефт|газ|угл)"
    ),
    "rates": re.compile(
        r"ключев\w* ставк|ставк\w* (?:цб|банка россии|фрс)|"
        r"(?:цб|банк россии)\b.{0,24}инфляц|инфляци\w* в росси"
    ),
    "fx_policy": re.compile(
        r"бюджетн\w* правил|валютн\w* выруч|валютн\w* интервенц|"
        r"(?:продаж|покуп)\w* валют|ограничени\w* на перевод|"
        r"репатриац\w*|стабилизац\w* рубл|операци\w* с юан"
    ),
    "geopolitics": re.compile(
        r"перемири\w*|мирн\w* переговор\w*|эскалац\w*|прекращени\w* огня|"
        r"военн\w* операци\w* на украин|мобилизац\w*|аннекси\w*|вторжени\w*"
    ),
}
PROPOSED = re.compile(r"обсужда\w*|предлож\w*|не исключ\w*|призва\w*|рассматрива\w*")
ANNOUNCED = re.compile(r"объяви\w*|введет|вводит|повысят|снизят|принял решени\w*")
IMPLEMENTED = re.compile(
    r"повысил\w*|понизил\w*|снизил\w*|вступил\w*|ввел\w*|продлил\w*|отменил\w*"
)
COMMENT = re.compile(r"\b(?:эксперт\w*|аналитик\w*|омбудсмен\w*|прогноз\w*|предрек\w*)\b")
FX_MOVE = re.compile(
    r"^(?:(?:официальн\w* )?курс (?:доллар\w*|рубл\w*|юан\w*|евро)|"
    r"рубль|доллар|юань|евро)\b.{0,50}\b"
    r"(?:укреп\w*|ослаб\w*|снижен\w*|снизил\w*|вырос\w*|подрос\w*|"
    r"паден\w*|упал\w*|торгу\w*|подорожал\w*|стоит|поднял\w*)"
)
PROMPT = (
    "Разметь данные новостные заголовки. Выделяй конкретные факты, потенциально "
    "меняющие валютный рынок России. Игнорируй прогнозы, реакции на уже принятое решение, "
    "движение курса, заявления о личных санкциях без связи с платежами, мнения и повторы. "
    "Выбери один канал на событие: payments — банковские ограничения, санкции и расчёты "
    "(включая SWIFT); export — экспорт сырья; rates — решения ЦБ, ФРС, месячная инфляция РФ; "
    "fx_policy — валютная выручка и валютные операции; geopolitics — эскалация, переговоры, "
    "перемирие. proposed — обсуждение или предложение; announced — решение объявлено; "
    "implemented — мера явно действует или решение уже принято и выполнено; observation — "
    "опубликован факт о торговле, платежах или инфляции. При сомнении пропусти событие. "
    'Ответ только JSON {"events":[{"channel":"payments","stage":"announced","ids":[1]}]}. '
    "Используй только ID данных заголовков. Максимум восемь событий, пустой список допустим."
)
RETRY_SUFFIX = (
    " Всегда отвечай объектом с ключом events. Если подходящих событий нет, "
    'напечатай буквально {"events":[]}. Пример с одним событием: '
    '{"events":[{"channel":"rates","stage":"implemented","ids":[1]}]}.'
)
FINAL_SUFFIX = (
    " Оставь не более пяти событий. Поля каждого элемента: только channel, stage, ids. "
    "В channel одно из payments, export, rates, fx_policy, geopolitics. В stage одно из "
    "implemented, announced, proposed, observation. В ids только целые номера из входа."
)


def keyword_stage(text):
    """Словарная оценка стадии для контрольного признака."""
    for pattern, stage in (
        (PROPOSED, "proposed"),
        (IMPLEMENTED, "implemented"),
        (ANNOUNCED, "announced"),
    ):
        if pattern.search(text):
            return stage
    return "observation"


def daily_candidates(frame):
    """Выбрать ограниченную подборку заголовков с охватом разных каналов."""
    unique = {}
    for row in frame.itertuples(index=False):
        title = " ".join(row.headline.split())
        text = title.casefold().replace("ё", "е")
        if not text or FX_MOVE.search(text) or COMMENT.search(text):
            continue
        flags = [name for name, pattern in CHANNELS.items() if pattern.search(text)]
        if flags:
            unique.setdefault(
                headline_key(title),
                (title[:220], row.source, row.url, flags, keyword_stage(text)),
            )
    ordered = sorted(unique.values(), key=lambda row: (row[4] == "observation", row[1], row[0]))
    candidates = {}
    for source in sorted({row[1] for row in ordered}):
        for channel in CHANNELS:
            pool = [row for row in ordered if row[1] == source and channel in row[3]]
            candidates[(source, channel)] = pool
    selected = {}
    for rank in range(MAX_TITLES):
        for pool in candidates.values():
            if rank < len(pool):
                selected.setdefault(pool[rank][0], pool[rank])
                if len(selected) == MAX_TITLES:
                    break
        if len(selected) == MAX_TITLES:
            break
    return [
        {"id": i, "headline": t, "source": s, "url": u, "channels": c, "stage": st}
        for i, (t, s, u, c, st) in enumerate(selected.values(), 1)
    ]


def validate_events(content, ids):
    """Проверить поля и доказательства событий до записи признаков."""
    packet = json.loads(content)
    events = packet["events"]
    if not isinstance(events, list) or len(events) > 8:
        raise ValueError("Допустимо до восьми событий")
    for event in events:
        if (
            set(event) != {"channel", "stage", "ids"}
            or event["channel"] not in CHANNELS
            or event["stage"] not in STAGES
            or not isinstance(event["ids"], list)
            or not event["ids"]
            or any(type(i) is not int or i not in ids for i in event["ids"])
        ):
            raise ValueError("Некорректное событие")
    return list({json.dumps(event, sort_keys=True): event for event in events}.values())


def api_key(path):
    """Получить ключ из окружения или локального файла."""
    if value := os.environ.get("VSELLM_API_KEY"):
        return value
    for row in path.read_text().splitlines():
        key, sep, value = row.partition("=")
        if sep and key.strip() == "VSELLM_API_KEY" and value.strip():
            return value.strip().strip("\"'")
    raise ValueError("В окружении или .env отсутствует VSELLM_API_KEY")


async def request(client, selection, responses, recovering):
    """Отправить подборку с ограничением ответа и паузой при перегрузке."""
    route = "retry" if recovering else "initial"
    if not selection:
        return [], route
    text = "\n".join(f"{row['id']}. {row['headline']}" for row in selection)
    body = {
        "model": MODEL,
        "messages": [{"role": "system", "content": PROMPT}, {"role": "user", "content": text}],
        "temperature": 0,
        "reasoning_effort": "none",
        "max_completion_tokens": MAX_OUTPUT,
        "response_format": {"type": "json_object"},
    }
    for attempt in range(4):
        if route == "retry" and attempt >= 2:
            route = "final_retry"
        suffix = RETRY_SUFFIX if route != "initial" else ""
        if route == "final_retry":
            suffix += FINAL_SUFFIX
        body["messages"][0]["content"] = PROMPT + suffix
        try:
            response = await client.post("chat/completions", json=body)
        except httpx.TransportError:
            if attempt == 3:
                raise
            await asyncio.sleep(2**attempt)
            continue
        if response.status_code in {429, 500, 502, 503, 504} and attempt < 3:
            retry = response.headers.get("retry-after", "")
            await asyncio.sleep(
                max(2**attempt, min(int(retry), 60)) if retry.isdecimal() else 2**attempt
            )
            continue
        response.raise_for_status()
        decoded = response.json()
        responses.append(decoded.get("usage") or {})
        try:
            choice = decoded["choices"][0]
            if choice["finish_reason"] != "stop":
                raise ValueError("Незавершённый ответ")
            events = validate_events(choice["message"]["content"], {row["id"] for row in selection})
            return events, route
        except (ValueError, TypeError, KeyError, IndexError):
            if attempt == 3:
                raise
            route = "retry"
            await asyncio.sleep(1)
    raise RuntimeError("Повторы запроса исчерпаны")


def cost(usage):
    """Оценить стоимость по токенам, возвращённым API."""
    return (
        INPUT_RUB * usage.get("prompt_tokens", 0) + OUTPUT_RUB * usage.get("completion_tokens", 0)
    ) / 1_000_000


def save(path, payload):
    """Записать состояние запроса с атомарной заменой файла."""
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


async def label(args):
    """Разметить календарные дни с переиспользованием готовых ответов."""
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / "days"
    cache.mkdir(exist_ok=True)
    corpus = pd.read_csv(args.corpus, usecols=["date", "headline", "source", "url"])
    corpus = corpus.loc[corpus.date.between(args.start, args.end)]
    if corpus.empty:
        raise ValueError("Пустой диапазон архива")
    selections = {day: daily_candidates(frame) for day, frame in corpus.groupby("date")}
    print(f"Подготовлены {len(selections)} дней", flush=True)
    pending = []
    spent, done, failures = 0.0, 0, 0
    for day, titles in selections.items():
        encoded = json.dumps([MODEL, PROMPT, MAX_OUTPUT, titles], ensure_ascii=False).encode()
        fingerprint = hashlib.sha256(encoded).hexdigest()
        path = cache / f"{day}.json"
        if path.exists():
            original = json.loads(path.read_text())
            if original["fingerprint"] != fingerprint:
                raise ValueError(f"Промпт или данные изменились для {day}, укажите новый каталог")
            spent += original["estimated_rub"]
            if original["status"] == "ok":
                done += 1
                continue
        else:
            original = {"estimated_rub": 0.0}
        pending.append((day, titles, path, fingerprint, original["estimated_rub"], path.exists()))

    queue = iter(pending)
    fatal = None
    async with httpx.AsyncClient(
        base_url="https://api.vsellm.ru/v1",
        headers={"Authorization": f"Bearer {api_key(args.env)}"},
        timeout=55,
        limits=httpx.Limits(max_connections=args.workers, max_keepalive_connections=args.workers),
    ) as client:

        async def worker():
            nonlocal done, failures, fatal, spent
            while fatal is None and spent < args.budget_rub:
                item = next(queue, None)
                if item is None:
                    return
                day, titles, path, fingerprint, previous_cost, recovering = item
                usage = []
                record = {"date": day, "fingerprint": fingerprint, "selection": titles}
                try:
                    events, route = await request(client, titles, usage, recovering)
                    record.update(status="ok", events=events, prompt_route=route)
                    done += 1
                except (httpx.HTTPError, ValueError, TypeError, KeyError, IndexError) as exc:
                    status = (
                        exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
                    )
                    record.update(status="error", error=type(exc).__name__, http_status=status)
                    failures += 1
                    if status in {401, 402, 403, 404}:
                        fatal = f"VseLLM HTTP {status}"
                charge = sum(map(cost, usage))
                spent += charge
                record.update(
                    usage=usage, estimated_rub=previous_cost + charge, labeled_at=utc_now()
                )
                save(path, record)
                if (done + failures) % 50 == 0 or fatal:
                    print(
                        f"{done}/{len(selections)} дней, ошибок {failures}, ~{spent:.2f} ₽",
                        flush=True,
                    )

        await asyncio.gather(*(worker() for _ in range(args.workers)))
    save(
        args.output / "label_run.json",
        {
            "model": MODEL,
            "corpus": input_fingerprints([args.corpus]),
            "start": args.start,
            "end": args.end,
            "requested_days": len(selections),
            "labeled_days": done,
            "failed_this_run": failures,
            "estimated_rub": spent,
            "fatal": fatal,
            "prompt": PROMPT,
            "retry_suffix": RETRY_SUFFIX,
            "final_retry_suffix": FINAL_SUFFIX,
            "max_output": MAX_OUTPUT,
            "max_titles": MAX_TITLES,
        },
    )
    print(f"{done}/{len(selections)} дней, расчётная стоимость {spent:.2f} ₽", flush=True)
    return 0 if done == len(selections) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/news/headlines.csv.gz"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--output", type=Path, default=Path("data/reports/llm_news_holdout"))
    parser.add_argument("--start", default="2016-10-01")
    parser.add_argument("--end", default="2026-11-18")
    parser.add_argument("--workers", type=int, default=24, choices=range(1, 41), metavar="N")
    parser.add_argument("--budget-rub", type=float, default=95)
    raise SystemExit(asyncio.run(label(parser.parse_args())))
