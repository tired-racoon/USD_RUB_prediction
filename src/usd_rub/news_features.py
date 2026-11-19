"""Тональность заголовков и дневные признаки к полуночи по Москве."""

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd

from usd_rub.sources.news_archive import PRESS, read_smartlab, write_csv
from usd_rub.sources.rbc_telegram import read_rbc
from usd_rub.storage import input_fingerprints

ENCODER = ("cointegrated/rubert-tiny2", "e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae")
SENTIMENT = ("seara/rubert-tiny2-russian-sentiment", "ad9be38671d4444945734cce7cdaadacc5002cc4")
LABELS = ["negative", "neutral", "positive"]
SCORES = [f"p_{label}" for label in LABELS]
MAX_LENGTH = 64
EMBEDDING_DIM = 312


def headline_key(text):
    """Ключ для точных совпадений после нормализации написания."""
    text = unicodedata.normalize("NFKC", text).casefold().replace("ё", "е")
    return " ".join(re.findall(r"\w+|[.,:+%₽$€-]", text.replace("−", "-"))).strip(" .")


class HeadlineModel:
    """Батчевый расчёт замороженной модели RuBERT-tiny2."""

    def __init__(self, checkpoint, kind, device):
        try:
            import torch
            from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ValueError("Для анализа новостей выполните uv sync --extra news") from exc

        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        if device == "cpu":
            torch.set_num_threads(min(4, torch.get_num_threads()))
        self.torch = torch
        self.kind = kind
        self.device = device
        self.tokenize = AutoTokenizer.from_pretrained(checkpoint[0], revision=checkpoint[1])
        dtype = torch.float32 if device == "cpu" else torch.float16
        if kind == "embedding":
            self.model = AutoModel.from_pretrained(
                checkpoint[0], revision=checkpoint[1], dtype=dtype, add_pooling_layer=False
            )
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint[0], revision=checkpoint[1], dtype=dtype
            )
            indices = {value.lower(): key for key, value in self.model.config.id2label.items()}
            self.label_indices = [indices[label] for label in LABELS]
        self.model.to(device).eval()

    def score(self, texts, batch_size, result):
        """Записать нормированные векторы или вероятности классов."""
        torch = self.torch
        lengths = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        with torch.inference_mode():
            for offset in range(0, len(lengths), batch_size):
                subset = lengths[offset : offset + batch_size]
                tokens = self.tokenize(
                    [texts[i] for i in subset],
                    padding=True,
                    pad_to_multiple_of=8,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt",
                ).to(self.device)
                output = self.model(**tokens)
                if self.kind == "embedding":
                    values = torch.nn.functional.normalize(
                        output.last_hidden_state[:, 0, :].float()
                    )
                else:
                    values = output.logits[:, self.label_indices].float().softmax(dim=1)
                result[subset] = values.cpu().numpy()


def summarize(articles, vectors=None, vector_columns=()):
    """Агрегаты по источнику с поправкой на межредакционные повторы."""
    values = articles[SCORES].to_numpy()
    weights = articles["weight"].to_numpy()
    classes = values[:, : len(LABELS)].argmax(axis=1)
    share = np.bincount(classes, weights=weights, minlength=len(LABELS)) / weights.sum()
    means = np.average(values, axis=0, weights=weights)
    metrics = {
        "articles": len(articles),
        "unique_story_weight": weights.sum(),
        **dict(zip([f"{label}_share" for label in LABELS], share, strict=True)),
        **dict(zip(SCORES, means, strict=True)),
    }
    if vectors is not None:
        centroid = np.average(vectors[articles.text_id.to_numpy()], axis=0, weights=weights)
        metrics.update(zip(vector_columns, centroid, strict=True))
    return metrics


def aggregate_news(
    articles, text_ids, vectors, probabilities, dates, sources=PRESS, prefix="press"
):
    """Баланс источников и сдвиг к дате прогноза."""
    articles = articles.reset_index(drop=True)
    vector_columns = [f"embedding_{i:03d}" for i in range(vectors.shape[1])]
    article_scores = probabilities[text_ids]
    scored = articles.join(pd.DataFrame(article_scores, columns=SCORES))
    scored["label"] = np.asarray(LABELS)[article_scores.argmax(axis=1)]
    sample = scored.assign(text_id=text_ids)
    sample["headline_key"] = sample.headline.map(headline_key)
    sample = sample.drop_duplicates(["date", "source", "headline_key"])
    panel = sample.loc[sample.source.isin(sources)].copy()
    panel["weight"] = 1 / panel.groupby(["date", "headline_key"]).source.transform("size")

    by_source = []
    for (day, source), part in sample.groupby(["date", "source"], sort=True):
        metrics = summarize(part.assign(weight=1.0))
        by_source.append({"date": day, "source": source, **metrics})
    source_daily = pd.DataFrame(
        by_source,
        columns=[
            "date",
            "source",
            "articles",
            "unique_story_weight",
            *[f"{label}_share" for label in LABELS],
            *SCORES,
        ],
    )

    rows = []
    share_columns = [f"{label}_share" for label in LABELS]
    for day, part in panel.groupby("date", sort=True):
        metrics = {
            "articles": len(part),
            "unique_headlines": part.headline_key.nunique(),
            "sources_present": part.source.nunique(),
        }
        if set(part.source) == set(sources):
            whole = pd.DataFrame(
                [summarize(outlet, vectors, vector_columns) for _, outlet in part.groupby("source")]
            )
            metrics.update(whole[share_columns + SCORES + vector_columns].mean().to_dict())
            for topic in ("economy", "politics"):
                topical = part.loc[part.topic.eq(topic)]
                if not topical.empty:
                    means = pd.DataFrame(
                        [summarize(outlet) for _, outlet in topical.groupby("source")]
                    )[share_columns + SCORES].mean()
                    metrics.update({f"{topic}_{name}": val for name, val in means.items()})
        rows.append({"date": day, **metrics})
    columns = [
        "articles",
        "unique_headlines",
        "sources_present",
        *share_columns,
        *SCORES,
        *[
            f"{topic}_{name}"
            for topic in ("economy", "politics")
            for name in share_columns + SCORES
        ],
        *vector_columns,
    ]
    daily = pd.DataFrame(rows, columns=["date", *columns]).set_index("date")
    daily = daily.reindex(dates)
    daily["sentiment_balance"] = daily.p_positive - daily.p_negative
    daily["positive_negative_ratio"] = daily.positive_share / daily.negative_share.replace(
        0, np.nan
    )
    daily["positive_negative_prob_ratio"] = daily.p_positive / daily.p_negative.replace(0, np.nan)
    daily["sentiment_balance_7d"] = daily.sentiment_balance.rolling(7).mean()
    daily.index = daily.index + pd.Timedelta(days=1)
    return daily.add_prefix(f"news__{prefix}_").rename_axis("date"), source_daily, scored


def model_predictions(texts, checkpoint, kind, dimension, cache_dir, device, batch_size):
    """Вычислить или прочитать предсказания для неизменного набора текстов."""
    digest = hashlib.sha256(
        json.dumps(
            [checkpoint, kind, device, MAX_LENGTH, batch_size, texts], ensure_ascii=False
        ).encode()
    ).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{kind}_{digest}.npy"
    if not cache.exists():
        model = HeadlineModel(checkpoint, kind, device)
        print(f"{kind}: {len(texts):,} заголовков, устройство {model.device}", flush=True)
        with NamedTemporaryFile(dir=cache_dir, delete=False) as file:
            temporary = Path(file.name)
        try:
            result = np.lib.format.open_memmap(
                temporary, mode="w+", dtype=np.float32, shape=(len(texts), dimension)
            )
            for offset in range(0, len(texts), 16_384):
                stop = min(len(texts), offset + 16_384)
                part = result[offset:stop]
                model.score(texts[offset:stop], batch_size, part)
                if not np.isfinite(part).all():
                    raise ValueError("Модель вернула неконечные значения")
                print(f"{kind}: {stop:,}/{len(texts):,}", flush=True)
            result.flush()
            del part, result
            temporary.replace(cache)
        finally:
            temporary.unlink(missing_ok=True)
    return np.load(cache, mmap_mode="r")


def score_corpus(corpus, dates, output, device, batch_size, sources=PRESS, prefix="press"):
    """Свести оценки корпуса к дневной панели с фиксированным набором источников."""
    texts = list(pd.unique(corpus.headline))
    ids = pd.Index(texts).get_indexer(corpus.headline)
    cache_dir = output / "model_cache"
    vectors = model_predictions(
        texts, ENCODER, "embedding", EMBEDDING_DIM, cache_dir, device, batch_size
    )
    probabilities = model_predictions(
        texts, SENTIMENT, "sentiment", len(LABELS), cache_dir, device, batch_size
    )
    return aggregate_news(corpus, ids, vectors, probabilities, dates, sources, prefix)


def build_news(args):
    """Записать дневной датасет, тональность заголовков и параметры расчёта."""
    input_dir = Path(args.input or Path(args.data_dir) / "news")
    output = Path(args.output or input_dir)
    headlines_path, coverage_path = input_dir / "headlines.csv.gz", input_dir / "coverage.csv"
    corpus = pd.read_csv(headlines_path, parse_dates=["date"], keep_default_na=False)
    coverage = pd.read_csv(coverage_path, parse_dates=["date"])
    if corpus.empty or coverage.empty or args.batch_size < 1:
        raise ValueError("Нужны архивные заголовки, покрытие и положительный размер пакета")
    dates = pd.date_range(coverage.date.min(), coverage.date.max())
    daily, by_source, scored = score_corpus(corpus, dates, output, args.device, args.batch_size)
    supplement = {}
    archives = [headlines_path, coverage_path, *args.smartlab_parquet]
    extra = {}
    if args.smartlab_parquet:
        extra["smartlab"] = read_smartlab(args.smartlab_parquet, dates)
    if args.rbc_ids:
        extra["rbc"], rbc_path = read_rbc(args.data_dir, args.rbc_ids, dates)
        archives.append(rbc_path)
    for source, articles in extra.items():
        extra_daily, extra_by_source, extra_scored = score_corpus(
            articles, dates, output, args.device, args.batch_size, (source,), source
        )
        write_csv(
            extra_daily.dropna(axis=1, how="all"), output / f"daily_{source}.csv.gz", index=True
        )
        by_source = pd.concat([by_source, extra_by_source], ignore_index=True)
        scored = pd.concat([scored, extra_scored], ignore_index=True)
        supplement[source] = {
            "headlines": len(articles),
            "first_date": articles.date.min().date().isoformat(),
            "last_date": articles.date.max().date().isoformat(),
            "scored_days": articles.date.nunique(),
        }
    write_csv(daily, output / "daily.csv.gz", index=True)
    write_csv(by_source, output / "daily_by_source.csv")
    write_csv(scored, output / "headline_sentiment.csv.gz")
    with (output / "daily.csv.gz").open("rb") as file:
        checksum = hashlib.file_digest(file, "sha256").hexdigest()
    manifest = {
        "inputs": input_fingerprints(archives),
        "output_sha256": checksum,
        "encoder": ENCODER,
        "sentiment": SENTIMENT,
        "max_tokens": MAX_LENGTH,
        "origin_time": "00:00 Europe/Moscow",
        "panel_sources": PRESS,
        "first_origin": daily.index.min().date().isoformat(),
        "last_origin": daily.index.max().date().isoformat(),
        "calendar_days": len(daily),
        "scored_days": int(daily["news__press_sentiment_balance"].notna().sum()),
        "headlines": len(corpus),
        "sources": corpus.groupby("source").size().to_dict(),
        "supplement": supplement,
        "rbc_ids": args.rbc_ids,
        "device": args.device,
    }
    (output / "run.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"{output}: {manifest['scored_days']:,}/{len(daily):,} дат с признаками", flush=True)
    return 0
