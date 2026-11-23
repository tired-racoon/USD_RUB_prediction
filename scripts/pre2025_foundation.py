"""Проверить предобученные модели с опубликованными до 2025 года весами."""

import argparse
import json
from hashlib import file_digest
from pathlib import Path
from time import perf_counter

import httpx
import numpy as np
import pandas as pd
from forecast_extensions import load_inputs
from forecast_stacking import panel_index

from usd_rub.fixing_calendar import HORIZONS, fixing_mask
from usd_rub.ml_dataset import feature_panels
from usd_rub.sources.news_archive import write_csv
from usd_rub.storage import input_fingerprints
from usd_rub.t0_gamma import FEATURES

CONTEXT = 512
HORIZON = len(HORIZONS)
PINS = {
    "chronos_bolt_tiny_2024": (
        "amazon/chronos-bolt-tiny",
        "f6ff2d2ba9168d498c015bc8dd07e3b395b31b3f",
        "2024-11-28",
        "75068728d376d2bec670379eeef4bfb4d24c0cfe24d957451f8d19b447030a32",
    ),
    "ttm_r1_2024": (
        "ibm-granite/granite-timeseries-ttm-r1",
        "2ac89ed42f10643b61219a2c4d39c7145c171ddb",
        "2024-12-18",
        "30f8d9be619518e18c4c15c6ee504e491b9624b48e251661cd5f967bce097731",
    ),
    "moirai_small_2024": (
        "Salesforce/moirai-1.0-R-small",
        "79ae2f2a22a2a0424c53d7c50ed90afc960c2324",
        "2024-08-22",
        "eaafeb166a3dfcf5728ab52a9ccc5392a23333b42a41a14e5528089ece3ef377",
    ),
}


def checkpoint(args):
    """Скачать зафиксированную ревизию и проверить хеш весов."""
    repo, revision, date, expected = PINS[args.model]
    root = args.checkpoints / args.model
    root.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "model.safetensors"):
        file = root / name
        if not file.is_file():
            temporary = file.with_suffix(file.suffix + ".partial")
            try:
                url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
                with httpx.stream("GET", url, follow_redirects=True, timeout=90) as response:
                    response.raise_for_status()
                    with temporary.open("wb") as stream:
                        for chunk in response.iter_bytes():
                            stream.write(chunk)
                temporary.replace(file)
            finally:
                temporary.unlink(missing_ok=True)
    with (root / "model.safetensors").open("rb") as file:
        if file_digest(file, "sha256").hexdigest() != expected:
            raise ValueError(f"Контрольная сумма весов отличается для {args.model}")
    return root, {
        "repository": repo,
        "revision": revision,
        "revision_date": date,
        "sha256": expected,
    }


def load_predictor(model, local):
    """Вернуть точечный прогноз для зафиксированной модели."""
    import torch

    if model == "chronos_bolt_tiny_2024":
        from chronos import ChronosBoltPipeline

        pipeline = ChronosBoltPipeline.from_pretrained(
            str(local), device_map="cpu", local_files_only=True
        )

        def chronos(levels, past, known):
            quantiles, _ = pipeline.predict_quantiles(
                torch.from_numpy(levels),
                prediction_length=HORIZON,
                quantile_levels=[0.5],
            )
            return quantiles[:, :, 0].numpy()

        return chronos
    if model == "ttm_r1_2024":
        from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

        ttm = TinyTimeMixerForPrediction.from_pretrained(str(local), local_files_only=True).eval()

        def predict_ttm(levels, past, known):
            prediction = ttm(past_values=torch.from_numpy(levels[:, :, None])).prediction_outputs
            return prediction[:, :HORIZON, 0].numpy()

        return predict_ttm

    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

    module = MoiraiModule.from_pretrained(str(local), local_files_only=True).eval()
    variants = {
        has_market: MoiraiForecast(
            module=module,
            prediction_length=HORIZON,
            context_length=CONTEXT,
            patch_size=32,
            num_samples=128,
            target_dim=1,
            feat_dynamic_real_dim=int(has_market),
            past_feat_dynamic_real_dim=len(FEATURES) * int(has_market),
        ).eval()
        for has_market in (False, True)
    }

    def moirai(levels, past, known):
        x = torch.from_numpy(levels[:, :, None])
        args = {}
        if past is not None:
            args = {
                "feat_dynamic_real": torch.from_numpy(known[:, :, None]),
                "observed_feat_dynamic_real": torch.ones((*known.shape, 1), dtype=torch.bool),
                "past_feat_dynamic_real": torch.from_numpy(np.nan_to_num(past)),
                "past_observed_feat_dynamic_real": torch.from_numpy(np.isfinite(past)),
            }
        samples = variants[past is not None](
            x,
            torch.ones_like(x, dtype=torch.bool),
            torch.zeros(x.shape[:2], dtype=torch.bool),
            **args,
        )
        return samples.median(dim=1).values.reshape(len(x), HORIZON).numpy()

    return moirai


def run(args):
    """Сохранить сырые прогнозы для тех же дат, что у классического ансамбля."""
    import torch

    torch.set_num_threads(args.threads)
    torch.manual_seed(191)
    dataset, _, test, _, paths = load_inputs(args)
    dates = test.dates
    if args.limit:
        dates = dates[: args.limit]
    local, pin = checkpoint(args)
    if dates.min() <= pd.Timestamp(pin["revision_date"]):
        raise ValueError("Дата прогноза должна быть после публикации весов")
    predictor = load_predictor(args.model, local)
    calendar = json.loads(args.calendar.read_text())
    positions = dataset.index.get_indexer(dates)
    context = positions[:, None] - np.arange(CONTEXT - 1, -1, -1)
    if context.min() < 0:
        raise ValueError("Недостаточно дневной истории для предобученной модели")
    levels = dataset.usd_rub.to_numpy(dtype=np.float32)[context]
    features = feature_panels(dataset)["extended"]
    past = features[list(FEATURES)].to_numpy(dtype=np.float32)[context]
    was_fixing = fixing_mask(dataset.index - pd.Timedelta(days=1), calendar, True)[:, 0]
    will_fix = fixing_mask(dates, calendar, True)
    known = np.concatenate((was_fixing[context], will_fix), axis=1).astype(np.float32)
    columns = {}
    modes = ("usd_only", "market_calendar") if args.model == "moirai_small_2024" else ("usd_only",)
    with torch.inference_mode():
        for mode in modes:
            values = []
            start_time = perf_counter()
            for first in range(0, len(dates), args.batch_size):
                batch = slice(first, first + args.batch_size)
                side = (None, None) if mode == "usd_only" else (past[batch], known[batch])
                point = np.asarray(predictor(levels[batch], *side), dtype=float)
                if point.shape != (len(levels[batch]), HORIZON) or not np.isfinite(point).all():
                    raise ValueError(f"Некорректная форма или значения {args.model}/{mode}")
                values.append(point)
                if first == 0 or first % (10 * args.batch_size) == 0:
                    print(args.model, mode, first + len(point), len(dates), flush=True)
            print(args.model, mode, round(perf_counter() - start_time, 2), "sec", flush=True)
            columns[f"{args.model}/{mode}"] = np.vstack(values).ravel()
    if args.limit:
        return
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"{args.model}.csv.gz"
    write_csv(pd.DataFrame(columns, index=panel_index(dates)).reset_index(), path)
    metadata = {
        **pin,
        "context": CONTEXT,
        "torch": torch.__version__,
        "batch_size": args.batch_size,
        "inputs": input_fingerprints([*paths, args.calendar, local / "config.json"]),
        "predictions": input_fingerprints([path])[0],
    }
    (args.output / f"{args.model}.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=PINS)
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/reports/theory_holdout/dataset.csv.gz")
    )
    parser.add_argument("--config", type=Path, default=Path("config/ml_holdout.json"))
    parser.add_argument("--calendar", type=Path, default=Path("config/cbr_fixing_calendar.json"))
    parser.add_argument("--archive", type=Path, default=Path("data/reports/forecast_stacking"))
    parser.add_argument("--checkpoints", type=Path, default=Path("data/model_runtime/checkpoints"))
    parser.add_argument("--output", type=Path, default=Path("data/reports/pre2025_foundation"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--limit", type=int)
    run(parser.parse_args())
