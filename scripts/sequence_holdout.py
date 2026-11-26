"""Сравнить модели временных рядов и отбор признаков на holdout USD/RUB."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from export_ml_dataset import write_dataset
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from usd_rub.backtest import score
from usd_rub.ml_dataset import EXTENDED_PREFIX, FEATURE_GROUPS, feature_panels, read_dataset
from usd_rub.ml_experiment import (
    actual_at,
    blend_weights,
    choose_shrink,
    interval_from_validation,
    prediction_rows,
    selection_loss,
    to_levels,
)
from usd_rub.sequence_features import LOOKBACK, TOP_K, TRAIN_WINDOW, prepare_fold
from usd_rub.sequence_features import normalized_targets as normalized_targets
from usd_rub.storage import input_fingerprints

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise SystemExit(
        "Запустите uv run --extra sequence python scripts/sequence_holdout.py"
    ) from exc

ARCHITECTURES = ("lstm", "gru", "cnn", "transformer")
CHECKPOINTS = (4, 8, 16)
BATCH_SIZE = 128
SEED = 191


class SequenceRegressor(nn.Module):
    """Небольшая модель поправки к случайному блужданию."""

    def __init__(self, architecture, features, horizons, context_features=0):
        super().__init__()
        hidden = 24
        self.architecture = architecture
        self.context_scale = (2 * hidden / context_features) ** 0.5 if context_features else 1
        self.project = nn.Sequential(nn.Linear(features, hidden), nn.LayerNorm(hidden), nn.GELU())
        if architecture == "lstm":
            self.temporal = nn.LSTM(hidden, hidden, batch_first=True)
        elif architecture == "gru":
            self.temporal = nn.GRU(hidden, hidden, batch_first=True)
        elif architecture == "cnn":
            self.temporal = nn.Sequential(
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(hidden, hidden, kernel_size=3, dilation=2, padding=2),
                nn.GELU(),
            )
        elif architecture == "transformer":
            self.position = nn.Parameter(torch.zeros(1, LOOKBACK, hidden))
            nn.init.normal_(self.position, std=0.02)
            self.temporal = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=3, dim_feedforward=48, dropout=0.1, batch_first=True
            )
        else:
            raise ValueError(f"Неизвестная архитектура {architecture}")
        self.head = nn.Sequential(
            nn.Linear(2 * hidden + context_features, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, horizons),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, sequences, context=None):
        steps = self.project(sequences)
        if self.architecture in {"gru", "lstm"}:
            encoded = self.temporal(steps)[0][:, -1]
        elif self.architecture == "cnn":
            encoded = self.temporal(steps.transpose(1, 2)).mean(dim=2)
        else:
            encoded = self.temporal(steps + self.position)[:, -1]
        features = (steps[:, -1], encoded)
        if context is not None:
            features = (*features, context * self.context_scale)
        return self.head(torch.cat(features, dim=1))


def train_sequence(
    architecture, width, x_train, y_train, weights, x_query, epochs, device, *, context=None
):
    """Один прогон обучения с прогнозами на заранее заданных эпохах."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    context_features = context[0].shape[1] if context is not None else 0
    train_inputs = (x_train, context[0]) if context is not None else (x_train,)
    query_inputs = (x_query, context[1]) if context is not None else (x_query,)
    model = SequenceRegressor(architecture, width, y_train.shape[1], context_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=0.02)
    predictions = {}
    for epoch in range(1, max(epochs) + 1):
        model.train()
        for batch in np.array_split(
            rng.permutation(len(x_train)), (len(x_train) - 1) // BATCH_SIZE + 1
        ):
            optimizer.zero_grad(set_to_none=True)
            residual = model(*(value[batch] for value in train_inputs)) - y_train[batch]
            loss = (weights[batch, None] * residual.square()).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        if epoch in epochs:
            model.eval()
            with torch.inference_mode():
                batches = zip(*(value.split(BATCH_SIZE) for value in query_inputs), strict=True)
                query = [model(*batch).cpu().numpy() for batch in batches]
            predictions[epoch] = np.concatenate(query, axis=0)
    return predictions


def fit_fold(panel, levels, cutoff, origins, horizons, min_hit_rate, device, schedule):
    """Обучить сравнимые модели на полной и отобранной панелях."""
    train, values, targets, weights, scale, order, features = prepare_fold(
        panel, levels, cutoff, horizons, min_hit_rate
    )
    offsets = np.arange(1 - LOOKBACK, 1)
    results = {}
    for budget, indices in (("all", np.arange(values.shape[1])), ("top64", order[:TOP_K])):
        x = values[:, indices]
        if "ridge" in schedule[budget]:
            ridge = Ridge(alpha=0.15 * len(train), fit_intercept=False)
            ridge.fit(x[train], targets, sample_weight=weights)
            results[("ridge", budget, 0)] = ridge.predict(x[origins]) * scale

        def tensor(value):
            return torch.as_tensor(value, dtype=torch.float32, device=device)

        x_train, x_query = (
            tensor(x[train[:, None] + offsets]),
            tensor(x[origins[:, None] + offsets]),
        )
        y_train, sample_weight = tensor(targets), tensor(weights)
        for name, epochs in schedule[budget].items():
            if name == "ridge":
                continue
            predictions = train_sequence(
                name, len(indices), x_train, y_train, sample_weight, x_query, epochs, device
            )
            for epoch, prediction in predictions.items():
                results[(name, budget, epoch)] = prediction * scale
            print(f"{panel.index[cutoff].date()} {budget} ({len(indices)}) {name}", flush=True)
    return results, features.assign(selected_top64=features["rank"].le(TOP_K)), len(train)


def previous_forecasts(path, reference, truth, dataset_path):
    """Сверить предыдущий ансамбль с данным снимком и теми же целями."""
    run = json.loads((path.parent / "run.json").read_text())
    fingerprint = input_fingerprints([dataset_path])[0]["sha256"]
    if not any(item["sha256"] == fingerprint for item in run["inputs"]):
        raise ValueError("Предыдущий ансамбль рассчитан на другом снимке")
    old = pd.read_csv(path, parse_dates=["origin", "target_date"])
    old = old.loc[old.model.eq("ensemble")].sort_values(["origin", "horizon"])
    keys = ["origin", "horizon", "target_date"]
    pd.testing.assert_frame_equal(
        old[keys].reset_index(drop=True), truth[keys].reset_index(drop=True)
    )
    np.testing.assert_allclose(old.actual, truth.actual, rtol=1e-12)
    np.testing.assert_allclose(old["last"], truth["last"], rtol=1e-12)
    if not np.isfinite(old.prediction).all():
        raise ValueError("Предыдущий ансамбль содержит неконечные прогнозы")
    return old.assign(model=reference)


def label(key):
    """Имя модели и выбранной эпохи в отчётах."""
    architecture, budget, epoch = key
    suffix = f"_e{epoch:02d}" if epoch else ""
    return f"{architecture}_{budget}{suffix}"


def compare(args):
    """Валидация, фиксация выбора и финальная проверка на общих датах."""
    config = json.loads(args.config.read_text())
    dataset = read_dataset(args.dataset)
    dates = pd.date_range(config["start"], config["end"])
    split = dates.searchsorted(config["test_start"])
    if not dataset.index.equals(dates) or not np.array_equal(
        dataset.split, np.where(dates < dates[split], "train", "test")
    ):
        raise ValueError("Даты и разбиение снимка расходятся с конфигурацией")
    target = dataset.usd_rub
    levels = target.to_numpy(dtype=float)
    if not (np.isfinite(levels).all() and (levels > 0).all()):
        raise ValueError("Ожидается конечный положительный курс")
    panel = feature_panels(dataset)["extended"]
    horizons = np.arange(1, config["horizon"] + 1)
    min_hit_rate = config.get("min_hit_rate", 0.5)
    paths = [
        args.dataset,
        args.config,
        args.previous,
        args.previous.parent / "run.json",
        Path(__file__),
    ]
    fingerprints = input_fingerprints(paths)
    full_schedule = {
        budget: {"ridge": (0,), **dict.fromkeys(ARCHITECTURES, CHECKPOINTS)}
        for budget in ("all", "top64")
    }
    splits, reports, feature_ranks = [], [], []
    args.output.mkdir(parents=True, exist_ok=True)
    for year in config["validation_years"]:
        cutoff = dates.searchsorted(f"{year}-01-01")
        end = dates.searchsorted(f"{year + 1}-01-01")
        if end > split:
            raise ValueError("Валидация пересекает holdout")
        valid = np.arange(cutoff, end - horizons[-1])
        report, rank, _ = fit_fold(
            panel, levels, cutoff, valid, horizons, min_hit_rate, args.device, full_schedule
        )
        splits.append((valid, actual_at(levels, valid, horizons), levels[valid]))
        reports.append(report)
        feature_ranks.append(rank.assign(fold=str(year)))

    candidates, validation = [], {}
    for key in reports[0]:
        raw = [report[key] for report in reports]
        if not all(np.isfinite(fold).all() for fold in raw):
            raise ValueError(f"Неконечный прогноз {label(key)}")
        cv, shrink = choose_shrink(raw, splits)
        validation[label(key)] = [
            to_levels(z, s[2], shrink) for z, s in zip(raw, splits, strict=True)
        ]
        candidates.append(
            {
                "model": label(key),
                "architecture": key[0],
                "features": key[1],
                "epoch": key[2],
                "cv_rmse_to_rw": cv,
                "shrink": shrink,
                "cv_rmse_to_rw_raw": selection_loss(
                    [to_levels(z, s[2]) for z, s in zip(raw, splits, strict=True)], splits
                ),
            }
        )
    ranking = pd.DataFrame(candidates).sort_values(["cv_rmse_to_rw", "epoch", "model"])
    ranking.to_csv(args.output / "validation_candidates.csv", index=False)
    selected = ranking.drop_duplicates(["architecture", "features"])
    best_neural = selected.loc[selected.architecture.isin(ARCHITECTURES), "model"].iloc[0]
    families = selected.drop_duplicates("architecture").model.tolist()
    validation["rw"] = [np.repeat(last[:, None], len(horizons), axis=1) for _, _, last in splits]
    components = ["rw", *families]
    weights = blend_weights([validation[name] for name in components], splits)
    validation["new_ensemble"] = [
        sum(w * z for w, z in zip(weights, fold, strict=True))
        for fold in zip(*(validation[name] for name in components), strict=True)
    ]
    ensemble_cv = selection_loss(validation["new_ensemble"], splits)
    decisions = {
        "best_neural": best_neural,
        "best_individual": selected.iloc[0].model,
        "ensemble_cv_rmse_to_rw": ensemble_cv,
        "ensemble_weights": dict(zip(components, weights.tolist(), strict=True)),
        "selected": selected.to_dict("records"),
    }
    (args.output / "selection.json").write_text(json.dumps(decisions, indent=2))

    schedule = {budget: {} for budget in full_schedule}
    for row in selected.itertuples(index=False):
        schedule[row.features][row.architecture] = (row.epoch,)
    origins = np.arange(split, len(dates) - horizons[-1])
    test, test_rank, train_n = fit_fold(
        panel, levels, split, origins, horizons, min_hit_rate, args.device, schedule
    )
    feature_ranks.append(test_rank.assign(fold="final"))
    pd.concat(feature_ranks, ignore_index=True).to_csv(
        args.output / "features_ranked.csv", index=False
    )
    kept = set(test_rank.loc[test_rank.selected_top64, "feature"])
    reduced = [
        column
        for column in dataset
        if (name := column.removeprefix(EXTENDED_PREFIX)) in kept
        or name.split("__", 1)[0] not in FEATURE_GROUPS
    ]
    write_dataset(dataset[reduced], args.output / "dataset_top64.csv.gz")

    last = levels[origins]
    forecasts = {"rw": np.repeat(last[:, None], len(horizons), axis=1)}
    for key, raw in test.items():
        name = label(key)
        shrink = selected.loc[selected.model.eq(name), "shrink"].iat[0]
        forecasts[name] = to_levels(raw, last, shrink)
    forecasts["new_ensemble"] = sum(
        w * forecasts[name] for w, name in zip(weights, components, strict=True)
    )
    mase_scale = np.abs(np.diff(levels[split - config["baseline_window"] : split])).mean()
    records = []
    for name, forecast in forecasts.items():
        lower, upper = interval_from_validation(forecast, last, validation[name], splits)
        records.append(
            prediction_rows(
                name, target, origins, horizons, forecast, lower, upper, mase_scale, train_n
            )
        )
    records.append(previous_forecasts(args.previous, "previous_ensemble", records[0], args.dataset))
    predictions = pd.concat(records, ignore_index=True)
    metrics = score(predictions)
    predictions.to_csv(args.output / "predictions.csv", index=False)
    metrics.to_csv(args.output / "metrics.csv", index=False)
    summary = (
        metrics.assign(rmse_to_rw=1 - metrics.rmse_skill_vs_rw)
        .groupby("model")[["rmse_to_rw", "mape_pct", "smape_pct", "rmspe_pct"]]
        .mean()
    )
    for horizon in (1, horizons[-1]):
        summary[f"rmse_{horizon}d"] = (
            metrics.loc[metrics.horizon.eq(horizon)].set_index("model").rmse
        )
    summary["cv_rmse_to_rw"] = selected.set_index("model").cv_rmse_to_rw
    summary.loc["new_ensemble", "cv_rmse_to_rw"] = ensemble_cv
    summary.sort_values("rmse_to_rw").to_csv(args.output / "summary.csv")
    if fingerprints != input_fingerprints(paths):
        raise ValueError("Входные файлы менялись во время эксперимента")
    metadata = {
        **decisions,
        "inputs": fingerprints,
        "torch": torch.__version__,
        "device": args.device,
        "lookback": LOOKBACK,
        "train_window": TRAIN_WINDOW,
        "seed": SEED,
        "top_features": TOP_K,
        "test_origins": len(origins),
        "first_origin": str(dates[origins[0]].date()),
        "last_origin": str(dates[origins[-1]].date()),
    }
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2))
    print(summary.sort_values("rmse_to_rw").to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/ml_holdout.json"))
    parser.add_argument(
        "--dataset", type=Path, default=Path("datasets/usd_rub_2011_2026_full.csv.gz")
    )
    parser.add_argument(
        "--previous",
        type=Path,
        default=Path("data/reports/ml_holdout_minfin_rosstat/with_file/predictions.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/reports/sequence_holdout"))
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        compare(args)
