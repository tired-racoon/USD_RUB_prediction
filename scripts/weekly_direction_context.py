"""Сравнить полную панель и обучение временного энкодера для недельного знака."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from forecast_stacking import ARCHIVE_FILES, read_panel
from foundation_ensemble import CLASSICAL, RW
from scipy.optimize import minimize
from scipy.special import expit, logit
from sequence_holdout import BATCH_SIZE, CHECKPOINTS, LOOKBACK, SEED, SequenceRegressor
from si_options_holdout import point_forecast
from threadpoolctl import threadpool_limits
from torch import nn
from weekly_direction_models import pair_intervals, scores
from weekly_literature_holdout import HORIZON, gamma_frame, residual_features
from weekly_sign_size import direction

from usd_rub.ml_dataset import feature_panels, read_dataset
from usd_rub.sources.news_archive import write_csv
from usd_rub.storage import input_fingerprints

PRETRAIN_END = pd.Timestamp("2021-01-01")
PRETRAIN_HORIZONS = np.array([1, 3, 7])
PRETRAIN_EPOCHS = 12
ADAPTER_EPOCH = 8
ADAPTER_STRENGTHS = (0.25, 0.5, 1.0)
ARCHITECTURES = ("transformer", "cnn")
MODES = ("market", "fusion_scratch", "fusion_frozen", "fusion_tuned")
SETUPS = [(arch, mode) for arch in ARCHITECTURES for mode in MODES] + [
    ("transformer", "full_window"),
    ("mlp", "snapshot"),
]


def standardize(features, train):
    """Стандартизировать признаки по обучению и сохранить список доступных колонок."""
    past = features.loc[train]
    std = past.std(ddof=0)
    columns = std.index[std.gt(1e-12)]
    x = (features[columns] - past[columns].mean()) / std[columns]
    return torch.tensor(x.fillna(0).clip(-6, 6).to_numpy(), dtype=torch.float32), columns


def batches(rows, rng):
    """Разбить обучающие строки на пакеты одного порядка величины."""
    return np.array_split(rng.permutation(rows), (len(rows) - 1) // BATCH_SIZE + 1)


def pretrain(arch, history, origins, offsets, returns):
    """Обучить энкодер прогнозу доходности на более ранней истории."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = SequenceRegressor(arch, history.shape[-1], len(PRETRAIN_HORIZONS))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=0.02)
    for _ in range(PRETRAIN_EPOCHS):
        for batch in batches(np.arange(len(origins)), rng):
            optimizer.zero_grad(set_to_none=True)
            estimate = model(history[origins[batch, None] + offsets])
            loss = (estimate - returns[batch]).square().mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
    return model.state_dict()


class ContextClassifier(nn.Module):
    """Добавить признаки текущего среза к выходу временного энкодера."""

    def __init__(self, arch, mode, history_width, snapshot_width, state):
        super().__init__()
        self.mode = mode
        self.encoder = None
        self.snapshot = None
        head_width = 0
        if history_width:
            self.encoder = SequenceRegressor(arch, history_width, len(PRETRAIN_HORIZONS))
            head_width += self.encoder.head[0].in_features
            if state is not None:
                self.encoder.load_state_dict(state)
            self.encoder.head = nn.Identity()
            if mode == "fusion_frozen":
                self.encoder.requires_grad_(False)
            elif mode == "fusion_tuned":
                self.encoder.project.requires_grad_(False)
                if arch == "transformer":
                    self.encoder.position.requires_grad_(False)
        if snapshot_width:
            self.snapshot = nn.Sequential(
                nn.Dropout(0.15), nn.Linear(snapshot_width, 24), nn.LayerNorm(24), nn.GELU()
            )
            head_width += 24
        self.head = nn.Sequential(
            nn.Linear(head_width, 24), nn.GELU(), nn.Dropout(0.15), nn.Linear(24, 1)
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, history, snapshot):
        parts = []
        if self.encoder is not None:
            parts.append(self.encoder(history))
        if self.snapshot is not None:
            parts.append(self.snapshot(snapshot))
        return self.head(torch.cat(parts, dim=1)).squeeze(-1)


@dataclass
class Inputs:
    """Тензоры истории и текущего среза на дневном индексе."""

    history: torch.Tensor | None
    snapshot: torch.Tensor | None
    offsets: np.ndarray

    def batch(self, rows):
        history = None if self.history is None else self.history[rows[:, None] + self.offsets]
        snapshot = None if self.snapshot is None else self.snapshot[rows]
        return history, snapshot


def train_classifier(arch, mode, x, state, train, labels, query, checkpoints):
    """Обучить голову и при необходимости обновить часть энкодера."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    history_width, snapshot_width = (
        tensor.shape[-1] if tensor is not None else 0 for tensor in (x.history, x.snapshot)
    )
    model = ContextClassifier(arch, mode, history_width, snapshot_width, state)
    groups = [
        {
            "params": list(model.head.parameters())
            + (list(model.snapshot.parameters()) if model.snapshot is not None else [])
        }
    ]
    if model.encoder is not None and mode != "fusion_frozen":
        groups.append(
            {
                "params": [p for p in model.encoder.parameters() if p.requires_grad],
                "lr": 0.00025 if mode == "fusion_tuned" else 0.0015,
            }
        )
    optimizer = torch.optim.AdamW(groups, lr=0.0015, weight_decay=0.02)
    loss_fn = nn.BCEWithLogitsLoss()
    trained = {}
    for epoch in range(1, max(checkpoints) + 1):
        model.train()
        if mode == "fusion_frozen":
            model.encoder.eval()
        for batch in batches(np.arange(len(train)), rng):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(*x.batch(train[batch])), labels[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        if epoch in checkpoints:
            model.eval()
            with torch.inference_mode():
                trained[epoch] = np.concatenate(
                    [
                        model(*x.batch(batch)).sigmoid().numpy()
                        for batch in np.array_split(query, (len(query) - 1) // BATCH_SIZE + 1)
                    ]
                )
    if state is not None:
        for name, parameter in model.encoder.named_parameters():
            if not parameter.requires_grad:
                torch.testing.assert_close(parameter, state[name], atol=0, rtol=0)
    return trained


def add_context(snapshot, columns, train, test, labels, base, strengths):
    """Обучить регуляризованную поправку к замороженному классификатору."""
    groups = pd.Series(columns.str.split("__", n=1).str[0], index=columns)
    scale = np.sqrt(groups.map(groups.value_counts()).to_numpy())
    x_train = snapshot[train].numpy() / scale
    x_test = snapshot[test].numpy() / scale
    prediction = logit(np.clip(base, 1e-6, 1 - 1e-6))
    offset, test_offset = np.split(prediction, [len(train)])
    y = labels.numpy()

    def objective(w):
        values = offset + x_train @ w
        cost = np.mean(np.logaddexp(0, values) - y * values) + 0.075 * np.dot(w, w)
        grad = x_train.T @ (expit(values) - y) / len(y) + 0.15 * w
        return cost, grad

    fitted = minimize(objective, np.zeros(len(columns)), method="L-BFGS-B", jac=True)
    if not fitted.success:
        raise RuntimeError(f"Не обучилась поправка: {fitted.message}")
    increment = x_test @ fitted.x
    return {weight: expit(test_offset + weight * increment) for weight in strengths}


def fit_fold(compact, full, history, calibration, query, setup_epochs, states, offsets):
    cutoff = query.index[0]
    moves = calibration.loc[
        (calibration.index + HORIZON < cutoff) & calibration.actual.ne(calibration["last"])
    ]
    train, test = (full.index.get_indexer(index) for index in (moves.index, query.index))
    y = torch.tensor(moves.actual.gt(moves["last"]).to_numpy(), dtype=torch.float32)
    snapshot, active_cols = standardize(full, moves.index)
    market, _ = standardize(compact, moves.index)
    prepared = {
        "market": Inputs(market, None, offsets),
        "full_window": Inputs(snapshot, None, offsets),
        "snapshot": Inputs(None, snapshot, offsets),
    }
    prepared.update(
        {mode: Inputs(history, snapshot, offsets) for mode in MODES if mode.startswith("fusion")}
    )
    predictions = {}
    for arch in ARCHITECTURES:
        market_epochs = setup_epochs.get((arch, "market"), [])
        strengths = setup_epochs.get((arch, "adapter"), [])
        if not market_epochs and not strengths:
            continue
        checkpoints = set(market_epochs) | ({ADAPTER_EPOCH} if strengths else set())
        origins = np.r_[train, test] if strengths else test
        results = train_classifier(
            arch, "market", prepared["market"], None, train, y, origins, checkpoints
        )
        for epoch in market_epochs:
            predictions[arch, "market", epoch] = pd.Series(
                results[epoch][-len(test) :], index=query.index
            )
        if strengths:
            for weight, values in add_context(
                snapshot, active_cols, train, test, y, results[ADAPTER_EPOCH], strengths
            ).items():
                predictions[arch, "adapter", weight] = pd.Series(values, index=query.index)
    for (arch, mode), epochs in setup_epochs.items():
        if mode in {"market", "adapter"}:
            continue
        state = states[arch] if mode in {"fusion_frozen", "fusion_tuned"} else None
        forecasts = train_classifier(arch, mode, prepared[mode], state, train, y, test, epochs)
        predictions.update(
            {(arch, mode, epoch): pd.Series(p, index=query.index) for epoch, p in forecasts.items()}
        )
    return predictions, active_cols


def run(args):
    dataset = read_dataset(args.dataset)
    if not dataset.index.equals(pd.date_range(dataset.index.min(), dataset.index.max())):
        raise ValueError("Нужен непрерывный дневной датасет")
    panel = feature_panels(dataset)["extended"]
    events = feature_panels(read_dataset(args.events))["extended"]
    manifest = args.archive / "archives.json"
    groups = json.loads(manifest.read_text())["groups"]
    archives = [args.archive / filename for filename in ARCHIVE_FILES]
    names = [CLASSICAL, RW, *sum(groups.values(), [])]
    calibration, test = [gamma_frame(read_panel(path, names)) for path in archives]
    for data in (calibration, test):
        np.testing.assert_allclose(data["last"], dataset.usd_rub.loc[data.index], rtol=1e-12)
        np.testing.assert_allclose(
            data.actual, dataset.usd_rub.loc[data.index + HORIZON], rtol=1e-12
        )
    if (
        calibration.index.min() < PRETRAIN_END
        or calibration.index.max() + HORIZON >= test.index.min()
    ):
        raise ValueError("Периоды обучения пересекаются")

    inputs = residual_features(panel)["spread_dynamics"]
    projected = pd.concat([calibration, test])
    trend = ((projected.ensemble - projected["last"]) / projected["last"]).rename("gamma_return")
    compact = trend.to_frame().join(inputs, how="right")
    full = panel.join(events, validate="one_to_one").join(trend)
    full["own__current_usd_rub"] = dataset.usd_rub
    history_dates = dataset.index[
        (dataset.index >= "2015-01-01") & (dataset.index + HORIZON < PRETRAIN_END)
    ]
    history, history_cols = standardize(inputs, history_dates)
    origins = dataset.index.get_indexer(history_dates)
    levels = dataset.usd_rub.to_numpy()
    returns = levels[origins[:, None] + PRETRAIN_HORIZONS] / levels[origins, None] - 1
    targets = torch.tensor((returns / returns.std(axis=0)).clip(-5, 5), dtype=torch.float32)
    offsets = np.arange(1 - LOOKBACK, 1)
    states = {arch: pretrain(arch, history, origins, offsets, targets) for arch in ARCHITECTURES}
    print(f"Предобучение завершено, {len(origins)} дат", flush=True)

    validation, counts = {}, {}
    dates = calibration.loc["2022":].index
    for year, fold in calibration.loc[dates].groupby(dates.year):
        forecasts, cols = fit_fold(
            compact,
            full,
            history,
            calibration,
            fold,
            {
                **dict.fromkeys(SETUPS, CHECKPOINTS),
                **{(arch, "adapter"): ADAPTER_STRENGTHS for arch in ARCHITECTURES},
            },
            states,
            offsets,
        )
        for key, p in forecasts.items():
            validation.setdefault(key, []).append(p)
        counts[int(year)] = len(cols)
        print(f"Готов фолд {year}, {len(cols)} признаков", flush=True)
    outcomes = (calibration.actual - calibration["last"]).loc[dates]
    ranking = scores(outcomes, {key: pd.concat(p) for key, p in validation.items()})
    selected = ranking.drop_duplicates(["model", "panel"])
    settings = {
        (r.model, r.panel): [float(r.setting) if r.panel == "adapter" else int(r.setting)]
        for r in selected.itertuples(index=False)
    }
    forecasts, full_columns = fit_fold(
        compact, full, history, calibration, test, settings, states, offsets
    )
    truth = test.actual - test["last"]
    baseline = point_forecast(inputs, calibration, test, test.index[0])
    old = pd.read_csv(args.previous, parse_dates=["origin"]).set_index("origin")
    pd.testing.assert_index_equal(old.index.rename(None), test.index.rename(None))
    np.testing.assert_allclose(old.direct, baseline, rtol=1e-12)
    past = calibration.loc[calibration.index + HORIZON < test.index[0]]
    logistic = direction({"market": compact}, past, test)["market"]
    np.testing.assert_allclose(old.logistic_market, logistic, rtol=1e-12)
    for name in ARCHITECTURES:
        forecasts[name, "previous", 0] = old[name]
    forecasts["logistic", "market", 0] = logistic
    test_scores = scores(truth, forecasts)
    active = truth.ne(0)
    direct = np.sign(baseline - test["last"])
    test_scores.loc[len(test_scores)] = [
        "direct",
        "existing",
        "0",
        active.sum(),
        direct[active].eq(np.sign(truth[active])).mean(),
        np.nan,
    ]
    probabilities = {f"{arch}_{mode}": p for (arch, mode, _), p in forecasts.items()}
    intervals = pair_intervals(truth, probabilities, baseline - test["last"], logistic.to_numpy())
    additions = []
    for arch in ARCHITECTURES:
        name = f"{arch}_adapter"
        comparisons = pair_intervals(
            truth,
            {name: probabilities[name]},
            baseline - test["last"],
            probabilities[f"{arch}_market"].to_numpy(),
            reference_name="market",
        )
        additions.append(comparisons.loc[comparisons.metric.ne("accuracy_vs_direct")])
    intervals = pd.concat([intervals, *additions], ignore_index=True)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, table in {
        "validation": ranking,
        "selected": selected,
        "test": test_scores,
        "paired": intervals,
    }.items():
        write_csv(table, args.output / f"{name}.csv")
    write_csv(
        pd.DataFrame(probabilities)
        .join(test[["actual", "last"]])
        .rename_axis("origin")
        .reset_index(),
        args.output / "predictions.csv.gz",
    )
    meta = {
        "pretrain_end_exclusive": str(PRETRAIN_END.date()),
        "pretrain_dates": len(origins),
        "pretrain_last_target": str(dataset.index[origins[-1] + PRETRAIN_HORIZONS[-1]].date()),
        "history_features": history_cols.tolist(),
        "total_snapshot_columns": full.shape[1],
        "validation_columns": counts,
        "test_columns": full_columns.tolist(),
        "inputs": input_fingerprints(
            [args.dataset, args.events, args.previous, manifest, *archives]
        ),
    }
    (args.output / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        selected.round(5).to_string(index=False),
        test_scores.round(5).to_string(index=False),
        sep="\n",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/datasets/usd_rub_2011_2026_full_si_options.csv.gz"),
    )
    parser.add_argument(
        "--events", type=Path, default=Path("datasets/llm_news_event_features_2016_2026.csv.gz")
    )
    parser.add_argument("--archive", type=Path, default=Path("data/reports/forecast_stacking"))
    parser.add_argument(
        "--previous",
        type=Path,
        default=Path("data/reports/weekly_direction_models/predictions.csv.gz"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/reports/weekly_direction_context")
    )
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        run(parser.parse_args())
