"""Заморозка нейросети и новые входы для прямого прогноза USD/RUB."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from correlation_holdout import HORIZONS
from forecast_extensions import load_inputs
from foundation_ensemble import CLASSICAL
from late_fusion_holdout import standardize
from sequence_holdout import (
    BATCH_SIZE,
    LOOKBACK,
    SEED,
    TRAIN_WINDOW,
    SequenceRegressor,
    normalized_targets,
)
from si_options_holdout import point_forecast
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from torch import nn
from weekly_direction_context import batches
from weekly_literature_holdout import (
    HORIZON,
    annual_score,
    gamma_frame,
    residual_features,
    summarize,
)

from usd_rub.ml_dataset import feature_panels, read_dataset
from usd_rub.ml_experiment import eligible_rows, to_levels
from usd_rub.sources.news_archive import write_csv
from usd_rub.storage import input_fingerprints

ARCHITECTURES = ("cnn", "transformer")
CONTEXT_MODES = ("from_scratch", "frozen", "tuned_head", "tuned_temporal")
BASE_EPOCHS = 4
CONTEXT_EPOCHS = (2, 4, 8)
SHRINK = (0, 0.25, 0.5, 0.75, 1.0)
BLEND_WEIGHTS = (0, 0.05, 0.15, 0.3)
ADAPTER_PENALTIES = (0.02, 0.15, 1.0)
ADAPTER_WEIGHTS = (0, 0.25, 0.5, 1.0)


class ExpandedLinear(nn.Module):
    """Разделить старые и новые коэффициенты первого линейного слоя головы."""

    def __init__(self, existing, context_width):
        super().__init__()
        self.existing = existing
        self.additional = nn.Linear(context_width, existing.out_features, bias=False)
        nn.init.zeros_(self.additional.weight)

    def forward(self, features):
        width = self.existing.in_features
        return self.existing(features[:, :width]) + self.additional(features[:, width:])


def train_network(
    model,
    optimizer,
    history,
    context,
    train,
    query,
    target,
    weights,
    offsets,
    checkpoints,
    *,
    frozen=False,
):
    """Обучить прямую регрессию на семи горизонтах."""
    rng = np.random.default_rng(SEED)
    forecasts = {}

    def predict(rows):
        window = history[rows[:, None] + offsets]
        return model(window) if context is None else model(window, context[rows])

    for epoch in range(1, max(checkpoints) + 1):
        model.train(not frozen)
        for batch in batches(np.arange(len(train)), rng):
            optimizer.zero_grad(set_to_none=True)
            error = predict(train[batch]) - target[batch]
            loss = (weights[batch, None] * error.square()).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        if epoch in checkpoints:
            model.eval()
            with torch.inference_mode():
                parts = np.array_split(query, (len(query) - 1) // BATCH_SIZE + 1)
                forecasts[epoch] = np.concatenate([predict(part).numpy() for part in parts])
    return forecasts


def context_branch(model, width, mode):
    """Добавить нулевые входы и настроить, какие коэффициенты обучаются."""
    branch = deepcopy(model).requires_grad_(False)
    branch.context_scale = 1.0
    branch.head[0] = ExpandedLinear(branch.head[0], width)
    groups = [{"params": branch.head[0].additional.parameters()}]
    if mode != "frozen":
        previous = [*branch.head[0].existing.parameters(), *branch.head[3].parameters()]
        if mode == "tuned_temporal":
            previous.extend(branch.temporal.parameters())
        for parameter in previous:
            parameter.requires_grad_(True)
        groups.append({"params": previous, "lr": 0.00025})
    return branch, torch.optim.AdamW(groups, lr=0.0015, weight_decay=0.02)


def equal_state(before, after):
    """Проверить неизменность замороженного модуля."""
    for name, weight in before.state_dict().items():
        torch.testing.assert_close(weight, after.state_dict()[name], rtol=0, atol=0)


def fold_predictions(full, market, levels, archive, schedules):
    dates = full.index
    query = dates.get_indexer(archive.dates)
    cutoff = query[0]
    train = eligible_rows(cutoff, int(HORIZONS[-1]), TRAIN_WINDOW)
    train = train[train >= LOOKBACK - 1]
    x, _, variable = standardize(full.to_numpy(dtype=float), train)
    tabular = ~market
    groups = full.columns[tabular].str.split("__", n=1).str[0]
    counts = pd.Series(variable[tabular], index=groups).groupby(level=0).sum()
    scales = np.sqrt(counts.reindex(groups).clip(lower=1).to_numpy())
    context = torch.as_tensor(x[:, tabular] / scales, dtype=torch.float32)
    history = torch.as_tensor(x[:, market])
    offsets = np.arange(1 - LOOKBACK, 1)
    target, weights, scale = normalized_targets(levels, train, HORIZONS)
    target, weights = (torch.as_tensor(v, dtype=torch.float32) for v in (target, weights))
    forecasts = {}
    for arch in ARCHITECTURES:
        torch.manual_seed(SEED)
        backbone = SequenceRegressor(arch, history.shape[1], len(HORIZONS))
        optimizer = torch.optim.AdamW(backbone.parameters(), lr=0.0015, weight_decay=0.02)
        shared = (train, query, target, weights, offsets)
        raw = train_network(backbone, optimizer, history, None, *shared, (BASE_EPOCHS,))
        forecasts[arch, "market", BASE_EPOCHS] = raw[BASE_EPOCHS]
        for mode in CONTEXT_MODES:
            epochs = schedules.get((arch, mode), ())
            if not epochs:
                continue
            torch.manual_seed(SEED)
            if mode == "from_scratch":
                network = SequenceRegressor(arch, history.shape[1], len(HORIZONS), context.shape[1])
                network.context_scale = 1.0
                optimizer = torch.optim.AdamW(network.parameters(), lr=0.0015, weight_decay=0.02)
            else:
                network, optimizer = context_branch(backbone, context.shape[1], mode)
                network.eval()
                with torch.inference_mode():
                    first = query[:16]
                    window = history[first[:, None] + offsets]
                    torch.testing.assert_close(
                        network(window, context[first]), backbone(window), atol=0, rtol=0
                    )
            seen = train_network(
                network, optimizer, history, context, *shared, epochs, frozen=mode == "frozen"
            )
            if mode != "from_scratch":
                equal_state(backbone.project, network.project)
                if mode != "tuned_temporal":
                    equal_state(backbone.temporal, network.temporal)
                if arch == "transformer":
                    torch.testing.assert_close(backbone.position, network.position, atol=0, rtol=0)
                if mode == "frozen":
                    equal_state(backbone.head[0], network.head[0].existing)
                    equal_state(backbone.head[3], network.head[3])
            forecasts.update({(arch, mode, epoch): value for epoch, value in seen.items()})
    projected = {
        key: (archive.project(to_levels(value * scale, levels[query]))[:, -1] / levels[query] - 1)
        for key, value in forecasts.items()
    }
    return {
        key: pd.Series(value, index=archive.dates) for key, value in projected.items()
    }, variable


def adapter(full, compact, calibration, query, cutoff):
    """Обучить дополнительную поправку к текущему недельному гибриду."""
    history = calibration.loc[(calibration.index + HORIZON < cutoff) & calibration.new_fixing]
    in_sample = point_forecast(compact, calibration, history, cutoff)
    kept = full.loc[history.index].std(ddof=0).gt(1e-12)
    cols = full.columns[kept]
    groups = cols.str.split("__", n=1).str[0]
    balance = np.sqrt(groups.map(groups.value_counts()).to_numpy())
    data = full[cols]
    moments = data.loc[history.index]
    x = ((data - moments.mean()) / moments.std(ddof=0)).fillna(0).clip(-6, 6) / balance
    y = (history.actual - in_sample) / history["last"]
    weights = (history["last"] / history["last"].median()) ** 2
    base = point_forecast(compact, calibration, query, cutoff)
    forecasts = {}
    for penalty in ADAPTER_PENALTIES:
        model = Ridge(alpha=penalty * len(history), fit_intercept=False)
        model.fit(x.loc[history.index], y, sample_weight=weights)
        delta = model.predict(x.loc[query.index]) * query["last"].where(query.new_fixing, 0)
        forecasts.update({(penalty, weight): base + weight * delta for weight in ADAPTER_WEIGHTS})
    return base, forecasts


def run(args):
    dataset, calibration, test, _, paths = load_inputs(args)
    panel = feature_panels(dataset)["extended"]
    full = panel.join(feature_panels(read_dataset(args.events))["extended"], validate="one_to_one")
    groups = full.columns.str.split("__", n=1).str[0]
    market = groups.isin(("own", "fx", "markets", "global")) & ~full.columns.str.endswith("_age")
    levels = dataset.usd_rub.to_numpy(dtype=float)
    schedules = {(arch, mode): CONTEXT_EPOCHS for arch in ARCHITECTURES for mode in CONTEXT_MODES}
    oof, counts = {}, {}
    valid_adapter, valid_direct = {}, []
    ref = gamma_frame(calibration)
    query = gamma_frame(test)
    compact = residual_features(panel)["spread_dynamics"]

    for year in calibration.dates.year.unique():
        select = calibration.dates.year == year
        fold = type(calibration)(
            calibration.dates[select],
            calibration.actual[select],
            calibration.last[select],
            calibration.fixes[select],
            {},
            {},
        )
        predictions, variable = fold_predictions(full, market, levels, fold, schedules)
        for key, value in predictions.items():
            oof.setdefault(key, []).append(value)
        counts[int(year)] = int(variable.sum())
        if year >= 2022:
            base, corrections = adapter(full, compact, ref, ref.loc[fold.dates], fold.dates[0])
            valid_direct.append(base)
            for key, value in corrections.items():
                valid_adapter.setdefault(key, []).append(value)
        print(f"Готов фолд {year}, {variable.sum()} признаков", flush=True)

    grid = []
    validation = {}
    for (arch, mode, epoch), parts in oof.items():
        returns = pd.concat(parts)
        for shrink in SHRINK:
            key = (arch, mode, epoch, shrink)
            provisional = ref["last"] * (1 + shrink * returns)
            validation[key] = provisional
            grid.append(
                {
                    "arch": arch,
                    "mode": mode,
                    "epoch": epoch,
                    "shrink": shrink,
                    "rmse_to_rw": annual_score(provisional, ref),
                }
            )
    ranking = pd.DataFrame(grid).sort_values(["rmse_to_rw", "epoch", "shrink"])
    selected = ranking.drop_duplicates(["arch", "mode"])
    final_schedule = {(row.arch, row.mode): [row.epoch] for row in selected.itertuples(index=False)}
    estimated, active = fold_predictions(full, market, levels, test, final_schedule)
    direct, corrected = adapter(full, compact, ref, query, query.index[0])
    final = {
        "direct_hybrid": direct,
        "classical": pd.Series(test.projected[CLASSICAL][:, -1], index=query.index),
    }
    valid_base = pd.concat(valid_direct)
    meta_ref = ref.loc[valid_base.index]
    selection = []
    for row in selected.itertuples(index=False):
        name = f"{row.arch}_{row.mode}"
        final[name] = query["last"] * (1 + row.shrink * estimated[row.arch, row.mode, row.epoch])
        neural = validation[row.arch, row.mode, row.epoch, row.shrink].loc[valid_base.index]
        loss, weight = min(
            (annual_score(valid_base + w * (neural - valid_base), meta_ref), w)
            for w in BLEND_WEIGHTS
        )
        final[f"blend_{name}"] = direct + weight * (final[name] - direct)
        selection.append({"model": name, "weight": weight, "validation_rmse_to_rw": loss})
    residual_rows = [
        {
            "penalty": key[0],
            "weight": key[1],
            "validation_rmse_to_rw": annual_score(pd.concat(vals), meta_ref),
        }
        for key, vals in valid_adapter.items()
    ]
    residual_grid = pd.DataFrame(residual_rows).sort_values("validation_rmse_to_rw")
    best = residual_grid.iloc[0]
    final["hybrid_plus_full_context"] = corrected[best.penalty, best.weight]
    if best.weight == 0:
        nonzero = residual_grid.loc[residual_grid.weight.gt(0)].iloc[0]
        final["hybrid_plus_full_context_nonzero"] = corrected[nonzero.penalty, nonzero.weight]
    metrics = summarize(query.assign(ensemble=direct), final)
    paired = pd.concat(
        [
            summarize(
                query.assign(ensemble=final[f"{arch}_market"]),
                {name: final[name] for name in (f"{arch}_{mode}" for mode in CONTEXT_MODES)},
            ).assign(reference=f"{arch}_market")
            for arch in ARCHITECTURES
        ],
        ignore_index=True,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    for name, frame in {
        "validation": ranking,
        "selected": selected,
        "metrics": metrics,
        "blend_validation": pd.DataFrame(selection),
        "adapter_validation": residual_grid,
        "paired_neural": paired,
    }.items():
        write_csv(frame, args.output / f"{name}.csv")
    predictions = pd.DataFrame(final).join(query[["actual", "last"]])
    write_csv(predictions.rename_axis("origin").reset_index(), args.output / "predictions.csv.gz")
    metadata = {
        "all_features": full.shape[1],
        "history_features": int(market.sum()),
        "active_test_features": int(active.sum()),
        "fold_active_features": counts,
        "validation_hybrid_rmse_to_rw": annual_score(valid_base, meta_ref),
        "adapter": best.to_dict(),
        "inputs": input_fingerprints([*paths, args.events]),
    }
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        selected.round(5).to_string(index=False),
        metrics.round(5).to_string(index=False),
        pd.DataFrame(selection).round(5).to_string(index=False),
        residual_grid.head(2).round(5).to_string(index=False),
        sep="\n",
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
    parser.add_argument("--config", type=Path, default=Path("config/ml_holdout.json"))
    parser.add_argument("--output", type=Path, default=Path("data/reports/weekly_level_context"))
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        run(parser.parse_args())
