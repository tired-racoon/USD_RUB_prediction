"""Бустинг и нейросети для знака недельного изменения USD/RUB."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from forecast_stacking import ARCHIVE_FILES, read_panel
from foundation_ensemble import CLASSICAL, RW
from sequence_holdout import (
    ARCHITECTURES,
    BATCH_SIZE,
    CHECKPOINTS,
    LOOKBACK,
    SEED,
    SequenceRegressor,
)
from si_options_holdout import magnitude_forecast, option_blocks, point_forecast
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from torch import nn
from weekly_literature_holdout import HORIZON, gamma_frame, residual_features, summarize
from weekly_sign_size import direction

from usd_rub.ml_dataset import feature_panels, read_dataset
from usd_rub.sources.news_archive import write_csv
from usd_rub.storage import input_fingerprints


def train_network(architecture, x, y, query, checkpoints):
    """Обучить прежнюю архитектуру с одним логитом и бинарной целью."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = SequenceRegressor(architecture, x.shape[-1], 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=0.02)
    loss_fn = nn.BCEWithLogitsLoss()
    x, y, query = (torch.as_tensor(a, dtype=torch.float32) for a in (x, y, query))
    trained = {}
    for epoch in range(1, max(checkpoints) + 1):
        model.train()
        for indices in np.array_split(rng.permutation(len(x)), (len(x) - 1) // BATCH_SIZE + 1):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x[indices]).squeeze(-1), y[indices])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        if epoch in checkpoints:
            model.eval()
            with torch.inference_mode():
                trained[epoch] = np.concatenate(
                    [
                        model(batch).squeeze(-1).sigmoid().numpy()
                        for batch in query.split(BATCH_SIZE)
                    ]
                )
    return trained


def candidates():
    """Параметры сравнения на одинаковых рыночных панелях."""
    return [
        (model, panel, setting)
        for panel in ("market", "market_rr")
        for model, settings in (
            ("catboost", ((2, 100), (2, 250), (4, 100), (4, 250))),
            ("histboost", (4, 8)),
            *((net, CHECKPOINTS) for net in ARCHITECTURES),
        )
        for setting in settings
    ]


def fit_fold(factors, calibration, query, settings):
    """Обучить модели на известных семидневных целях перед началом фолда."""
    history = calibration.loc[calibration.index + HORIZON < query.index[0]]
    y = history.actual.gt(history["last"]).loc[history.actual.ne(history["last"])]
    result = {
        (f"logistic_{name}", name, 0): p for name, p in direction(factors, history, query).items()
    }
    for panel, features in factors.items():
        panel_settings = [(model, setting) for model, key, setting in settings if key == panel]
        if not panel_settings:
            continue
        past = features.loc[y.index]
        past = past.loc[:, past.notna().mean().ge(0.5)]
        scale = past.std(ddof=0).where(lambda s: s.gt(1e-12), 1)
        standardized = ((features[past.columns] - past.mean()) / scale).fillna(0).clip(-6, 6)
        x, q = standardized.loc[y.index], standardized.loc[query.index]
        for model, setting in panel_settings:
            if model == "histboost":
                fitted = HistGradientBoostingClassifier(
                    max_leaf_nodes=setting,
                    max_iter=120,
                    learning_rate=0.05,
                    min_samples_leaf=30,
                    l2_regularization=5,
                    early_stopping=False,
                    random_state=SEED,
                )
            elif model == "catboost":
                depth, iterations = setting
                fitted = CatBoostClassifier(
                    iterations=iterations,
                    depth=depth,
                    learning_rate=0.035,
                    l2_leaf_reg=12,
                    random_seed=SEED,
                    has_time=True,
                    thread_count=2,
                    allow_writing_files=False,
                    verbose=False,
                )
            else:
                continue
            fitted.fit(x, y)
            result[model, panel, setting] = pd.Series(fitted.predict_proba(q)[:, 1], index=q.index)

        offsets = np.arange(1 - LOOKBACK, 1)
        rows, q_rows = (features.index.get_indexer(dates) for dates in (y.index, query.index))
        values = standardized.to_numpy()
        for net in ARCHITECTURES:
            epochs = [setting for model, setting in panel_settings if model == net]
            if epochs:
                probabilities = train_network(
                    net,
                    values[rows[:, None] + offsets],
                    y.to_numpy(),
                    values[q_rows[:, None] + offsets],
                    epochs,
                )
                for epoch, prob in probabilities.items():
                    result[net, panel, epoch] = pd.Series(prob, index=q.index)
    return result


def scores(truth, predictions):
    """Оценить вероятности и знак на датах, когда курс изменился."""
    move = truth.ne(0)
    positive = truth.loc[move].gt(0)
    rows = []
    for (model, panel, setting), values in predictions.items():
        probabilities = values.loc[move[move].index]
        rows.append(
            {
                "model": model,
                "panel": panel,
                "setting": str(setting),
                "dates": int(move.sum()),
                "accuracy": probabilities.ge(0.5).eq(positive).mean(),
                "brier": (probabilities - positive).pow(2).mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("brier", kind="stable")


def pair_intervals(truth, estimates, direct, reference, *, reference_name="logistic"):
    """Сравнить классификаторы с контрольными вероятностями и знаком прямого прогноза."""
    size, block = len(truth), 28
    random = np.random.default_rng(SEED)
    starts = random.integers(size, size=(4000, (size + block - 1) // block))
    draws = ((starts[:, :, None] + np.arange(block)) % size).reshape(len(starts), -1)[:, :size]
    active = truth.ne(0).to_numpy()
    positive = truth.gt(0).to_numpy()
    sign = np.sign(truth.to_numpy())
    direct_good = np.sign(direct.to_numpy()) == sign
    pair_rows = []
    for model, probability in estimates.items():
        p = probability.to_numpy()
        good = (p >= 0.5) == positive
        comparisons = {
            "accuracy_vs_direct": good.astype(float) - direct_good,
            f"accuracy_vs_{reference_name}": good.astype(float) - ((reference >= 0.5) == positive),
            f"brier_vs_{reference_name}": (reference - positive) ** 2 - (p - positive) ** 2,
        }
        for metric, difference in comparisons.items():
            values = np.where(active, difference, 0)[draws].sum(axis=1) / active[draws].sum(axis=1)
            low, high = np.quantile(values, (0.025, 0.975))
            pair_rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "gain": difference[active].mean(),
                    "gain_ci_low": low,
                    "gain_ci_high": high,
                }
            )
    return pd.DataFrame(pair_rows)


def run(args):
    dataset = read_dataset(args.dataset)
    panel = feature_panels(dataset)["extended"]
    manifest = args.archive / "archives.json"
    groups = json.loads(manifest.read_text())["groups"]
    archives = [args.archive / name for name in ARCHIVE_FILES]
    names = [CLASSICAL, RW, *sum(groups.values(), [])]
    calibration, test = [gamma_frame(read_panel(path, names)) for path in archives]
    for data in (calibration, test):
        np.testing.assert_allclose(data["last"], dataset.usd_rub.loc[data.index], rtol=1e-12)
        np.testing.assert_allclose(
            data.actual, dataset.usd_rub.loc[data.index + HORIZON], rtol=1e-12
        )
    if calibration.index.max() + HORIZON >= test.index.min():
        raise ValueError("Архив обучения пересекает тест")

    inputs = residual_features(panel)["spread_dynamics"]
    blocks = option_blocks(panel)
    archive = pd.concat([calibration, test])
    trend = ((archive.ensemble - archive["last"]) / archive["last"]).rename("gamma_return")
    market = trend.to_frame().join(inputs, how="right").sort_index()
    factors = {"market": market, "market_rr": market.join(blocks["rr"])}
    validation, grid = {}, candidates()
    for year, fold in calibration.loc["2022":].groupby(calibration.loc["2022":].index.year):
        for key, values in fit_fold(factors, calibration, fold, grid).items():
            validation.setdefault(key, []).append(values)
        print(f"Готов фолд {year}", flush=True)
    valid = {key: pd.concat(vals) for key, vals in validation.items()}
    valid_truth = (calibration.actual - calibration["last"]).loc["2022":]
    ranking = scores(valid_truth, valid)
    by_model = ranking.drop_duplicates("model").set_index("model")
    lookup = {
        (model, panel_name, str(setting)): (model, panel_name, setting)
        for model, panel_name, setting in grid
    }
    new_ranking = ranking.loc[ranking.model.isin({model for model, _, _ in grid})]
    selected = [
        lookup[row.model, row.panel, row.setting]
        for row in new_ranking.drop_duplicates("model").itertuples(index=False)
    ]
    accuracy_choice = new_ranking.sort_values(["accuracy", "brier"], ascending=[False, True]).iloc[
        0
    ]
    accuracy_key = lookup[accuracy_choice.model, accuracy_choice.panel, accuracy_choice.setting]
    out = fit_fold(
        factors,
        calibration,
        test,
        [*selected, *([accuracy_key] if accuracy_key not in selected else [])],
    )
    if accuracy_key not in selected:
        out["selected_by_accuracy", accuracy_key[1], accuracy_key[2]] = out.pop(accuracy_key)
    neural = {key: values for key, values in out.items() if key[0] in ARCHITECTURES}
    out["mean_neural", "selected", 0] = pd.DataFrame(neural).mean(axis=1)
    actual = test.actual - test["last"]
    baseline = point_forecast(inputs, calibration, test, test.index[0])
    headline = scores(actual, out)
    direct_accuracy = (
        np.sign(baseline - test["last"])
        .loc[actual.ne(0)]
        .eq(np.sign(actual.loc[actual.ne(0)]))
        .mean()
    )
    headline.loc[len(headline)] = [
        "direct_point",
        "existing",
        "0",
        int(actual.ne(0).sum()),
        direct_accuracy,
        np.nan,
    ]

    vol = panel[
        ["own__vol_7", "own__vol_21", "own__vol_60", "markets__rvi_level", "markets__rvi_change_5"]
    ]
    size = magnitude_forecast(vol.join(blocks["iv"]), calibration, test, test.index[0])
    probabilities = {key[0]: values for key, values in out.items()}
    point = {
        "direct": baseline,
        **{
            name: test["last"] + (size * (2 * values - 1)).where(test.new_fixing, 0)
            for name, values in probabilities.items()
        },
    }
    point_metrics = summarize(test.assign(ensemble=baseline), point)
    paired = pair_intervals(
        actual, probabilities, baseline - test["last"], probabilities["logistic_market"].to_numpy()
    )
    predictions = pd.DataFrame(probabilities).join(test[["last", "actual"]])
    predictions["direct"] = baseline
    predictions["size"] = size
    args.output.mkdir(parents=True, exist_ok=True)
    for name, table in {
        "validation": ranking,
        "sign_test": headline,
        "paired": paired,
        "course_test": point_metrics,
    }.items():
        write_csv(table, args.output / f"{name}.csv")
    write_csv(predictions.rename_axis("origin").reset_index(), args.output / "predictions.csv.gz")
    meta = {
        "selected_new": selected[0],
        "selected_by_accuracy": accuracy_key,
        "selected_families": selected,
        "lookback_days": LOOKBACK,
        "inputs": input_fingerprints([args.dataset, manifest, *archives]),
    }
    (args.output / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        by_model.round(5).to_string(),
        headline.round(5).to_string(index=False),
        point_metrics.round(5).to_string(index=False),
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
    parser.add_argument("--archive", type=Path, default=Path("data/reports/forecast_stacking"))
    parser.add_argument("--output", type=Path, default=Path("data/reports/weekly_direction_models"))
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        run(parser.parse_args())
