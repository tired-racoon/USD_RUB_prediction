"""Рыночная поправка XGBoost к базовому прогнозу USD/RUB."""

import gzip
import json
from pathlib import Path

import numpy as np

from usd_rub.fixing_calendar import HORIZONS, apply_calendar

MARKET_FEATURES = (
    "own__return_1",
    "own__return_5",
    "own__vol_21",
    "fx__cny_basis",
    "fx__si_front_settle_basis",
    "global__synthetic_basis",
)
MARKET_FRACTION = 0.25
MARKET_BUNDLE = Path(__file__).with_name("market_correction.json.gz")
XGB_PARAMS = {
    "n_estimators": 80,
    "max_depth": 2,
    "learning_rate": 0.04,
    "max_bin": 128,
    "min_child_weight": 20,
    "reg_lambda": 15,
    "tree_method": "hist",
    "n_jobs": 2,
    "random_state": 191,
}


def fit_corrections(features, actual, baseline, last, signs=None):
    """Обучить семь моделей логарифмической ошибки базового прогноза."""
    from xgboost import XGBRegressor

    target = 100 * np.log(actual / baseline)
    weights = (last / np.median(last)) ** 2
    models = []
    for h in range(len(HORIZONS)):
        model = XGBRegressor(**XGB_PARAMS, monotone_constraints=signs or None)
        model.fit(
            features.assign(anchor_return=np.log(baseline[:, h] / last)),
            target[:, h],
            sample_weight=weights,
        )
        models.append(model.get_booster())
    return models


def predict_levels(models, features, baseline, last):
    """Перевести логарифмические поправки в прогнозы курса."""
    predictions = []
    for h, model in enumerate(models):
        delta = model.inplace_predict(features.assign(anchor_return=np.log(baseline[:, h] / last)))
        predictions.append(baseline[:, h] * np.exp(delta.astype(float) / 100))
    return np.column_stack(predictions)


def market_forecast(models, panel, baseline, last, changes):
    """Добавить четверть поправки и согласовать горизонты по календарю ЦБ."""
    corrected = predict_levels(models, panel[list(MARKET_FEATURES)], baseline, last)
    prediction = apply_calendar(
        baseline + MARKET_FRACTION * (corrected - baseline), last, changes, pool=True
    )
    if not (np.isfinite(prediction).all() and (prediction > 0).all()):
        raise ValueError("Рыночная поправка дала некорректный прогноз")
    return prediction


def load_market_bundle(path=MARKET_BUNDLE):
    """Загрузить обученные деревья и проверить порядок рыночных входов."""
    try:
        from xgboost import Booster
    except ImportError as exc:
        raise ValueError("Для рыночной поправки нужен uv run --extra ensemble") from exc

    with gzip.open(path, "rt", encoding="utf-8") as stream:
        bundle = json.load(stream)
    models = []
    expected = [*MARKET_FEATURES, "anchor_return"]
    if bundle["format_version"] != 1:
        raise ValueError("Неизвестный формат набора XGBoost")
    if bundle["horizons"] != HORIZONS.tolist() or bundle["features"] != expected:
        raise ValueError("Горизонты или признаки набора XGBoost расходятся со спецификацией")
    for entry in bundle["boosters"]:
        model = Booster(params={"nthread": XGB_PARAMS["n_jobs"]})
        model.load_model(bytearray(json.dumps(entry, separators=(",", ":")).encode("utf-8")))
        if model.feature_names != expected:
            raise ValueError("Некорректные признаки в модели XGBoost")
        models.append(model)
    if len(models) != len(HORIZONS):
        raise ValueError("Набор XGBoost должен содержать семь моделей")
    return models, {key: value for key, value in bundle.items() if key != "boosters"}
