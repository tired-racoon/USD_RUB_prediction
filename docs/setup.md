# Установка и запуск

Нужны Python 3.11+ и `uv`. Для экспериментов с ML используются зависимости из `ml`, для JupyterLab — из группы `notebook`.

## Эксперимент с ML

[Ноутбук](../notebooks/ml_holdout.ipynb) запускается на [снимке датасета](../datasets/usd_rub_2011_2026.csv.gz), который есть в репозитории.

```sh
uv run --extra ml --group notebook jupyter lab notebooks/ml_holdout.ipynb
```

Запустите все ячейки ноутбука. Тот же эксперимент можно запустить из терминала.

```sh
uv run --extra ml python -m usd_rub.ml_experiment --dataset datasets/usd_rub_2011_2026.csv.gz
```

Результаты сохраняются в `data/reports/ml_holdout_50pct/`. Сводка по моделям находится в `summary.csv`, результаты по каждому горизонту — в `metrics.csv`, проверка вклада групп признаков — в `ablation.csv`.

## Сбор и обновление данных

```sh
uv sync --python 3.12
uv run usd-rub sources
uv run usd-rub fetch --profile core --start 2015-01-01 --end 2026-11-18 --require-10y --browser-fallback
uv run usd-rub fetch --profile extended --start 2015-01-01 --end 2026-11-17
uv run usd-rub fetch --sources cbr_usd,cbr_ruonia,fred_brent
uv run usd-rub fetch --sources nbs_pmi,worldbank_commodities,minfin_budget,cbr_correspondent_accounts --start 2015-01-01 --end 2026-11-18 --browser-fallback
uv run usd-rub ofac --start 2015-01-01 --end 2026-11-18 --documents --output data/documents/ofac_2015_2026.json
uv run usd-rub cbr-releases --start 2015-01-01 --end 2026-11-18
uv run usd-rub sanctions
uv run usd-rub audit
uv run usd-rub repair-data --as-of 2026-11-18
```

Если `--end` не задан, при загрузке используется текущая дата по Москве. Глобальный параметр `--data-dir` ставится перед подкомандой. `--browser-fallback` использует Chrome на macOS для публичных файлов Росстата, когда обычный HTTPS-запрос возвращает ошибку. Бесплатный ключ FRED нужен для исторических винтажей ALFRED. [Доступы к источникам](access.md)

При повторной загрузке используется кэш. `--refresh` скачивает ответы заново и сохраняет предыдущие версии. Исходные ответы находятся в `data/raw/objects/`, журнал загрузок — в `data/raw/history.jsonl`, подготовленные ряды — в `data/normalized/`. Отчёт последней загрузки находится в `data/reports/latest_fetch.json`. [Реестр источников](sources.md) · [Импорт локальных таблиц](imports.md)

После исправлений через `repair-data` пересоберите признаки и повторите прогнозы. Состояние исправленных файлов перед запуском сохраняется в `data/repairs/<дата>/before/`.

## Сборка признаков

```sh
uv run usd-rub features --start 2015-01-01 --end 2026-11-18 --cutoff 15:00
uv run usd-rub features --series usd_rub,cny_tom_close,usd_cny,ruonia,key_rate,ofz_2y,ofz_10y --start 2015-01-01 --end 2026-11-18 --assumed-lags config/assumed_lags.example.json --output data/features/assumed.csv
uv run python scripts/export_ml_dataset.py
```

Последняя команда собирает очищенный CSV для ML из локальных рядов и календарей. Описания правил доступности и столбцов приведены в [отчёте эксперимента](ml_holdout_2025_2026.md). [Обновление новостного индекса](gpr_russia.md)

## Основной прогноз USD/RUB

Команда запускает Ridge, HistGradientBoosting и Extra Trees с весами 43,84%, 38,44% и 17,72%. К их прогнозу после календарной обработки добавляется 50% поправки XGBoost на восьми признаках. Затем календарь применяется повторно. Семь моделей поправки обучены на 1 433 вневыборочных прогнозах классической базы 2021–2024 годов и входят в пакет. [Отбор признаков и сравнение моделей](shap_holdout.md)

```sh
uv run --locked --extra ml usd-rub forecast-ensemble
```

Команда берёт последнюю известную дату из локальных рядов и обучает классические модели на доступной к ней истории. Рыночные и макроэкономические признаки дополняются данными ECB и FRED. Если передан готовый снимок с рассчитанными колонками, команда использует их. Для воспроизведения с зафиксированными выгрузками FRED можно передать `--fred-cache data/reports/theory_holdout/fred`.

Прогноз сохраняется в `data/reports/forecast_ensemble.csv`, дата, веса и контрольные суммы входов — в соседнем JSON. Флаг `--classical-only` запускает классическую базу и подходит для границ обучения раньше 01.01.2025.

Для воспроизведения прежней модели на Mac с Apple Silicon используется `uv run --locked --extra ensemble usd-rub forecast-ensemble --legacy-t0-gamma`. Ей нужен локальный чекпойнт. Флаги `--market-only` и `--base-only` запускают прежнюю рыночную и базовую конфигурации. [Подготовка чекпойнта](foundation_holdout.md#запуск-на-macbook-air-m4)

Для повтора отдельной даты исторического теста используется [снимок основного ансамбля на 69 признаках](../datasets/usd_rub_2011_2026_ensemble.csv.gz) с фиксированной границей обучения.

```sh
uv run --locked --extra ml usd-rub forecast-ensemble \
  --dataset datasets/usd_rub_2011_2026_ensemble.csv.gz \
  --end 2026-11-11 --train-before 2025-01-01 \
  --output data/reports/forecast_ensemble_holdout.csv
```

Поправка XGBoost переобучается по основному снимку и архиву калибровки. [Архив прогнозов](forecast_stacking.md)

```sh
uv run --locked --extra ml python scripts/fit_market_correction.py --compact-macro
```

Для пересборки основного снимка сначала формируется расширенная панель. [Панель макропризнаков](theory_holdout.md)

```sh
uv run --locked --extra ml python scripts/theory_holdout.py
uv run --locked --extra ml python scripts/export_ensemble_dataset.py
```

## Бейзлайны, факторная модель и тесты

```sh
uv run usd-rub backtest --start 2015-01-01 --end 2026-11-18 --min-train 730 --window 730 --test-start 2017-01-01
uv run usd-rub forecast --window 730 --horizon 7
uv run usd-rub factor-backtest --start 2015-01-01 --end 2026-11-18 --test-start 2017-01-01 --window 730 --min-train 730 --mode both --output data/reports/giannone_20261118
uv run pytest -q
```

Параметры `backtest` и `factor-backtest` в этом примере воспроизводят исторические сравнения с началом теста в 2017 году. Для пересчёта бейзлайнов на датах 2025–2026 используется ML-эксперимент выше. [Бейзлайны](baselines_percent_20261118.md) · [Факторная модель](giannone_results.md)
