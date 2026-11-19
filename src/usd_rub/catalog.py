"""Реестр загрузчиков и показателей из материалов проекта."""

from dataclasses import dataclass, field

from usd_rub.sources import cbr, markets, tables


@dataclass(frozen=True)
class Source:
    name: str
    description: str
    kind: str
    config: dict = field(default_factory=dict)
    profile: str = "core"
    limitation: str = ""
    enabled: bool = True


SOURCES = {
    s.name: s
    for s in [
        Source("cbr_usd", "Официальный USD/RUB", "fx", {"code": "R01235", "series": "usd_rub"}),
        Source("cbr_cny", "Официальный CNY/RUB", "fx", {"code": "R01375", "series": "cny_rub"}),
        *[
            Source(
                f"moex_{name}",
                description,
                "moex",
                {
                    "path": f"history/engines/stock/markets/index/boards/{board}/securities/{ticker}",
                    "columns": {"CLOSE": (name, unit)},
                },
                "extended",
                limitation,
            )
            for name, description, board, ticker, unit, limitation in [
                ("rgbi", "Индекс гособлигаций RGBI", "SNDX", "RGBI", "index points", ""),
                ("moexog", "Индекс нефти и газа MOEXOG", "SNDX", "MOEXOG", "index points", ""),
                (
                    "rusfar_cny",
                    "Ставка RUSFAR CNY overnight",
                    "MMIX",
                    "RUSFARCNY",
                    "% p.a.",
                    "История с сентября 2022",
                ),
            ]
        ],
        Source(
            "cbr_eur",
            "Официальный EUR/RUB",
            "fx",
            {"code": "R01239", "series": "eur_rub"},
            "extended",
        ),
        Source("cbr_key_rate", "Ключевая ставка", "soap", {"table": "key_rate"}),
        Source("cbr_ruonia", "RUONIA и объём сделок", "soap", {"table": "ruonia"}),
        *[
            Source("cbr_" + name, description, "soap", {"table": name}, "extended")
            for name, description in [
                ("correspondent_accounts", "Корсчета банков: РФ и Московский регион"),
                ("bank_deposits", "Депозиты банков в ЦБ: всего и overnight"),
                ("deposit_rate", "Ставка депозитных операций overnight ЦБ"),
                (
                    "repo_claims",
                    "Требования ЦБ по РЕПО: всего, основной аукцион и тонкая настройка",
                ),
            ]
        ],
        Source("cbr_ofz", "Бескупонная кривая ОФЗ: 1, 2, 5, 10 лет", "soap", {"table": "ofz"}),
        Source("cbr_liquidity", "Структурный дефицит (+) / профицит (-)", "liquidity"),
        Source(
            "cbr_saldo",
            "Сальдо операций по ликвидности",
            "soap",
            {"table": "liquidity_operations"},
            "extended",
            "SOAP-ряд в проверенной выгрузке заканчивается 30.12.2025",
            False,
        ),
        Source("cbr_reserves", "Международные резервы, еженедельно", "soap", {"table": "reserves"}),
        Source("cbr_money", "Денежные агрегаты M2 и M2X", "money"),
        Source("cbr_trade", "Экспорт, импорт и торговый баланс", "trade"),
        Source(
            "fns_calendar",
            "Архив налогового календаря и выходных ФНС",
            "tax_calendar",
            profile="extended",
            limitation="Количество записей об уплате не равно сумме налогов; версии и даты выпуска сохраняются",
        ),
        Source(
            "economy_urals",
            "Официальные месячные цены Urals",
            "urals",
            profile="extended",
            limitation="Сайт публикует текущие документы; архив до 2018 года может отсутствовать",
        ),
        Source(
            "rosstat_industry",
            "Промпроизводство: месячные индексы",
            "rosstat_extra",
            {"table": "industry"},
            "extended",
            "Старый файл возвращает 404; актуальная таблица с 2020 года",
        ),
        Source(
            "rosstat_weekly_items",
            "Недельные индексы отдельных товаров, не общий ИПЦ",
            "rosstat_extra",
            {"table": "weekly_items"},
            "extended",
            "Проверенный файл содержит историю с 2022 года",
        ),
        Source(
            "cbr_orfr",
            "Продажи валюты компаниями и спрос населения из обзоров ЦБ",
            "orfr",
            profile="extended",
            limitation="Извлекаются только однозначные числовые фрагменты; разные определения продавцов разделены",
        ),
        Source(
            "minfin_fnb", "ФНБ: объём, поступления, изъятия и переоценка", "fnb", profile="extended"
        ),
        Source(
            "minfin_fx_budget",
            "Бюджетные валютные операции, НДПИ/НДД",
            "budget_operations",
            profile="extended",
            limitation="Валютные операции включают золото; объявленный месячный объём не равен фактическим дневным торгам",
        ),
        Source(
            "cbr_bop",
            "Платёжный баланс: текущий счёт, услуги, доходы",
            "cbr_extra",
            {"table": "bop"},
            "extended",
        ),
        Source(
            "cbr_bop_estimate",
            "Ранняя оценка платежного баланса: отдельный текущий винтаж",
            "cbr_extra",
            {"table": "bop_estimate"},
            "extended",
            "Не склеивается с подробным платежным балансом; доступна с даты обновления файла",
        ),
        Source(
            "cbr_business",
            "Деловой климат и ожидания предприятий",
            "cbr_extra",
            {"table": "business"},
            "extended",
        ),
        Source(
            "cbr_expectations",
            "Ожидаемая и наблюдаемая инфляция населения",
            "cbr_extra",
            {"table": "expectations"},
            "extended",
        ),
        Source(
            "cbr_cny_funding",
            "Вменённая юаневая ставка overnight",
            "cbr_extra",
            {"table": "cny_funding"},
            "extended",
            "Ряд с 2023 года, десяти лет нет",
        ),
        Source(
            "cbr_settlements",
            "Валютная структура экспортных и импортных расчётов",
            "cbr_extra",
            {"table": "settlements"},
            "extended",
            "Новая классификация государств; не USD/CNY по отдельности",
        ),
        Source(
            "cbr_fx_operations",
            "Покупка и продажа валюты ЦБ",
            "fx_operations",
            profile="extended",
            limitation="Таблица ЦБ начинается в 2017; в проверке лишь 2 числовые записи. Не тождественна операциям Минфина",
            enabled=False,
        ),
        Source(
            "cbr_meetings",
            "История плановых заседаний по ставке",
            "meetings",
            profile="extended",
            limitation="Архив календаря не доказывает, когда расписание стало известно",
        ),
        Source(
            "cbr_otc",
            "Внебиржевые обороты USD/EUR/CNY",
            "soap",
            {"table": "otc"},
            "extended",
            "Полученная история 04.10.2022–11.07.2024; десяти лет нет, ряд устарел",
            False,
        ),
        Source(
            "moex_cny",
            "CNYRUB_TOM: OHLC, VWAP, число сделок",
            "moex",
            {
                "path": "history/engines/currency/markets/selt/boards/CETS/securities/CNYRUB_TOM",
                "columns": {
                    column: (f"cny_tom_{name}", unit)
                    for column, name, unit in [
                        ("OPEN", "open", "RUB/CNY"),
                        ("HIGH", "high", "RUB/CNY"),
                        ("LOW", "low", "RUB/CNY"),
                        ("CLOSE", "close", "RUB/CNY"),
                        ("WAPRICE", "vwap", "RUB/CNY"),
                        ("NUMTRADES", "trades", "trades"),
                    ]
                },
            },
        ),
        Source(
            "moex_cny_volume",
            "Объём CNYRUB_TOM",
            "moex",
            {
                "path": "engines/currency/markets/selt/boards/CETS/securities/CNYRUB_TOM/candles",
                "block": "candles",
                "columns": {"volume": ("cny_tom_volume", "CNY")},
            },
            "extended",
            "В проверенном бесплатном ответе volume=null; пустой ряд считается ошибкой",
        ),
        Source(
            "moex_imoex",
            "Индекс IMOEX",
            "moex",
            {
                "path": "history/engines/stock/markets/index/boards/SNDX/securities/IMOEX",
                "columns": {"CLOSE": ("imoex", "index points")},
            },
        ),
        Source(
            "moex_usd_fixing",
            "Валютный фиксинг MOEX USD/RUB",
            "moex",
            {
                "path": "history/engines/currency/markets/index/securities/USDFIXME",
                "columns": {"CLOSE": ("moex_usd_fixing", "RUB/USD")},
            },
            "extended",
            "В полученном ISS-ответе ряд начинается 01.08.2019; десяти лет истории нет",
        ),
        Source("cboe_vix", "VIX от Cboe", "cboe"),
        Source(
            "nbs_pmi",
            "PMI Китая: промышленность, выпуск и новые заказы",
            "nbs_pmi",
            profile="extended",
            limitation="Официальный NBS, не Caixin; текущие тексты архивных таблиц",
        ),
        Source(
            "worldbank_commodities",
            "Золото и LNG Япония: World Bank Pink Sheet",
            "pink_sheet",
            profile="extended",
            limitation="Месячные цены; LNG Japan не является дневным JKM",
        ),
        Source(
            "minfin_budget",
            "Доходы, расходы, баланс бюджета, НДС и прибыль",
            "budget",
            profile="extended",
            limitation="Баланс: дефицит отрицательный, профицит положительный; текущий винтаж",
        ),
        Source(
            "moex_si_front",
            "Ближайший Si: цена, расчётная цена, OI и объём",
            "si_front",
            profile="extended",
            limitation="Роллирование за 5 календарных дней до последнего дня торгов; цены без склейки задним числом",
        ),
        *[
            Source(
                f"moex_{ticker.lower()}",
                ticker + ": perpetual фьючерс",
                "moex",
                {
                    "path": f"history/engines/futures/markets/forts/boards/RFUD/securities/{ticker}",
                    "columns": {
                        "CLOSE": (ticker.lower() + "_close", unit),
                        "OPENPOSITION": (ticker.lower() + "_oi", "contracts"),
                        "VOLUME": (ticker.lower() + "_volume", "contracts"),
                        "SWAPRATE": (ticker.lower() + "_swaprate", "MOEX SWAPRATE native units"),
                    },
                },
                "extended",
                "Новый инструмент, десяти лет истории нет",
            )
            for ticker, unit in [("USDRUBF", "RUB/USD"), ("CNYRUBF", "RUB/CNY")]
        ],
        Source(
            "gpr_monthly",
            "GPR, threats, acts, Россия: месячные индексы",
            "gpr",
            {"daily": False},
            "extended",
            "Текущий винтаж пересматривается; архив винтажей доступен у авторов",
        ),
        Source(
            "gpr_ru_press",
            "Геополитический риск для России по заголовкам Интерфакса и МК",
            "gpr_ru_press",
            profile="research",
            limitation="Архивные заголовки, равные веса изданий; текущий винтаж",
        ),
        Source(
            "gpr_daily",
            "GPR: дневные индексы",
            "gpr",
            {"daily": True},
            "extended",
            "Дневные наблюдения публикуются раз в неделю; дата наблюдения не дата публикации",
        ),
        Source(
            "rosstat_cpi",
            "ИПЦ РФ: месяц к предыдущему месяцу",
            "rosstat",
            {
                "page": "https://rosstat.gov.ru/statistics/price/",
                "pattern": r"Индексы потребительских цен.*месяц.*1991",
                "series": "cpi_ru_mom_index",
                "unit": "% previous month=100",
                "header_pattern": r"предыдущего месяца",
                "sheet": "01",
            },
            "core",
            "На macOS резервная загрузка через --browser-fallback; также поддержан импорт XLSX",
        ),
        Source(
            "rosstat_retail",
            "Оборот розничной торговли",
            "rosstat",
            {
                "page": "https://rosstat.gov.ru/statistics/roznichnayatorgovlya",
                "pattern": r"Оборот розничной торговли\s*\(с\s*2000",
                "series": "retail_turnover",
                "unit": "RUB bn",
            },
            "extended",
            "Проверенные ссылки Oborot_11-2025.xls и Oborot_m_01-2026.xlsx возвращают 404",
        ),
        Source(
            "zenodo_research",
            "Исследовательский датасет из DOCX",
            "zenodo",
            profile="research",
            limitation="Качество не подтверждено; не используется для обучения и расчётных признаков",
            enabled=False,
        ),
        Source(
            "minfin_oil_gas",
            "Нефтегазовые доходы федерального бюджета",
            "minfin",
            profile="core",
            limitation="Месячные суммы рассчитаны из накопленных, исходный ряд сохраняется отдельно",
        ),
    ]
}

FRED_SERIES = {
    "usd_brl": ("DEXBZUS", "BRL/USD", "Бразильский реал за USD: компонент EM FX"),
    "usd_mxn": ("DEXMXUS", "MXN/USD", "Мексиканский песо за USD: компонент EM FX"),
    "usd_zar": ("DEXSFUS", "ZAR/USD", "Южноафриканский ранд за USD: компонент EM FX"),
    "usd_inr": ("DEXINUS", "INR/USD", "Индийская рупия за USD: компонент EM FX"),
    "usd_cny": ("DEXCHUS", "CNY/USD", "USD/CNY: H.10, дневные наблюдения, недельная публикация"),
    "brent": ("DCOILBRENTEU", "USD/bbl", "Brent spot Europe"),
    "ust_2y": ("DGS2", "% p.a.", "UST 2Y"),
    "ust_10y": ("DGS10", "% p.a.", "UST 10Y"),
    "ust_10y_2y": ("T10Y2Y", "percentage points", "Спред UST 10Y–2Y"),
    "usd_broad": ("DTWEXBGS", "index Jan 2006=100", "Широкий индекс USD Fed; это не DXY ICE"),
    "fed_funds": ("DFF", "% p.a.", "Эффективная ставка федеральных фондов"),
    "cpi_us": ("CPIAUCSL", "index 1982-84=100", "ИПЦ США, сезонно скорректированный"),
    "vix_fred": ("VIXCLS", "index points", "VIX через FRED"),
    "wti": ("DCOILWTICO", "USD/bbl", "Нефть WTI"),
    "gas_eu": ("PNGASEUUSDM", "USD/MMBtu", "Газ Европа, месячный proxy"),
}

for name, (series_id, unit, description) in FRED_SERIES.items():
    SOURCES[f"fred_{name}"] = Source(
        f"fred_{name}",
        description,
        "fred",
        {"series_id": series_id, "name": name, "unit": unit},
        "core"
        if name
        in {
            "usd_cny",
            "brent",
            "ust_2y",
            "ust_10y",
            "ust_10y_2y",
            "usd_broad",
            "fed_funds",
            "cpi_us",
        }
        else "extended",
        "Текущая версия ряда; исторические моменты публикации не восстановлены. ALFRED — отдельно.",
    )


def fetch(source: Source, client, start, end):
    if not source.enabled:
        raise ValueError(f"Источник {source.name} исключён: {source.limitation}")
    config = source.config
    if source.kind == "fx":
        result = cbr.fetch_fx(client, start=start, end=end, **config)
    elif source.kind == "soap":
        result = cbr.fetch_soap_table(client, config["table"], start, end)
    elif source.kind == "fred":
        result = markets.fetch_fred(client, start=start, end=end, **config)
    elif source.kind == "moex":
        result = markets.fetch_moex(client, config, start, end)
    elif source.kind == "rosstat":
        result = tables.fetch_rosstat(client, config, start, end)
    elif source.kind == "minfin":
        from usd_rub.sources.minfin import fetch_oil_gas_revenue

        result = fetch_oil_gas_revenue(client, start, end)
    elif source.kind == "budget":
        from usd_rub.sources.minfin import fetch_oil_gas_revenue

        result = fetch_oil_gas_revenue(client, start, end, budget=True)
    elif source.kind == "pink_sheet":
        from usd_rub.sources.commodities import fetch_pink_sheet

        result = fetch_pink_sheet(client, start, end)
    elif source.kind == "nbs_pmi":
        from usd_rub.sources.china import fetch_pmi

        result = fetch_pmi(client, start, end)
    elif source.kind == "fnb":
        from usd_rub.sources.minfin import fetch_fnb

        result = fetch_fnb(client, start, end)
    elif source.kind == "budget_operations":
        from usd_rub.sources.minfin import fetch_budget_operations

        result = fetch_budget_operations(client, start, end)
    elif source.kind == "orfr":
        from usd_rub.sources.reports import fetch_orfr

        result = fetch_orfr(client, start, end)
    elif source.kind == "si_front":
        from usd_rub.sources.futures import fetch_si_front

        result = fetch_si_front(client, start, end)
    elif source.kind == "urals":
        from usd_rub.sources.urals import fetch_urals

        result = fetch_urals(client, start, end)
    elif source.kind == "tax_calendar":
        from usd_rub.sources.tax_calendar import fetch_tax_calendar

        result = fetch_tax_calendar(client, start, end)
    elif source.kind == "gpr_ru_press":
        from usd_rub.sources.gpr_ru_press import fetch_ru_press

        result = fetch_ru_press(client, start, end)
    elif source.kind == "gpr":
        from usd_rub.sources.gpr import fetch_gpr

        result = fetch_gpr(client, start, end, **config)
    elif source.kind == "cbr_extra":
        from usd_rub.sources import cbr_extra

        result = getattr(cbr_extra, "fetch_" + config["table"])(client, start, end)
    elif source.kind == "rosstat_extra":
        from usd_rub.sources import rosstat_extra

        result = getattr(rosstat_extra, "fetch_" + config["table"])(client, start, end)
    else:
        from usd_rub.sources.events import fetch_meetings

        loader = {
            "liquidity": cbr.fetch_liquidity,
            "money": tables.fetch_money,
            "trade": tables.fetch_trade,
            "cboe": markets.fetch_cboe,
            "zenodo": tables.fetch_zenodo,
            "meetings": fetch_meetings,
            "fx_operations": cbr.fetch_fx_operations,
        }[source.kind]
        result = loader(client, start, end)
    return result.loc[result["date"].between(str(start), str(end))].reset_index(drop=True)
