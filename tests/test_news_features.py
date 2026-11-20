import numpy as np
import pandas as pd
import pytest

from usd_rub.news_features import aggregate_news, headline_key
from usd_rub.sources.news_archive import press_topic
from usd_rub.sources.rbc_telegram import parse_rbc_page


@pytest.mark.parametrize(
    ("source", "url", "title", "topic"),
    [
        ("interfax", "https://www.interfax.ru/russia/1", "ЦБ повысил ключевую ставку", "economy"),
        ("interfax", "https://www.interfax.ru/world/1", "В ЕС обсуждают санкции", "politics"),
        ("interfax", "https://www.sport-interfax.ru/1", "Продажа акций клуба", None),
        ("mk", "https://www.mk.ru/economics/2025/01/01/a.html", "Новые правила", "economy"),
    ],
)
def test_press_topic(source, url, title, topic):
    assert press_topic(source, url, title) == topic


def test_headline_key_keeps_the_sign_of_a_return():
    assert headline_key("Рынок −2%") == headline_key("рынок -2%")
    assert headline_key("Рынок -2%") != headline_key("Рынок 2%")


def test_rbc_page_uses_moscow_publication_date():
    content = """
    <div data-post="rbc_news/55">
        <div class="tgme_widget_message_text">ЦБ снизил <b>ставку</b><br>Подробности далее</div>
        <a class="tgme_widget_message_date"><time datetime="2025-01-01T22:00:00Z"/></a>
    </div>
    """.encode()
    [(number, posted, headline)] = parse_rbc_page(content, 56)
    assert (number, posted.date().isoformat(), headline) == (55, "2025-01-02", "ЦБ снизил ставку")
    with pytest.raises(ValueError, match="не подтверждена"):
        parse_rbc_page(content, 55)


def test_news_uses_previous_day_and_balances_outlets():
    days = pd.date_range("2025-01-01", periods=3)
    articles = pd.DataFrame(
        [
            (days[0], "interfax", "economy", "Курс вырос"),
            (days[0], "interfax", "economy", "Курс вырос!"),
            (days[0], "interfax", "economy", "Курс упал"),
            (days[0], "mk", "economy", "курс вырос"),
            (days[1], "interfax", "politics", "Переговоры отменили"),
            (days[2], "mk", "economy", "Ставку снизили"),
            (days[2], "interfax", "economy", "Ставку оставили прежней"),
        ],
        columns=["date", "source", "topic", "headline"],
    )
    probabilities = np.array(
        [[0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 0]]
    )
    vectors = np.array([[1, 0], [1, 0], [0, 1], [1, 0], [1, 0], [0, 1], [0, 1]])
    daily, by_source, scored = aggregate_news(
        articles, np.arange(len(articles)), vectors, probabilities, days
    )
    assert daily.index.equals(days + pd.Timedelta(days=1))
    row = daily.iloc[0]
    assert row.news__press_articles == 3
    assert row.news__press_unique_headlines == 2
    assert row.news__press_positive_share == pytest.approx(2 / 3)
    assert row.news__press_positive_negative_ratio == pytest.approx(2)
    assert row.news__press_embedding_000 == pytest.approx(2 / 3)
    assert daily.iloc[1].news__press_sources_present == 1
    assert np.isnan(daily.iloc[1].news__press_sentiment_balance)
    assert daily.iloc[2].news__press_neutral_share == 1
    assert by_source.iloc[0].articles == 2
    assert scored.iloc[0].label == "positive"
