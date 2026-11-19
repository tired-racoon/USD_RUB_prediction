import json
from importlib import import_module
from pathlib import Path

import pandas as pd
import pytest


def test_llm_events_enter_the_next_day_and_keep_unknown_days(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    event_features = import_module("llm_news_holdout").event_features
    validate_events = import_module("label_llm_news").validate_events
    content = '{"events":[{"channel":"rates","stage":"implemented","ids":[1]}]}'
    events = validate_events(content, {1})
    with pytest.raises(ValueError):
        validate_events(content, {2})

    days = tmp_path / "days"
    days.mkdir()
    (tmp_path / "label_run.json").write_text(
        json.dumps({"start": "2025-01-01", "end": "2025-01-04"})
    )
    for day, status, values in [
        ("2025-01-01", "ok", events),
        ("2025-01-02", "error", []),
        ("2025-01-03", "ok", []),
        ("2025-01-04", "ok", []),
    ]:
        (days / f"{day}.json").write_text(
            json.dumps({"status": status, "events": values, "selection": []})
        )
    frame = event_features(tmp_path)
    column = "news__llm_event_rates_implemented_1d"
    assert frame.loc["2025-01-02", column] == 1
    assert pd.isna(frame.loc["2025-01-03", column])
    assert frame.loc["2025-01-04", column] == 0
    assert pd.isna(frame.loc["2025-01-05", "news__llm_event_rates_implemented_3d"])
