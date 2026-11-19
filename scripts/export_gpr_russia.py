"""Выгрузка дневной истории и сопоставимых месячных наблюдений для России."""

import argparse
from pathlib import Path

from usd_rub.sources.gpr_ru_press import COUNTS, DAILY, MONTHLY
from usd_rub.storage import read_series


def latest(root, name):
    frame = read_series(root, name).sort_values("ingested_at")
    return frame.drop_duplicates("date", keep="last").set_index("date").sort_index()


def export(root, output):
    output.mkdir(parents=True, exist_ok=True)
    daily = latest(root, DAILY)[["value", *COUNTS]].rename(columns={"value": DAILY})
    monthly = latest(root, MONTHLY)[["value", *COUNTS]].rename(columns={"value": MONTHLY})
    original = latest(root, "gpr_russia_monthly")["value"].rename("gpr_russia_monthly")
    monthly = monthly.join(original)
    daily.to_csv(output / "gpr_russia_daily.csv")
    monthly.to_csv(output / "gpr_russia_monthly.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("docs"))
    args = parser.parse_args()
    export(args.data_dir, args.output)
