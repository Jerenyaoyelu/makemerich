from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from core.models import StockSignal


def load_sample_signals(csv_path: str | Path) -> List[StockSignal]:
    frame = pd.read_csv(csv_path)
    required_cols = {
        "symbol",
        "name",
        "theme",
        "theme_strength",
        "sector_linkage",
        "stock_strength",
        "capital_support",
        "risk_tag",
    }
    missing = required_cols - set(frame.columns)
    if missing:
        raise ValueError(f"样例数据缺少列: {sorted(missing)}")

    signals: List[StockSignal] = []
    for _, row in frame.iterrows():
        signals.append(
            StockSignal(
                symbol=str(row["symbol"]),
                name=str(row["name"]),
                theme=str(row["theme"]),
                theme_strength=float(row["theme_strength"]),
                sector_linkage=float(row["sector_linkage"]),
                stock_strength=float(row["stock_strength"]),
                capital_support=float(row["capital_support"]),
                risk_tag=str(row["risk_tag"]),
            )
        )
    return signals

