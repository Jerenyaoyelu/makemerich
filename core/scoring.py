from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import pandas as pd

from core.models import StockSignal


def score_signals(
    signals: Iterable[StockSignal],
    w_theme: float,
    w_sector: float,
    w_stock: float,
    w_capital: float,
) -> pd.DataFrame:
    rows = []
    for item in signals:
        total_score = (
            item.theme_strength * w_theme
            + item.sector_linkage * w_sector
            + item.stock_strength * w_stock
            + item.capital_support * w_capital
        )
        row = asdict(item)
        row["total_score"] = round(total_score, 2)
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(by="total_score", ascending=False).reset_index(drop=True)

