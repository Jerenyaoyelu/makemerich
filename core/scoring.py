from __future__ import annotations

import pandas as pd

REQUIRED_SCORE_COLS = {
    "theme_strength",
    "sector_linkage",
    "stock_strength",
    "capital_support",
}


def score_frame(
    frame: pd.DataFrame,
    w_theme: float,
    w_sector: float,
    w_stock: float,
    w_capital: float,
) -> pd.DataFrame:
    missing = REQUIRED_SCORE_COLS - set(frame.columns)
    if missing:
        raise ValueError(f"缺少打分字段: {sorted(missing)}")
    if frame.empty:
        return frame
    scored = frame.copy()
    scored["total_score"] = (
        scored["theme_strength"] * w_theme
        + scored["sector_linkage"] * w_sector
        + scored["stock_strength"] * w_stock
        + scored["capital_support"] * w_capital
    ).round(2)
    return scored.sort_values(by="total_score", ascending=False).reset_index(drop=True)

