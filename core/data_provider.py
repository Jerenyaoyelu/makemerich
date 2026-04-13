from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from core.models import StockSignal

REQUIRED_COLS = [
    "symbol",
    "name",
    "theme",
    "theme_strength",
    "sector_linkage",
    "stock_strength",
    "capital_support",
    "risk_tag",
]


def _pick_col(columns: Iterable[str], candidates: List[str]) -> str | None:
    for c in candidates:
        if c in columns:
            return c
    return None


def _normalize_0_100(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    min_v = values.min()
    max_v = values.max()
    if max_v - min_v < 1e-9:
        return pd.Series([50.0] * len(values), index=values.index)
    return ((values - min_v) / (max_v - min_v) * 100).clip(0, 100)


def _build_risk_tag(amplitude: pd.Series, turnover: pd.Series) -> pd.Series:
    amp = pd.to_numeric(amplitude, errors="coerce").fillna(0.0)
    turn = pd.to_numeric(turnover, errors="coerce").fillna(0.0)
    conditions = [
        (amp >= 12) | (turn >= 18),
        (amp >= 7) | (turn >= 10),
    ]
    choices = ["高波动", "中波动"]
    return pd.Series(np.select(conditions, choices, default="低波动"), index=amp.index)


def _ensure_required(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED_COLS) - set(frame.columns)
    if missing:
        raise ValueError(f"数据缺少字段: {sorted(missing)}")
    return frame


def load_sample_signals(csv_path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    frame = _ensure_required(frame)
    return frame.copy()


def fetch_live_signals(limit: int = 300) -> pd.DataFrame:
    try:
        import akshare as ak
    except Exception as exc:
        raise RuntimeError("未安装 akshare，无法自动采集数据。请先执行 pip install -r requirements.txt") from exc

    spot = _fetch_spot_with_proxy_strategy(ak)
    if spot is None or spot.empty:
        raise RuntimeError("未获取到实时行情数据。")

    colmap: Dict[str, str | None] = {
        "symbol": _pick_col(spot.columns, ["代码"]),
        "name": _pick_col(spot.columns, ["名称"]),
        "industry": _pick_col(spot.columns, ["所属行业", "行业"]),
        "pct_chg": _pick_col(spot.columns, ["涨跌幅"]),
        "turnover": _pick_col(spot.columns, ["换手率"]),
        "amount": _pick_col(spot.columns, ["成交额"]),
        "volume_ratio": _pick_col(spot.columns, ["量比"]),
        "amplitude": _pick_col(spot.columns, ["振幅"]),
    }
    required = ["symbol", "name", "pct_chg", "turnover", "amount", "volume_ratio", "amplitude"]
    missing_basic = [k for k in required if not colmap[k]]
    if missing_basic:
        raise RuntimeError(f"行情字段不完整，缺少: {missing_basic}")

    data = pd.DataFrame()
    data["symbol"] = spot[colmap["symbol"]].astype(str)
    data["name"] = spot[colmap["name"]].astype(str)
    if colmap["industry"]:
        data["theme"] = spot[colmap["industry"]].astype(str)
    else:
        data["theme"] = "未知行业"

    for out, src in [
        ("pct_chg", "pct_chg"),
        ("turnover", "turnover"),
        ("amount", "amount"),
        ("volume_ratio", "volume_ratio"),
        ("amplitude", "amplitude"),
    ]:
        data[out] = pd.to_numeric(spot[colmap[src]], errors="coerce").fillna(0.0)

    data = data[~data["name"].str.contains("ST", na=False)].copy()
    data = data[data["amount"] > 0].copy()

    industry_stats = data.groupby("theme").agg(
        industry_mean_pct=("pct_chg", "mean"),
        industry_up_ratio=("pct_chg", lambda s: (s > 0).mean() * 100),
        industry_amount=("amount", "sum"),
    )
    data = data.join(industry_stats, on="theme")

    data["theme_strength"] = _normalize_0_100(
        data["industry_mean_pct"] * 0.6 + np.log1p(data["industry_amount"]) * 0.4
    )
    data["sector_linkage"] = _normalize_0_100(
        data["industry_mean_pct"] * 0.5 + data["industry_up_ratio"] * 0.5
    )
    data["stock_strength"] = _normalize_0_100(
        data["pct_chg"] * 0.5 + data["volume_ratio"] * 22 + data["amplitude"] * 0.2
    )
    data["capital_support"] = _normalize_0_100(
        data["turnover"] * 0.35 + np.log1p(data["amount"]) * 6 + data["volume_ratio"] * 35
    )
    data["risk_tag"] = _build_risk_tag(data["amplitude"], data["turnover"])
    data["snapshot_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cols = REQUIRED_COLS + [
        "pct_chg",
        "turnover",
        "amount",
        "volume_ratio",
        "amplitude",
        "industry_mean_pct",
        "industry_up_ratio",
        "snapshot_time",
    ]
    result = data[cols].copy()
    result = result.sort_values(by="stock_strength", ascending=False).head(limit).reset_index(drop=True)
    return result


def _fetch_spot_with_proxy_strategy(ak_module) -> pd.DataFrame:
    # 先使用系统代理；若失败再尝试禁用代理重试（解决常见 ProxyError）
    first_error: Exception | None = None
    try:
        return ak_module.stock_zh_a_spot_em()
    except Exception as exc:  # noqa: BLE001
        first_error = exc

    proxy_keys = ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]
    backup = {k: os.environ.get(k) for k in proxy_keys}
    try:
        for key in proxy_keys:
            os.environ.pop(key, None)
        return ak_module.stock_zh_a_spot_em()
    except Exception as second_error:  # noqa: BLE001
        raise RuntimeError(
            "实时采集失败：系统代理与直连都未成功。"
            f"\n代理模式错误: {first_error}"
            f"\n直连模式错误: {second_error}"
        ) from second_error
    finally:
        for key, val in backup.items():
            if val is not None:
                os.environ[key] = val


def to_signal_objects(frame: pd.DataFrame) -> List[StockSignal]:
    frame = _ensure_required(frame)
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

