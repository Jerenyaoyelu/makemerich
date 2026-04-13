from __future__ import annotations

from datetime import datetime
import os
import re
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

    spot, source = _fetch_spot_multi_source(ak)
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
        "high": _pick_col(spot.columns, ["最高"]),
        "low": _pick_col(spot.columns, ["最低"]),
        "pre_close": _pick_col(spot.columns, ["昨收"]),
    }
    required = ["symbol", "name", "pct_chg", "amount"]
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

    data["pct_chg"] = pd.to_numeric(spot[colmap["pct_chg"]], errors="coerce").fillna(0.0)
    data["amount"] = pd.to_numeric(spot[colmap["amount"]], errors="coerce").fillna(0.0)

    if colmap["turnover"]:
        data["turnover"] = pd.to_numeric(spot[colmap["turnover"]], errors="coerce")
    else:
        data["turnover"] = np.nan

    if colmap["volume_ratio"]:
        data["volume_ratio"] = pd.to_numeric(spot[colmap["volume_ratio"]], errors="coerce")
    else:
        # 新浪等源不提供官方量比；导出为缺失，打分见下面对 neutral 的处理
        data["volume_ratio"] = np.nan

    if colmap["amplitude"]:
        data["amplitude"] = pd.to_numeric(spot[colmap["amplitude"]], errors="coerce").fillna(0.0)
    elif colmap["high"] and colmap["low"] and colmap["pre_close"]:
        high = pd.to_numeric(spot[colmap["high"]], errors="coerce").fillna(0.0)
        low = pd.to_numeric(spot[colmap["low"]], errors="coerce").fillna(0.0)
        pre_close = pd.to_numeric(spot[colmap["pre_close"]], errors="coerce").replace(0, np.nan)
        data["amplitude"] = ((high - low).abs() / pre_close * 100).fillna(0.0)
    else:
        data["amplitude"] = 0.0

    data = data[~data["name"].str.contains("ST", na=False)].copy()
    data = data[data["amount"] > 0].copy()

    industry_stats = data.groupby("theme").agg(
        industry_mean_pct=("pct_chg", "mean"),
        industry_up_ratio=("pct_chg", lambda s: (s > 0).mean() * 100),
        industry_amount=("amount", "sum"),
    )
    data = data.join(industry_stats, on="theme")

    # 无量比字段时用 1.0 作中性占位，避免整表相同常数扭曲排序；有真实量比时不受影响
    vol_for_score = data["volume_ratio"].fillna(1.0)
    turn_for_risk = data["turnover"].fillna(0.0)

    data["theme_strength"] = _normalize_0_100(
        data["industry_mean_pct"] * 0.6 + np.log1p(data["industry_amount"]) * 0.4
    )
    data["sector_linkage"] = _normalize_0_100(
        data["industry_mean_pct"] * 0.5 + data["industry_up_ratio"] * 0.5
    )
    data["stock_strength"] = _normalize_0_100(
        data["pct_chg"] * 0.5 + vol_for_score * 22 + data["amplitude"] * 0.2
    )
    data["capital_support"] = _normalize_0_100(
        data["turnover"].fillna(0.0) * 0.35 + np.log1p(data["amount"]) * 6 + vol_for_score * 35
    )
    data["risk_tag"] = _build_risk_tag(data["amplitude"], turn_for_risk)
    data["snapshot_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data["source"] = source

    cols = REQUIRED_COLS + [
        "pct_chg",
        "turnover",
        "amount",
        "volume_ratio",
        "amplitude",
        "industry_mean_pct",
        "industry_up_ratio",
        "snapshot_time",
        "source",
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


def _fetch_spot_multi_source(ak_module) -> tuple[pd.DataFrame, str]:
    primary_error: Exception | None = None
    fallback_error: Exception | None = None
    try:
        return _fetch_spot_with_proxy_strategy(ak_module), "eastmoney_em"
    except Exception as exc:  # noqa: BLE001
        primary_error = exc

    try:
        # akshare 的 stock_zh_a_spot会丢弃新浪返回的 turnoverratio，导致换手恒为 0
        return _fetch_sina_zh_a_spot_with_turnover(), "sina_finance"
    except Exception as exc:  # noqa: BLE001
        fallback_error = exc

    raise RuntimeError(
        "实时采集失败：主源与备源均不可用。"
        f"\n主源 stock_zh_a_spot_em 错误: {primary_error}"
        f"\n备源新浪财经错误: {fallback_error}"
    ) from fallback_error


def _get_sina_a_page_count() -> int:
    import requests
    from akshare.stock.cons import zh_sina_a_stock_count_url

    res = requests.get(zh_sina_a_stock_count_url, timeout=20)
    page_count = int(re.findall(re.compile(r"\d+"), res.text)[0]) / 80
    if isinstance(page_count, int):
        return page_count
    return int(page_count) + 1


def _fetch_sina_zh_a_spot_with_turnover() -> pd.DataFrame:
    """新浪财经 A 股全市场快照（与 akshare.stock_zh_a_spot 同源分页），保留换手率；不含量比。"""
    import requests
    from akshare.stock.cons import zh_sina_a_stock_payload, zh_sina_a_stock_url
    from akshare.utils import demjson
    from akshare.utils.tqdm import get_tqdm

    page_count = _get_sina_a_page_count()
    big_df = pd.DataFrame()
    zh_sina_stock_payload_copy = zh_sina_a_stock_payload.copy()
    tqdm = get_tqdm()
    for page in tqdm(range(1, page_count + 1), leave=False, desc="Please wait for a moment"):
        zh_sina_stock_payload_copy.update({"page": page})
        r = requests.get(zh_sina_a_stock_url, params=zh_sina_stock_payload_copy, timeout=45)
        data_json = demjson.decode(r.text)
        big_df = pd.concat(objs=[big_df, pd.DataFrame(data_json)], ignore_index=True)

    big_df = big_df.astype(
        {
            "trade": "float",
            "pricechange": "float",
            "changepercent": "float",
            "buy": "float",
            "sell": "float",
            "settlement": "float",
            "open": "float",
            "high": "float",
            "low": "float",
            "volume": "float",
            "amount": "float",
            "per": "float",
            "pb": "float",
            "mktcap": "float",
            "nmc": "float",
            "turnoverratio": "float",
        }
    )
    out = pd.DataFrame(
        {
            "代码": big_df["symbol"].astype(str),
            "名称": big_df["name"].astype(str),
            "涨跌幅": big_df["changepercent"],
            "成交额": big_df["amount"],
            "换手率": big_df["turnoverratio"],
            "最高": big_df["high"],
            "最低": big_df["low"],
            "昨收": big_df["settlement"],
        }
    )
    return out


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

