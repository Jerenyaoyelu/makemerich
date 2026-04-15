from __future__ import annotations

from datetime import datetime
import os
import random
import re
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, TypeVar


import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from core.logger import get_logger
from core.models import StockSignal
from core.run_store import resolve_auto_spot_preference

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
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INDUSTRY_MAP_PATH = DATA_DIR / "industry_map.csv"
INDUSTRY_MAP_STATUS_PATH = DATA_DIR / "industry_map_status.txt"
ProgressCallback = Callable[[int, str], None]
_T = TypeVar("_T")
logger = get_logger("data_provider")

_PROXY_ENV_KEYS = [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
]


def _ak_call_with_retries(
    label: str,
    fn: Callable[[], _T],
    *,
    max_attempts: int = 5,
    base_sleep: float = 0.75,
    clear_proxy_after_attempt: int = 2,
) -> _T:
    """
    AkShare 底层走 HTTP，批量请求易被服务端掐断（RemoteDisconnected）。
    策略：指数退避 + 若干次失败后临时清空系统代理再试（与实时东财逻辑一致）。
    """
    last_exc: BaseException | None = None
    proxy_backup: dict[str, str | None] = {}
    proxy_cleared = False
    try:
        for attempt in range(max_attempts):
            try:
                if attempt > 0:
                    logger.info("%s: 第 %s 次重试", label, attempt + 1)
                return fn()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning("%s 失败（attempt=%s/%s）: %s", label, attempt + 1, max_attempts, repr(exc))
                if attempt >= clear_proxy_after_attempt and not proxy_cleared:
                    proxy_backup = {k: os.environ.get(k) for k in _PROXY_ENV_KEYS}
                    for k in _PROXY_ENV_KEYS:
                        os.environ.pop(k, None)
                    proxy_cleared = True
                    logger.info("%s: 已临时清除代理环境变量后继续重试", label)
                jitter = random.uniform(0.05, 0.35)
                time.sleep(base_sleep * (1.55**attempt) + jitter)
        msg = (
            f"{label} 在 {max_attempts} 次重试后仍失败。"
            "常见原因：网络不稳定、代理干扰、或东财/新浪限流。可尝试关闭系统代理、减小历史扫描上限、稍后重试。"
        )
        logger.error("%s", msg)
        raise RuntimeError(msg) from last_exc
    finally:
        if proxy_cleared and proxy_backup:
            for key, val in proxy_backup.items():
                if val is not None:
                    os.environ[key] = val
            logger.info("%s: 已恢复代理环境变量", label)


try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except Exception:
    pass


def _pick_col(columns: Iterable[str], candidates: List[str]) -> str | None:
    for c in candidates:
        if c in columns:
            return c
    return None


def _normalize_symbol_code(symbol: object) -> str:
    s = str(symbol).strip().lower()
    if s.startswith(("sh", "sz", "bj")):
        s = s[2:]
    return s


def _format_symbol_for_hist(symbol: str) -> str:
    """与 `core.evaluation.format_symbol_for_ak` 一致，供 AkShare 日线接口使用。"""
    s = str(symbol).strip().zfill(6)
    if s.startswith(("6", "9")):
        return f"sh{s}"
    if s.startswith(("8", "4")):
        return f"bj{s}"
    return f"sz{s}"


def _symbol_in_market_scope(symbol_norm: str, market_scope: str) -> bool:
    s = str(symbol_norm)
    if market_scope == "all_a":
        return True
    if market_scope == "hs_main":
        return bool(re.match(r"^(600|601|603|605|000|001|002)\d{3}$", s))
    if market_scope == "gem":
        return bool(re.match(r"^(300|301)\d{3}$", s))
    if market_scope == "star":
        return bool(re.match(r"^688\d{3}$", s))
    return True


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


def fetch_live_signals(
    limit: int = 300,
    progress_callback: ProgressCallback | None = None,
    market_scope: str = "all_a",
    primary_source_strategy: str = "stability_first",
) -> pd.DataFrame:
    eff = (
        resolve_auto_spot_preference()
        if primary_source_strategy == "auto"
        else primary_source_strategy
    )
    if eff not in ("completeness_first", "stability_first"):
        eff = "stability_first"
    logger.info(
        "开始实时采集: limit=%s, market_scope=%s, primary_strategy=%s (effective=%s)",
        limit,
        market_scope,
        primary_source_strategy,
        eff,
    )
    try:
        import akshare as ak
    except Exception as exc:
        raise RuntimeError("未安装 akshare，无法自动采集数据。请先执行 pip install -r requirements.txt") from exc

    if progress_callback:
        progress_callback(5, "准备连接数据源...")
    spot, source = _fetch_spot_multi_source(
        ak, progress_callback=progress_callback, primary_strategy=eff
    )
    if spot is None or spot.empty:
        raise RuntimeError("未获取到实时行情数据。")
    if progress_callback:
        progress_callback(40, f"数据源 {source} 已返回，正在清洗与补齐字段...")
    logger.info("实时采集源=%s, rows=%s", source, 0 if spot is None else len(spot))

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
    data["symbol_norm"] = data["symbol"].map(_normalize_symbol_code)
    data["name"] = spot[colmap["name"]].astype(str)
    data["theme_source"] = "spot"
    if colmap["industry"]:
        data["theme"] = spot[colmap["industry"]].astype(str)
    else:
        data["theme"] = "未知行业"
        data["theme_source"] = "unknown"

    data["pct_chg"] = pd.to_numeric(spot[colmap["pct_chg"]], errors="coerce").fillna(0.0)
    data["amount"] = pd.to_numeric(spot[colmap["amount"]], errors="coerce").fillna(0.0)

    if colmap["turnover"]:
        data["turnover"] = pd.to_numeric(spot[colmap["turnover"]], errors="coerce")
        data["turnover_source"] = "spot"
    else:
        data["turnover"] = np.nan
        data["turnover_source"] = "missing"

    if colmap["volume_ratio"]:
        data["volume_ratio"] = pd.to_numeric(spot[colmap["volume_ratio"]], errors="coerce")
        data["volume_ratio_source"] = "spot"
    else:
        # 新浪等源不提供官方量比；导出为缺失，打分层按可用性降权处理
        data["volume_ratio"] = np.nan
        data["volume_ratio_source"] = "missing"

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
    data = _filter_by_market_scope(data, market_scope)
    if data.empty:
        raise RuntimeError("按当前采集范围过滤后无可用股票，请调整采集范围。")

    nmc_col = _pick_col(spot.columns, ["流通市值", "nmc"])
    if nmc_col:
        nmc_series = pd.to_numeric(spot[nmc_col], errors="coerce").replace(0, np.nan)
        data["float_mktcap"] = nmc_series.reindex(data.index)
    else:
        data["float_mktcap"] = np.nan

    result = _finalize_signals_from_base_data(
        data,
        source=source,
        snapshot_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        limit=limit,
        ak_module=ak,
        progress_callback=progress_callback,
    )
    logger.info("实时采集完成: source=%s, result_rows=%s", source, len(result))
    return result


def _finalize_signals_from_base_data(
    data: pd.DataFrame,
    *,
    source: str,
    snapshot_time: str,
    limit: int,
    ak_module,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    """由「已清洗的当日行情宽表」计算行业聚合、四因子与输出列（实时/历史共用）。"""
    industry_map = _load_or_refresh_industry_map(ak_module)
    if not industry_map.empty:
        missing_theme_mask = (data["theme"].isin(["未知行业", "", "None"])) | data["theme"].isna()
        filled_theme = data.loc[missing_theme_mask, "symbol_norm"].map(industry_map)
        fill_ok = filled_theme.notna()
        if fill_ok.any():
            idx = filled_theme[fill_ok].index
            data.loc[idx, "theme"] = filled_theme.loc[idx]
            data.loc[idx, "theme_source"] = "industry_map_cache"

    if "float_mktcap" not in data.columns:
        data["float_mktcap"] = np.nan
    need_turnover_est = data["turnover"].isna() | (data["turnover"] <= 0)
    est_turnover = (data["amount"] / data["float_mktcap"] * 100).replace([np.inf, -np.inf], np.nan)
    est_ok = need_turnover_est & est_turnover.notna() & data["float_mktcap"].notna()
    if est_ok.any():
        data.loc[est_ok, "turnover"] = est_turnover.loc[est_ok]
        data.loc[est_ok, "turnover_source"] = "estimated_by_amount_nmc"

    data.loc[data["volume_ratio"].isna(), "volume_ratio_source"] = "missing"

    industry_stats = data.groupby("theme").agg(
        industry_mean_pct=("pct_chg", "mean"),
        industry_up_ratio=("pct_chg", lambda s: (s > 0).mean() * 100),
        industry_amount=("amount", "sum"),
    )
    data = data.join(industry_stats, on="theme")

    vol_for_score = data["volume_ratio"].fillna(1.0)
    vol_available = data["volume_ratio"].notna().astype(float)
    vol_coef_stock = 22 * (0.3 + 0.7 * vol_available)
    vol_coef_capital = 35 * (0.3 + 0.7 * vol_available)
    turn_for_risk = data["turnover"].fillna(0.0)

    data["theme_strength"] = _normalize_0_100(
        data["industry_mean_pct"] * 0.6 + np.log1p(data["industry_amount"]) * 0.4
    )
    data["sector_linkage"] = _normalize_0_100(
        data["industry_mean_pct"] * 0.5 + data["industry_up_ratio"] * 0.5
    )
    data["stock_strength"] = _normalize_0_100(
        data["pct_chg"] * 0.5 + vol_for_score * vol_coef_stock + data["amplitude"] * 0.2
    )
    data["capital_support"] = _normalize_0_100(
        data["turnover"].fillna(0.0) * 0.35 + np.log1p(data["amount"]) * 6 + vol_for_score * vol_coef_capital
    )
    data["risk_tag"] = _build_risk_tag(data["amplitude"], turn_for_risk)
    data["snapshot_time"] = snapshot_time
    data["source"] = source
    if progress_callback:
        progress_callback(85, "正在计算评分因子...")

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
        "theme_source",
        "turnover_source",
        "volume_ratio_source",
    ]
    result = data[cols].copy()
    result = result.sort_values(by="stock_strength", ascending=False).head(limit).reset_index(drop=True)
    if progress_callback:
        progress_callback(95, "数据处理完成，准备返回结果...")
    return result


def _extract_hist_row_for_trade_date(
    hist: pd.DataFrame,
    *,
    target_date_ymd: str,
) -> pd.Series | None:
    """从任意源返回的日线表中抽取目标交易日一行。"""
    if hist is None or hist.empty:
        return None
    date_col = _pick_col(hist.columns, ["日期", "date", "Date", "交易日期"])
    if date_col:
        d = pd.to_datetime(hist[date_col], errors="coerce").dt.strftime("%Y%m%d")
        hit = hist[d == target_date_ymd]
        if hit.empty:
            return None
        return hit.iloc[-1]
    # 无日期列时退化为最后一行（极少见）
    return hist.iloc[-1]


def _pick_first_numeric(row: pd.Series, candidates: list[str]) -> float | None:
    for c in candidates:
        if c in row.index:
            v = pd.to_numeric(row.get(c), errors="coerce")
            if pd.notna(v):
                return float(v)
    return None


def _hist_try_em(
    ak_module,
    symbol_prefixed: str,
    target_date_ymd: str,
    request_delay_sec: float,
) -> tuple[dict | None, str | None]:
    try:
        hist_em = _ak_call_with_retries(
            f"日线(东财) {symbol_prefixed} {target_date_ymd}",
            lambda s=symbol_prefixed: ak_module.stock_zh_a_hist(
                symbol=s,
                period="daily",
                start_date=target_date_ymd,
                end_date=target_date_ymd,
                adjust="qfq",
            ),
            max_attempts=4,
            base_sleep=0.55,
            clear_proxy_after_attempt=1,
        )
        if request_delay_sec > 0:
            time.sleep(request_delay_sec)
        row = _extract_hist_row_for_trade_date(hist_em, target_date_ymd=target_date_ymd)
        if row is not None:
            return {
                "pct_chg": _pick_first_numeric(row, ["涨跌幅"]),
                "amount": _pick_first_numeric(row, ["成交额"]),
                "turnover": _pick_first_numeric(row, ["换手率"]),
                "amplitude": _pick_first_numeric(row, ["振幅"]),
            }, "em"
    except RuntimeError:
        pass
    return None, None


def _hist_try_sina(
    ak_module,
    symbol_prefixed: str,
    target_date_ymd: str,
    request_delay_sec: float,
) -> tuple[dict | None, str | None]:
    try:
        hist_sina = _ak_call_with_retries(
            f"日线(新浪) {symbol_prefixed} {target_date_ymd}",
            lambda s=symbol_prefixed: ak_module.stock_zh_a_daily(
                symbol=s,
                start_date=target_date_ymd,
                end_date=target_date_ymd,
                adjust="qfq",
            ),
            max_attempts=3,
            base_sleep=0.6,
            clear_proxy_after_attempt=1,
        )
        if request_delay_sec > 0:
            time.sleep(request_delay_sec)
        row = _extract_hist_row_for_trade_date(hist_sina, target_date_ymd=target_date_ymd)
        if row is not None:
            close_v = _pick_first_numeric(row, ["close", "收盘"])
            open_v = _pick_first_numeric(row, ["open", "开盘"])
            high_v = _pick_first_numeric(row, ["high", "最高"])
            low_v = _pick_first_numeric(row, ["low", "最低"])
            pct_v = _pick_first_numeric(row, ["涨跌幅", "changepercent"])
            if pct_v is None and close_v is not None and open_v is not None and open_v != 0:
                pct_v = (close_v / open_v - 1.0) * 100.0
            amp_v = _pick_first_numeric(row, ["振幅", "amplitude"])
            if amp_v is None and high_v is not None and low_v is not None and close_v and close_v != 0:
                amp_v = abs(high_v - low_v) / close_v * 100.0
            return {
                "pct_chg": pct_v,
                "amount": _pick_first_numeric(row, ["amount", "成交额"]),
                "turnover": _pick_first_numeric(row, ["turnover", "换手率"]),
                "amplitude": amp_v,
            }, "sina"
    except RuntimeError:
        pass
    return None, None


def _hist_try_tx(
    ak_module,
    symbol_prefixed: str,
    target_date_ymd: str,
    request_delay_sec: float,
) -> tuple[dict | None, str | None]:
    target_date_iso = f"{target_date_ymd[:4]}-{target_date_ymd[4:6]}-{target_date_ymd[6:8]}"
    try:
        hist_tx = _ak_call_with_retries(
            f"日线(腾讯) {symbol_prefixed} {target_date_ymd}",
            lambda s=symbol_prefixed: ak_module.stock_zh_a_hist_tx(
                symbol=s,
                start_date=target_date_iso,
                end_date=target_date_iso,
                adjust="qfq",
            ),
            max_attempts=3,
            base_sleep=0.65,
            clear_proxy_after_attempt=1,
        )
        if request_delay_sec > 0:
            time.sleep(request_delay_sec)
        row = _extract_hist_row_for_trade_date(hist_tx, target_date_ymd=target_date_ymd)
        if row is not None:
            close_v = _pick_first_numeric(row, ["收盘", "close"])
            open_v = _pick_first_numeric(row, ["开盘", "open"])
            high_v = _pick_first_numeric(row, ["最高", "high"])
            low_v = _pick_first_numeric(row, ["最低", "low"])
            pct_v = _pick_first_numeric(row, ["涨跌幅"])
            if pct_v is None and close_v is not None and open_v is not None and open_v != 0:
                pct_v = (close_v / open_v - 1.0) * 100.0
            amp_v = _pick_first_numeric(row, ["振幅"])
            if amp_v is None and high_v is not None and low_v is not None and close_v and close_v != 0:
                amp_v = abs(high_v - low_v) / close_v * 100.0
            return {
                "pct_chg": pct_v,
                "amount": _pick_first_numeric(row, ["成交额", "amount"]),
                "turnover": _pick_first_numeric(row, ["换手率", "turnover"]),
                "amplitude": amp_v,
            }, "tx"
    except RuntimeError:
        pass
    return None, None


def _fetch_hist_row_multi_source(
    ak_module,
    *,
    code6: str,
    symbol_prefixed: str,
    target_date_ymd: str,
    request_delay_sec: float,
    primary_strategy: str = "stability_first",
) -> tuple[dict | None, str | None]:
    """
    历史单票多源，顺序由 primary_strategy 决定：
    - stability_first: 新浪 -> 腾讯 -> 东财
    - completeness_first: 东财 -> 新浪 -> 腾讯
    """
    if primary_strategy == "completeness_first":
        order = ("em", "sina", "tx")
    else:
        order = ("sina", "tx", "em")

    for src in order:
        if src == "em":
            d, tag = _hist_try_em(ak_module, symbol_prefixed, target_date_ymd, request_delay_sec)
        elif src == "sina":
            d, tag = _hist_try_sina(ak_module, symbol_prefixed, target_date_ymd, request_delay_sec)
        else:
            d, tag = _hist_try_tx(ak_module, symbol_prefixed, target_date_ymd, request_delay_sec)
        if d is not None:
            return d, tag
    return None, None


def fetch_historical_signals(
    trade_date: str,
    limit: int = 300,
    progress_callback: ProgressCallback | None = None,
    market_scope: str = "all_a",
    max_universe_scan: int = 3500,
    request_delay_sec: float = 0.08,
    primary_source_strategy: str = "stability_first",
) -> pd.DataFrame:
    """
    按**某一交易日**的 AkShare 后复权日线，重建当日全市场截面并走与实时相同的因子流水线。
    `snapshot_time` 固定为该日 15:00:00，便于复评脚本以该日为 T0 计算 T+1、T+2…

    说明：日线不含「量比」，与新浪备源一致按缺失处理；行业依赖本地映射缓存回填。
    扫描上限用于控制请求条数（逐票 `stock_zh_a_hist`），过大较慢。
    `request_delay_sec`：每笔日线请求后的间隔，降低被服务端掐断（RemoteDisconnected）的概率。
    `primary_source_strategy`：与实时采集相同，`auto` 依据近期成功采集记录选择先试东财或新浪。
    """
    eff = (
        resolve_auto_spot_preference()
        if primary_source_strategy == "auto"
        else primary_source_strategy
    )
    if eff not in ("completeness_first", "stability_first"):
        eff = "stability_first"
    logger.info(
        "开始历史截面采集: trade_date=%s, scope=%s, limit=%s, scan_cap=%s, delay=%.2f, primary=%s (effective=%s)",
        trade_date,
        market_scope,
        limit,
        max_universe_scan,
        request_delay_sec,
        primary_source_strategy,
        eff,
    )
    try:
        import akshare as ak
    except Exception as exc:
        raise RuntimeError("未安装 akshare，无法拉取历史数据。请先执行 pip install -r requirements.txt") from exc

    d = pd.to_datetime(trade_date).strftime("%Y-%m-%d")
    ymd = d.replace("-", "")
    if progress_callback:
        progress_callback(
            3,
            f"准备拉取历史截面：{d}（逐票日线，已启用重试/节流；若仍失败请关代理或减小扫描上限）…",
        )

    info = _ak_call_with_retries(
        "获取 A 股代码表 stock_info_a_code_name",
        lambda: ak.stock_info_a_code_name(),
        max_attempts=6,
        base_sleep=1.0,
        clear_proxy_after_attempt=2,
    )
    if info is None or info.empty:
        raise RuntimeError("无法获取 A 股代码表 stock_info_a_code_name。")
    logger.info("历史采集代码表获取成功: total_codes=%s", len(info))

    code_col = _pick_col(info.columns, ["code", "代码"])
    name_col = _pick_col(info.columns, ["name", "名称"])
    if not code_col or not name_col:
        raise RuntimeError(f"A 股代码表缺少 code/name 列: {list(info.columns)}")

    rows: List[dict] = []
    source_hit = {"em": 0, "sina": 0, "tx": 0, "none": 0}
    scanned = 0
    for _, info_row in info.iterrows():
        code_raw = str(info_row[code_col]).strip()
        if not code_raw.replace(".", "").isdigit():
            continue
        code = code_raw.split(".")[0].zfill(6)
        name = str(info_row[name_col])
        if "ST" in name.upper():
            continue
        sym_norm = _normalize_symbol_code(code)
        if not _symbol_in_market_scope(sym_norm, market_scope):
            continue
        if scanned >= max_universe_scan:
            break
        scanned += 1

        sym = _format_symbol_for_hist(code)
        unified, hit_source = _fetch_hist_row_multi_source(
            ak,
            code6=code,
            symbol_prefixed=sym,
            target_date_ymd=ymd,
            request_delay_sec=request_delay_sec,
            primary_strategy=eff,
        )
        if unified is None:
            source_hit["none"] += 1
            continue
        source_hit[hit_source or "none"] += 1

        pct = unified.get("pct_chg")
        amt = unified.get("amount")
        if amt is None or pd.isna(amt) or float(amt) <= 0:
            continue
        turn = unified.get("turnover")
        amp = unified.get("amplitude")
        if amp is None or pd.isna(amp):
            amp = 0.0
        if pct is None or pd.isna(pct):
            pct = 0.0

        ts = "spot" if turn is not None and pd.notna(turn) and float(turn) > 0 else "missing"
        rows.append(
            {
                "symbol": code,
                "symbol_norm": sym_norm,
                "name": name,
                "theme": "未知行业",
                "theme_source": "unknown",
                "pct_chg": float(pct),
                "amount": float(amt),
                "turnover": float(turn) if turn is not None and pd.notna(turn) else np.nan,
                "turnover_source": ts,
                "volume_ratio": np.nan,
                "volume_ratio_source": "missing",
                "amplitude": float(amp),
            }
        )

        if progress_callback and scanned % 20 == 0:
            p = min(75, 5 + int(70 * scanned / max(1, max_universe_scan)))
            progress_callback(
                p,
                f"历史日线请求进度 {scanned}/{max_universe_scan}，已入库 {len(rows)} 只有效标的…",
            )
            logger.info(
                "历史采集进度: scanned=%s/%s, collected=%s, hits=%s",
                scanned,
                max_universe_scan,
                len(rows),
                source_hit,
            )

    if not rows:
        raise RuntimeError(
            f"交易日 {d} 未采到有效日线数据。请确认当日为交易日、网络正常，或提高「历史扫描上限」。"
        )
    logger.info("历史采集原始数据完成: collected_rows=%s, source_hits=%s", len(rows), source_hit)

    data = pd.DataFrame(rows)
    data["float_mktcap"] = np.nan
    snap = f"{d} 15:00:00"
    result = _finalize_signals_from_base_data(
        data,
        source="historical_daily_multi",
        snapshot_time=snap,
        limit=limit,
        ak_module=ak,
        progress_callback=progress_callback,
    )
    logger.info("历史截面构建完成: trade_date=%s, result_rows=%s", d, len(result))
    return result


def _filter_by_market_scope(data: pd.DataFrame, market_scope: str) -> pd.DataFrame:
    if "symbol_norm" not in data.columns:
        return data
    s = data["symbol_norm"].astype(str)
    if market_scope == "all_a":
        return data
    if market_scope == "hs_main":
        mask = s.str.match(r"^(600|601|603|605|000|001|002)\d{3}$")
        return data[mask].copy()
    if market_scope == "gem":
        mask = s.str.match(r"^(300|301)\d{3}$")
        return data[mask].copy()
    if market_scope == "star":
        mask = s.str.match(r"^688\d{3}$")
        return data[mask].copy()
    return data


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


def _fetch_spot_multi_source(
    ak_module,
    progress_callback: ProgressCallback | None = None,
    *,
    primary_strategy: str = "stability_first",
) -> tuple[pd.DataFrame, str]:
    """
    primary_strategy:
    - stability_first: 新浪分页 -> 东财
    - completeness_first: 东财 -> 新浪分页
    """
    first_error: Exception | None = None
    second_error: Exception | None = None

    def _try_sina() -> tuple[pd.DataFrame, str]:
        if progress_callback:
            progress_callback(12, "尝试主源：新浪财经（分页采集）...")
        return _fetch_sina_zh_a_spot_with_turnover(progress_callback=progress_callback), "sina_finance"

    def _try_em() -> tuple[pd.DataFrame, str]:
        if progress_callback:
            progress_callback(12, "尝试主源：东方财富...")
        return _fetch_spot_with_proxy_strategy(ak_module), "eastmoney_em"

    if primary_strategy == "completeness_first":
        try:
            return _try_em()
        except Exception as exc:  # noqa: BLE001
            first_error = exc
        try:
            if progress_callback:
                progress_callback(18, "东财失败，切换备源：新浪财经（分页采集）...")
            return _try_sina()
        except Exception as exc:  # noqa: BLE001
            second_error = exc
    else:
        try:
            return _try_sina()
        except Exception as exc:  # noqa: BLE001
            first_error = exc
        try:
            if progress_callback:
                progress_callback(18, "新浪失败，切换备源：东方财富...")
            return _try_em()
        except Exception as exc:  # noqa: BLE001
            second_error = exc

    raise RuntimeError(
        "实时采集失败：两个行情源均不可用。"
        f"\n第一次尝试错误: {first_error}"
        f"\n第二次尝试错误: {second_error}"
    ) from second_error


def _get_sina_a_page_count() -> int:
    import requests
    from akshare.stock.cons import zh_sina_a_stock_count_url

    res = requests.get(zh_sina_a_stock_count_url, timeout=20)
    page_count = int(re.findall(re.compile(r"\d+"), res.text)[0]) / 80
    if isinstance(page_count, int):
        return page_count
    return int(page_count) + 1


def _fetch_sina_zh_a_spot_with_turnover(progress_callback: ProgressCallback | None = None) -> pd.DataFrame:
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
        if progress_callback:
            # 新浪分页长任务映射到 20~38 区间，给 UI 提供细粒度进度
            p = int(20 + (page / page_count) * 18)
            progress_callback(min(p, 38), f"新浪分页采集中... {page}/{page_count}")

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
            "nmc": big_df["nmc"],
            "最高": big_df["high"],
            "最低": big_df["low"],
            "昨收": big_df["settlement"],
        }
    )
    return out


def _load_or_refresh_industry_map(ak_module) -> pd.Series:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if INDUSTRY_MAP_PATH.exists():
        cached = pd.read_csv(INDUSTRY_MAP_PATH, dtype={"symbol": str, "theme": str})
        cached = cached.dropna(subset=["symbol", "theme"])
        if not cached.empty:
            cached["symbol"] = cached["symbol"].map(_normalize_symbol_code)
            _write_industry_map_status("using_cached_industry_map")
            return cached.drop_duplicates("symbol").set_index("symbol")["theme"]
        _write_industry_map_status("cached_industry_map_exists_but_empty")
    try:
        industry_map = _build_industry_map_from_ak(ak_module)
        if not industry_map.empty:
            df = industry_map.reset_index()
            df.columns = ["symbol", "theme"]
            df.to_csv(INDUSTRY_MAP_PATH, index=False, encoding="utf-8-sig")
            _write_industry_map_status(f"built_industry_map_success rows={len(df)}")
            return industry_map
        _write_industry_map_status("build_industry_map_returned_empty")
    except Exception as exc:
        _write_industry_map_status(f"build_industry_map_exception: {repr(exc)}")
    # 返回空映射（上层按未知行业处理）
    return pd.Series(dtype=str)


def _build_industry_map_from_ak(ak_module) -> pd.Series:
    # 优先使用 tushare 提供的股票->行业映射
    ts_map = _build_industry_map_from_tushare()
    if not ts_map.empty:
        return ts_map
    # tushare 不可用时，回退同花顺行业链路
    ths_map = _build_industry_map_from_ths(ak_module)
    if not ths_map.empty:
        return ths_map
    return pd.Series(dtype=str)


def _build_industry_map_from_tushare() -> pd.Series:
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        _write_industry_map_status("tushare_token_missing")
        return pd.Series(dtype=str)
    try:
        import tushare as ts
    except Exception:
        _write_industry_map_status("tushare_not_installed")
        return pd.Series(dtype=str)

    try:
        pro = ts.pro_api(token)
        df = pro.stock_basic(
            exchange="",
            list_status="L",
            fields="symbol,industry",
        )
    except Exception as exc:
        _write_industry_map_status(f"tushare_stock_basic_failed: {repr(exc)}")
        return pd.Series(dtype=str)

    if df is None or df.empty:
        _write_industry_map_status("tushare_stock_basic_empty")
        return pd.Series(dtype=str)

    df = df.rename(columns={"symbol": "symbol", "industry": "theme"})
    df["symbol"] = df["symbol"].astype(str).map(_normalize_symbol_code)
    df["theme"] = df["theme"].astype(str).str.strip()
    df = df[df["theme"].notna() & (df["theme"] != "") & (df["theme"].str.lower() != "nan")]
    if df.empty:
        _write_industry_map_status("tushare_stock_basic_no_valid_industry")
        return pd.Series(dtype=str)
    _write_industry_map_status(f"tushare_stock_basic_ok rows={len(df)}")
    return df.drop_duplicates(subset=["symbol"], keep="first").set_index("symbol")["theme"]


def _build_industry_map_from_ths(ak_module) -> pd.Series:
    try:
        board_df = ak_module.stock_board_industry_name_ths()
    except Exception:
        _write_industry_map_status("ths_industry_name_api_failed")
        return pd.Series(dtype=str)
    if board_df is None or board_df.empty:
        _write_industry_map_status("ths_industry_name_empty")
        return pd.Series(dtype=str)

    board_col = _pick_col(board_df.columns, ["name", "名称", "板块名称"])
    code_col = _pick_col(board_df.columns, ["code", "代码", "板块代码"])
    if not board_col:
        _write_industry_map_status("ths_industry_name_missing_name_column")
        return pd.Series(dtype=str)
    if not code_col:
        _write_industry_map_status("ths_industry_name_missing_code_column")
        return pd.Series(dtype=str)

    mapping_rows: List[dict] = []
    headers = {"User-Agent": "Mozilla/5.0"}
    for _, row in board_df.iterrows():
        board_name = str(row[board_col])
        board_code = str(row[code_col])
        members = _fetch_ths_industry_members(board_code, headers=headers)
        if not members:
            continue
        for sym in members:
            mapping_rows.append({"symbol": _normalize_symbol_code(sym), "theme": board_name})

    if not mapping_rows:
        _write_industry_map_status("ths_industry_members_empty")
        return pd.Series(dtype=str)
    map_df = pd.DataFrame(mapping_rows).drop_duplicates(subset=["symbol"], keep="first")
    _write_industry_map_status(f"ths_industry_members_ok rows={len(map_df)}")
    return map_df.set_index("symbol")["theme"]


def _fetch_ths_industry_members(board_code: str, headers: Dict[str, str]) -> List[str]:
    import requests
    from io import StringIO

    symbols: List[str] = []
    base_url = f"http://q.10jqka.com.cn/thshy/detail/code/{board_code}/"
    try:
        resp = requests.get(base_url, headers=headers, timeout=20)
    except Exception:
        _write_industry_map_status(f"ths_members_request_exception code={board_code}")
        return symbols
    if resp.status_code != 200:
        _write_industry_map_status(f"ths_members_request_non_200 code={board_code} status={resp.status_code}")
        return symbols
    resp.encoding = "gbk"
    total_pages = _extract_page_count(resp.text)
    for page in range(1, total_pages + 1):
        url = base_url if page == 1 else f"http://q.10jqka.com.cn/thshy/detail/code/{board_code}/page/{page}"
        try:
            page_resp = requests.get(url, headers=headers, timeout=20)
            if page_resp.status_code != 200:
                continue
            page_resp.encoding = "gbk"
            tables = pd.read_html(StringIO(page_resp.text))
        except Exception:
            continue
        if not tables:
            continue
        table = tables[0]
        if table.shape[1] < 3:
            continue
        # 同花顺页面表头受编码影响，取第2列作为代码列最稳
        code_series = table.iloc[:, 1].astype(str).str.extract(r"(\d{6})", expand=False).dropna()
        symbols.extend(code_series.tolist())
    return list(dict.fromkeys(symbols))


def _extract_page_count(html_text: str) -> int:
    soup = BeautifulSoup(html_text, "lxml")
    node = soup.find("span", attrs={"class": "page_info"})
    if not node or not node.text:
        return 1
    m = re.search(r"/\s*(\d+)", node.text)
    if not m:
        return 1
    try:
        return max(1, int(m.group(1)))
    except Exception:
        return 1


def _write_industry_map_status(message: str) -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        INDUSTRY_MAP_STATUS_PATH.write_text(f"[{ts}] {message}\n", encoding="utf-8")
    except Exception:
        pass


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

