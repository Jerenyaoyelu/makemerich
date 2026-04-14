from __future__ import annotations

from datetime import datetime
import os
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

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
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INDUSTRY_MAP_PATH = DATA_DIR / "industry_map.csv"
INDUSTRY_MAP_STATUS_PATH = DATA_DIR / "industry_map_status.txt"
ProgressCallback = Callable[[int, str], None]

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
) -> pd.DataFrame:
    try:
        import akshare as ak
    except Exception as exc:
        raise RuntimeError("未安装 akshare，无法自动采集数据。请先执行 pip install -r requirements.txt") from exc

    if progress_callback:
        progress_callback(5, "准备连接数据源...")
    spot, source = _fetch_spot_multi_source(ak, progress_callback=progress_callback)
    if spot is None or spot.empty:
        raise RuntimeError("未获取到实时行情数据。")
    if progress_callback:
        progress_callback(40, f"数据源 {source} 已返回，正在清洗与补齐字段...")

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

    # 行业回填：优先读取本地行业映射缓存；若不存在则尝试自动更新一次
    industry_map = _load_or_refresh_industry_map(ak)
    if not industry_map.empty:
        missing_theme_mask = (data["theme"].isin(["未知行业", "", "None"])) | data["theme"].isna()
        filled_theme = data.loc[missing_theme_mask, "symbol_norm"].map(industry_map)
        fill_ok = filled_theme.notna()
        if fill_ok.any():
            idx = filled_theme[fill_ok].index
            data.loc[idx, "theme"] = filled_theme.loc[idx]
            data.loc[idx, "theme_source"] = "industry_map_cache"

    # 换手率回填：用成交额/流通市值估算
    nmc_col = _pick_col(spot.columns, ["流通市值", "nmc"])
    if nmc_col:
        nmc_series = pd.to_numeric(spot[nmc_col], errors="coerce").replace(0, np.nan)
        # data 已经过滤，索引不连续；必须按索引对齐，不能直接塞全量 values
        data["float_mktcap"] = nmc_series.reindex(data.index)
        need_turnover_est = data["turnover"].isna() | (data["turnover"] <= 0)
        est_turnover = (data["amount"] / data["float_mktcap"] * 100).replace([np.inf, -np.inf], np.nan)
        est_ok = need_turnover_est & est_turnover.notna()
        if est_ok.any():
            data.loc[est_ok, "turnover"] = est_turnover.loc[est_ok]
            data.loc[est_ok, "turnover_source"] = "estimated_by_amount_nmc"
    else:
        data["float_mktcap"] = np.nan

    # 量比缺失保持缺失；只在打分中中性处理，同时记录来源
    data.loc[data["volume_ratio"].isna(), "volume_ratio_source"] = "missing"

    industry_stats = data.groupby("theme").agg(
        industry_mean_pct=("pct_chg", "mean"),
        industry_up_ratio=("pct_chg", lambda s: (s > 0).mean() * 100),
        industry_amount=("amount", "sum"),
    )
    data = data.join(industry_stats, on="theme")

    # 量比缺失时用1.0作为中性占位；并降低量比系数权重，避免虚假量比污染排序
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
    data["snapshot_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
    ak_module, progress_callback: ProgressCallback | None = None
) -> tuple[pd.DataFrame, str]:
    primary_error: Exception | None = None
    fallback_error: Exception | None = None
    try:
        if progress_callback:
            progress_callback(12, "尝试主源：东方财富...")
        return _fetch_spot_with_proxy_strategy(ak_module), "eastmoney_em"
    except Exception as exc:  # noqa: BLE001
        primary_error = exc

    try:
        if progress_callback:
            progress_callback(18, "主源失败，切换备源：新浪财经（分页采集）...")
        # akshare 的 stock_zh_a_spot会丢弃新浪返回的 turnoverratio，导致换手恒为 0
        return _fetch_sina_zh_a_spot_with_turnover(progress_callback=progress_callback), "sina_finance"
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

