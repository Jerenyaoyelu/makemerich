from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import time
import pandas as pd

HORIZONS = (1, 2, 3, 5, 10)


def format_symbol_for_ak(symbol: str) -> str:
    """
    `stock_zh_a_hist` 使用 6 位代码（如 000001），
    不使用 sz/sh/bj 前缀。
    """
    s = str(symbol).strip().lower()
    if s.startswith(("sh", "sz", "bj")):
        s = s[2:]
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return str(symbol).strip()
    return digits[-6:].zfill(6)


def _snapshot_ymd(snapshot_time: str) -> str:
    return pd.to_datetime(snapshot_time).strftime("%Y%m%d")


def _kth_trading_row_after(
    hist: pd.DataFrame, trade_date: str, k: int
) -> tuple[pd.Series | None, pd.Series | None]:
    """T0 为交易日 trade_date 的最后一根 K 线，返回 (T0行, T+k 行)。"""
    d = pd.to_datetime(trade_date).date()
    h = hist.copy()
    h["日期"] = pd.to_datetime(h["日期"]).dt.date
    today = h[h["日期"] == d]
    future = h[h["日期"] > d].sort_values("日期")
    if today.empty or future.empty or len(future) < k:
        return None, None
    t0 = today.iloc[-1]
    tk = future.iloc[k - 1]
    return t0, tk


def _window_rows(hist: pd.DataFrame, trade_date: str, max_k: int) -> pd.DataFrame | None:
    """返回 T+1 .. T+max_k 的日线子表（含 最高/最低/收盘）。"""
    d = pd.to_datetime(trade_date).date()
    h = hist.copy()
    h["日期"] = pd.to_datetime(h["日期"]).dt.date
    today = h[h["日期"] == d]
    future = h[h["日期"] > d].sort_values("日期")
    if today.empty or future.empty:
        return None
    return future.iloc[:max_k]


def _row_metrics(
    t0_close: float,
    window: pd.DataFrame | None,
) -> tuple[float | None, float | None]:
    """相对 T0 收盘的最大回撤（用窗口最低价）与最大冲高（用窗口最高价）。"""
    if window is None or window.empty:
        return None, None
    lows = pd.to_numeric(window["最低"], errors="coerce")
    highs = pd.to_numeric(window["最高"], errors="coerce")
    if t0_close <= 0 or lows.isna().all() or highs.isna().all():
        return None, None
    mdd = float((lows.min() / t0_close - 1.0) * 100.0)
    mru = float((highs.max() / t0_close - 1.0) * 100.0)
    return round(mdd, 4), round(mru, 4)


def _prefixed_symbol(code6: str) -> str:
    if code6.startswith(("6", "9")):
        return f"sh{code6}"
    if code6.startswith(("8", "4")):
        return f"bj{code6}"
    return f"sz{code6}"


def _normalize_hist_df(hist: pd.DataFrame) -> pd.DataFrame | None:
    if hist is None or hist.empty:
        return None
    h = hist.copy()
    col_date = "日期" if "日期" in h.columns else ("date" if "date" in h.columns else None)
    col_close = "收盘" if "收盘" in h.columns else ("close" if "close" in h.columns else None)
    col_high = "最高" if "最高" in h.columns else ("high" if "high" in h.columns else None)
    col_low = "最低" if "最低" in h.columns else ("low" if "low" in h.columns else None)
    if not col_date or not col_close or not col_high or not col_low:
        return None
    return pd.DataFrame(
        {
            "日期": pd.to_datetime(h[col_date], errors="coerce"),
            "收盘": pd.to_numeric(h[col_close], errors="coerce"),
            "最高": pd.to_numeric(h[col_high], errors="coerce"),
            "最低": pd.to_numeric(h[col_low], errors="coerce"),
        }
    ).dropna(subset=["日期"])


def _fetch_hist_window_multi_source(
    ak_module,
    *,
    code6: str,
    start_ymd: str,
    end_ymd: str,
) -> pd.DataFrame | None:
    proxy_keys = ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]

    def _run_with_retries(fn, max_attempts: int = 3) -> pd.DataFrame | None:
        backup = {k: os.environ.get(k) for k in proxy_keys}
        last_err: Exception | None = None
        try:
            for i in range(max_attempts):
                try:
                    if i >= 1:
                        for key in proxy_keys:
                            os.environ.pop(key, None)
                    return fn()
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
                    time.sleep(0.35 * (i + 1))
            if last_err:
                raise last_err
            return None
        finally:
            for key in proxy_keys:
                os.environ.pop(key, None)
            for key, val in backup.items():
                if val is not None:
                    os.environ[key] = val

    pref = _prefixed_symbol(code6)
    start_iso = f"{start_ymd[:4]}-{start_ymd[4:6]}-{start_ymd[6:8]}"
    end_iso = f"{end_ymd[:4]}-{end_ymd[4:6]}-{end_ymd[6:8]}"
    fetchers = (
        lambda: ak_module.stock_zh_a_hist(
            symbol=code6, period="daily", start_date=start_ymd, end_date=end_ymd, adjust="qfq"
        ),
        lambda: ak_module.stock_zh_a_daily(symbol=pref, start_date=start_ymd, end_date=end_ymd, adjust="qfq"),
        lambda: ak_module.stock_zh_a_hist_tx(symbol=pref, start_date=start_iso, end_date=end_iso, adjust="qfq"),
    )
    for fn in fetchers:
        try:
            hist = _run_with_retries(fn)
            norm = _normalize_hist_df(hist) if hist is not None else None
            if norm is not None and not norm.empty:
                return norm
        except Exception:
            continue
    return None


def evaluate_multi_horizon(
    df: pd.DataFrame,
    *,
    stop_loss_pct: float = -5.0,
    take_profit_pct: float = 10.0,
) -> pd.DataFrame:
    """
    输入候选表（须含 symbol, snapshot_time；建议含 name, theme, total_score）。
    输出追加多周期收益、回撤、标签与 status。
    """
    import akshare as ak

    if df.empty:
        return df
    need = {"symbol", "snapshot_time"}
    if not need.issubset(df.columns):
        raise ValueError(f"缺少字段: {need - set(df.columns)}")

    out = df.copy()
    for n in HORIZONS:
        out[f"ret_close_t{n}"] = pd.NA
    for w in (3, 5, 10):
        out[f"max_drawdown_t{w}"] = pd.NA
        out[f"max_runup_t{w}"] = pd.NA
    out["hit_stop_loss"] = pd.NA
    out["hit_take_profit"] = pd.NA
    out["status"] = "数据不足"

    end_date = datetime.now().strftime("%Y%m%d")

    for idx, row in out.iterrows():
        symbol = format_symbol_for_ak(row["symbol"])
        snap = str(row["snapshot_time"])
        snapshot_date = _snapshot_ymd(snap)
        hist = _fetch_hist_window_multi_source(
            ak,
            code6=symbol,
            start_ymd=snapshot_date,
            end_ymd=end_date,
        )
        if hist is None or hist.empty:
            out.at[idx, "status"] = "拉取失败"
            continue

        t0, _ = _kth_trading_row_after(hist, snapshot_date, 1)
        if t0 is None:
            out.at[idx, "status"] = "数据不足"
            continue
        try:
            t0_close = float(t0["收盘"])
        except Exception:
            out.at[idx, "status"] = "数据不足"
            continue
        if t0_close <= 0:
            continue

        rets: dict[int, float | None] = {}
        ok_horizons = 0
        for n in HORIZONS:
            _t0, tk = _kth_trading_row_after(hist, snapshot_date, n)
            if tk is None:
                rets[n] = None
                continue
            try:
                ck = float(tk["收盘"])
                rets[n] = round((ck / t0_close - 1.0) * 100.0, 4)
                ok_horizons += 1
            except Exception:
                rets[n] = None

        for n in HORIZONS:
            if rets.get(n) is not None:
                out.at[idx, f"ret_close_t{n}"] = rets[n]

        for w in (3, 5, 10):
            win = _window_rows(hist, snapshot_date, w)
            mdd, mru = _row_metrics(t0_close, win)
            if mdd is not None:
                out.at[idx, f"max_drawdown_t{w}"] = mdd
            if mru is not None:
                out.at[idx, f"max_runup_t{w}"] = mru

        win10 = _window_rows(hist, snapshot_date, 10)
        if win10 is not None and not win10.empty:
            lows = pd.to_numeric(win10["最低"], errors="coerce")
            highs = pd.to_numeric(win10["最高"], errors="coerce")
            worst_ret = float((lows.min() / t0_close - 1.0) * 100.0) if not lows.isna().all() else None
            best_ret = float((highs.max() / t0_close - 1.0) * 100.0) if not highs.isna().all() else None
            if worst_ret is not None:
                out.at[idx, "hit_stop_loss"] = bool(worst_ret <= stop_loss_pct)
            if best_ret is not None:
                out.at[idx, "hit_take_profit"] = bool(best_ret >= take_profit_pct)

        if ok_horizons >= 5:
            out.at[idx, "status"] = "已完成"
        elif ok_horizons > 0:
            out.at[idx, "status"] = "待更新"
        else:
            out.at[idx, "status"] = "数据不足"

    return out


def evaluate_csv_to_path(input_csv: Path, output_csv: Path) -> Path:
    df = pd.read_csv(input_csv, dtype={"symbol": str})
    result = evaluate_multi_horizon(df)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return output_csv


def portfolio_summary(eval_df: pd.DataFrame, ret_col: str = "ret_close_t1") -> dict:
    if eval_df.empty or ret_col not in eval_df.columns:
        return {}
    s = pd.to_numeric(eval_df[ret_col], errors="coerce").dropna()
    if s.empty:
        return {}
    wins = (s > 0).mean()
    losses = s[s < 0]
    gains = s[s > 0]
    loss_mag = float(losses.abs().mean()) if not losses.empty else 0.0
    gain_mag = float(gains.mean()) if not gains.empty else 0.0
    pf = gain_mag / loss_mag if loss_mag > 1e-9 else None
    return {
        "n": int(len(s)),
        "mean_ret": float(s.mean()),
        "median_ret": float(s.median()),
        "win_rate": float(wins),
        "profit_factor_approx": float(pf) if pf is not None else None,
    }


def layer_summary(eval_df: pd.DataFrame, score_col: str = "total_score", ret_col: str = "ret_close_t1") -> pd.DataFrame:
    """Top3/5/10 按得分排序后的平均收益。"""
    if eval_df.empty or score_col not in eval_df.columns:
        return pd.DataFrame()
    sub = eval_df.copy()
    sub["_ret"] = pd.to_numeric(sub[ret_col], errors="coerce")
    sub = sub.sort_values(score_col, ascending=False).reset_index(drop=True)
    rows = []
    for k in (3, 5, 10):
        part = sub.head(k)
        r = pd.to_numeric(part["_ret"], errors="coerce").dropna()
        rows.append(
            {
                "layer": f"Top{k}",
                "n": len(r),
                "mean_ret": float(r.mean()) if not r.empty else None,
            }
        )
    return pd.DataFrame(rows)
