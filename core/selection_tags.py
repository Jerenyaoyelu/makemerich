from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from core.run_store import list_runs, load_run_candidates, load_run_evaluation


def _norm_symbol(symbol: object) -> str:
    s = str(symbol or "").strip().lower()
    if s.startswith(("sh", "sz", "bj")):
        s = s[2:]
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits[-6:].zfill(6) if digits else str(symbol or "").strip()


def _pick_ret_col(ev: pd.DataFrame) -> str | None:
    for c in ("ret_close_t1_net", "ret_close_t3_net", "ret_close_t1", "ret_close_t3"):
        if c in ev.columns:
            return c
    return None


def _safe_std(values: Iterable[float]) -> float:
    s = pd.Series(list(values), dtype=float)
    if s.empty:
        return float("nan")
    return float(s.std(ddof=0))


@dataclass
class _SymbolRollup:
    appear: dict[str, int]
    ret_hist: dict[str, list[float]]
    block_hit: dict[str, int]
    eval_cnt: dict[str, int]
    last_name: dict[str, str]
    ordered_run_ids: list[str]
    per_run_syms: list[set[str]]


def _build_symbol_rollup(last_n_runs: int) -> _SymbolRollup | None:
    runs = list_runs()
    if runs.empty:
        return None
    tail = runs.sort_values("created_at").tail(last_n_runs)
    ordered_run_ids = tail["run_id"].astype(str).tolist()

    appear: dict[str, int] = {}
    ret_hist: dict[str, list[float]] = {}
    block_hit: dict[str, int] = {}
    eval_cnt: dict[str, int] = {}
    last_name: dict[str, str] = {}
    per_run_syms: list[set[str]] = []

    for rid in ordered_run_ids:
        cands = load_run_candidates(rid)
        if cands is None or cands.empty or "symbol" not in cands.columns:
            per_run_syms.append(set())
            continue
        cands = cands.copy()
        cands["symbol_norm"] = cands["symbol"].map(_norm_symbol)
        syms = set(cands["symbol_norm"].tolist())
        per_run_syms.append(syms)
        for s in syms:
            appear[s] = appear.get(s, 0) + 1
            if "name" in cands.columns:
                row = cands[cands["symbol_norm"] == s]
                if not row.empty:
                    last_name[s] = str(row.iloc[-1]["name"])

        ev = load_run_evaluation(rid)
        if ev is None or ev.empty or "symbol" not in ev.columns:
            continue
        col = _pick_ret_col(ev)
        ev2 = ev.copy()
        ev2["symbol_norm"] = ev2["symbol"].map(_norm_symbol)
        if col:
            vals = pd.to_numeric(ev2[col], errors="coerce")
            for s, v in zip(ev2["symbol_norm"], vals):
                if pd.notna(v):
                    ret_hist.setdefault(s, []).append(float(v))
                    eval_cnt[s] = eval_cnt.get(s, 0) + 1
        if "status" in ev2.columns:
            status_s = ev2["status"].astype(str)
            reason_s = (
                ev2["trade_block_reason"].astype(str)
                if "trade_block_reason" in ev2.columns
                else pd.Series("", index=ev2.index)
            )
            blocked = status_s.str.contains("受限", na=False) | reason_s.str.len().gt(0)
            for s, b in zip(ev2["symbol_norm"], blocked):
                if bool(b):
                    block_hit[s] = block_hit.get(s, 0) + 1

    return _SymbolRollup(
        appear=appear,
        ret_hist=ret_hist,
        block_hit=block_hit,
        eval_cnt=eval_cnt,
        last_name=last_name,
        ordered_run_ids=ordered_run_ids,
        per_run_syms=per_run_syms,
    )


def _consecutive_streak(per_run_syms: list[set[str]], symbol: str) -> int:
    c = 0
    for sset in reversed(per_run_syms):
        if symbol in sset:
            c += 1
        else:
            break
    return c


def _tags_from_metrics(
    freq: int,
    evn: int,
    mean3: float,
    std3: float,
    br: float,
) -> tuple[list[str], list[str], str]:
    tags: list[str] = []
    reasons: list[str] = []
    if freq >= 5:
        tags.append("高频入选")
        reasons.append(f"近窗口入选 {freq} 次")
    if evn < 3:
        tags.append("观察中")
        reasons.append(f"可用复评样本 {evn} < 3")
    if freq >= 5 and evn >= 3 and pd.notna(mean3) and mean3 > 0 and (pd.isna(std3) or std3 <= 3.0):
        tags.append("高置信核心")
        reasons.append(f"近3次收益均值 {mean3:.2f}%")
    if freq >= 5 and evn >= 3 and ((pd.notna(mean3) and mean3 <= 0) or (pd.notna(std3) and std3 > 3.0)):
        tags.append("追涨风险")
        if pd.notna(mean3):
            reasons.append(f"近3次收益均值 {mean3:.2f}%")
        if pd.notna(std3):
            reasons.append(f"近3次收益波动 {std3:.2f}")
    if evn >= 1 and br >= 0.3:
        tags.append("可交易性折价")
        reasons.append(f"受限比例 {br * 100:.1f}%")

    alert = "正常"
    if "追涨风险" in tags or "可交易性折价" in tags:
        alert = "关注"
    if "高置信核心" in tags and "追涨风险" not in tags:
        alert = "积极"
    return tags, reasons, alert


def _snapshot_df_from_rollup(rollup: _SymbolRollup) -> pd.DataFrame:
    rows: list[dict] = []
    all_symbols = (
        set(rollup.appear.keys())
        | set(rollup.ret_hist.keys())
        | set(rollup.block_hit.keys())
        | set(rollup.eval_cnt.keys())
    )
    for s in sorted(all_symbols):
        freq = int(rollup.appear.get(s, 0))
        rets = rollup.ret_hist.get(s, [])
        recent3 = rets[-3:] if rets else []
        mean3 = float(pd.Series(recent3, dtype=float).mean()) if recent3 else float("nan")
        std3 = _safe_std(recent3) if len(recent3) >= 2 else float("nan")
        evn = int(rollup.eval_cnt.get(s, 0))
        br = float(rollup.block_hit.get(s, 0)) / max(1, evn)

        tags, reasons, alert = _tags_from_metrics(freq, evn, mean3, std3, br)
        rows.append(
            {
                "symbol_norm": s,
                "selection_tags": "｜".join(tags) if tags else "—",
                "selection_alert": alert,
                "selection_tooltip": "；".join(reasons) if reasons else "暂无异常提醒",
            }
        )

    return pd.DataFrame(rows)


def build_symbol_tag_snapshot(last_n_runs: int = 20) -> pd.DataFrame:
    rollup = _build_symbol_rollup(last_n_runs)
    if rollup is None:
        return pd.DataFrame(columns=["symbol_norm", "selection_tags", "selection_alert", "selection_tooltip"])
    return _snapshot_df_from_rollup(rollup)


def build_high_frequency_leaderboard(
    last_n_runs: int = 20,
    min_appear: int = 2,
) -> pd.DataFrame:
    rollup = _build_symbol_rollup(last_n_runs)
    if rollup is None:
        return pd.DataFrame()

    snap = _snapshot_df_from_rollup(rollup)

    rows: list[dict] = []
    for sym, freq in rollup.appear.items():
        if freq < min_appear:
            continue
        rets = rollup.ret_hist.get(sym, [])
        recent3 = rets[-3:] if rets else []
        recent5 = rets[-5:] if rets else []
        mean3 = float(pd.Series(recent3, dtype=float).mean()) if recent3 else float("nan")
        mean5 = float(pd.Series(recent5, dtype=float).mean()) if recent5 else float("nan")
        std3 = _safe_std(recent3) if len(recent3) >= 2 else float("nan")
        evn = int(rollup.eval_cnt.get(sym, 0))
        br = float(rollup.block_hit.get(sym, 0)) / max(1, evn)
        streak = _consecutive_streak(rollup.per_run_syms, sym)

        rows.append(
            {
                "symbol_norm": sym,
                "name": rollup.last_name.get(sym, ""),
                "appear_count": freq,
                "consecutive_runs": streak,
                "eval_samples": evn,
                "ret_mean_last3": round(mean3, 4) if pd.notna(mean3) else pd.NA,
                "ret_mean_last5": round(mean5, 4) if pd.notna(mean5) else pd.NA,
                "ret_std_last3": round(std3, 4) if pd.notna(std3) else pd.NA,
                "block_ratio": round(br, 4),
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    if not snap.empty:
        out = out.merge(snap, on="symbol_norm", how="left")
    else:
        out["selection_tags"] = "—"
        out["selection_alert"] = "正常"
        out["selection_tooltip"] = ""
    for col, default in (
        ("selection_tags", "—"),
        ("selection_alert", "正常"),
        ("selection_tooltip", ""),
    ):
        if col in out.columns:
            out[col] = out[col].fillna(default)

    return out.sort_values(["appear_count", "consecutive_runs"], ascending=False).reset_index(drop=True)


def annotate_with_selection_tags(df: pd.DataFrame, last_n_runs: int = 20) -> pd.DataFrame:
    if df is None or df.empty or "symbol" not in df.columns:
        return df
    out = df.copy()
    had_symbol_norm = "symbol_norm" in out.columns
    out["symbol_norm"] = out["symbol"].map(_norm_symbol)
    snap = build_symbol_tag_snapshot(last_n_runs=last_n_runs)
    if snap.empty:
        out["selection_tags"] = "—"
        out["selection_alert"] = "正常"
        out["selection_tooltip"] = "暂无历史样本"
        return out
    out = out.merge(snap, on="symbol_norm", how="left")
    out["selection_tags"] = out["selection_tags"].fillna("—")
    out["selection_alert"] = out["selection_alert"].fillna("正常")
    out["selection_tooltip"] = out["selection_tooltip"].fillna("暂无历史样本")
    if not had_symbol_norm:
        out = out.drop(columns=["symbol_norm"], errors="ignore")
    return out
