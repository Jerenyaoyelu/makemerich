from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class SellRuleConfig:
    stop_loss_pct: float = -5.0
    tp1_pct: float = 8.0
    tp2_pct: float = 12.0
    high_drawdown_pct: float = -6.0
    max_holding_days: int = 5
    risk_tag_reduce_enabled: bool = True
    block_reduce_enabled: bool = True


def _first_existing_numeric(row: pd.Series, cols: list[str]) -> float | None:
    for c in cols:
        if c in row.index:
            v = pd.to_numeric(row.get(c), errors="coerce")
            if pd.notna(v):
                return float(v)
    return None


def apply_sell_rules(df: pd.DataFrame, cfg: SellRuleConfig) -> pd.DataFrame:
    """
    对复评明细追加卖出建议：
    - sell_action: 止损 / 减仓 / 分批止盈 / 持有观察 / 到期退出
    - sell_reason: 触发原因
    - sell_priority: 高 / 中 / 低
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    out["sell_action"] = "持有观察"
    out["sell_reason"] = "未触发强制卖出规则"
    out["sell_priority"] = "低"

    for idx, row in out.iterrows():
        r1n = _first_existing_numeric(row, ["ret_close_t1_net", "ret_close_t1"])
        r3n = _first_existing_numeric(row, ["ret_close_t3_net", "ret_close_t3"])
        r5n = _first_existing_numeric(row, ["ret_close_t5_net", "ret_close_t5"])
        mdd3 = _first_existing_numeric(row, ["max_drawdown_t3"])
        status = str(row.get("status", ""))
        tag = str(row.get("selection_tags", ""))
        block_reason = str(row.get("trade_block_reason", ""))

        # 1) 强止损
        if (r1n is not None and r1n <= cfg.stop_loss_pct) or (mdd3 is not None and mdd3 <= cfg.high_drawdown_pct):
            out.at[idx, "sell_action"] = "止损"
            out.at[idx, "sell_reason"] = (
                f"T+1收益 {r1n:.2f}% <= 止损线 {cfg.stop_loss_pct:.2f}%"
                if r1n is not None and r1n <= cfg.stop_loss_pct
                else f"T+3最大回撤 {mdd3:.2f}% <= {cfg.high_drawdown_pct:.2f}%"
            )
            out.at[idx, "sell_priority"] = "高"
            continue

        # 2) 交易受限/追涨风险 -> 减仓
        if (cfg.block_reduce_enabled and ("受限" in status or len(block_reason) > 0)) or (
            cfg.risk_tag_reduce_enabled and "追涨风险" in tag
        ):
            out.at[idx, "sell_action"] = "减仓"
            if "追涨风险" in tag:
                out.at[idx, "sell_reason"] = "标签命中追涨风险，先降仓位控制回撤"
            else:
                out.at[idx, "sell_reason"] = "存在交易受限风险，实盘可成交性折价"
            out.at[idx, "sell_priority"] = "中"
            continue

        # 3) 分批止盈
        ref_ret = r3n if r3n is not None else r1n
        if ref_ret is not None and ref_ret >= cfg.tp2_pct:
            out.at[idx, "sell_action"] = "分批止盈"
            out.at[idx, "sell_reason"] = f"收益 {ref_ret:.2f}% >= 第二止盈线 {cfg.tp2_pct:.2f}%（建议留底仓）"
            out.at[idx, "sell_priority"] = "中"
            continue
        if ref_ret is not None and ref_ret >= cfg.tp1_pct:
            out.at[idx, "sell_action"] = "分批止盈"
            out.at[idx, "sell_reason"] = f"收益 {ref_ret:.2f}% >= 第一止盈线 {cfg.tp1_pct:.2f}%（建议先落袋1/3）"
            out.at[idx, "sell_priority"] = "中"
            continue

        # 4) 到期退出（无明显优势）
        if cfg.max_holding_days <= 5:
            if r5n is not None and r5n <= 0:
                out.at[idx, "sell_action"] = "到期退出"
                out.at[idx, "sell_reason"] = f"T+5收益 {r5n:.2f}% 未转正，按时间止损退出"
                out.at[idx, "sell_priority"] = "中"
                continue

    return out


def sell_action_summary(df: pd.DataFrame) -> dict[str, int]:
    if df is None or df.empty or "sell_action" not in df.columns:
        return {}
    vc = df["sell_action"].astype(str).value_counts()
    return {k: int(v) for k, v in vc.items()}

