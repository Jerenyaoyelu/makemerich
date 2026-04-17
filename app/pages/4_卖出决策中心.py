from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.selection_tags import annotate_with_selection_tags
from core.sell_rules import SellRuleConfig, apply_sell_rules, sell_action_summary
from core.run_store import list_runs, load_run_evaluation

st.set_page_config(page_title="卖出决策中心", page_icon="🛟", layout="wide")
st.title("卖出决策中心")
st.caption("把复评结果转成可执行卖出动作：止损 / 减仓 / 分批止盈 / 到期退出。")

runs = list_runs()
if runs.empty:
    st.info("暂无 run 数据。请先在实时页或历史页运行筛选并完成复评。")
    st.stop()

run_ids = runs.sort_values("created_at")["run_id"].astype(str).tolist()
choice = st.selectbox("选择 run_id", run_ids, index=len(run_ids) - 1)
ev = load_run_evaluation(choice)
if ev is None or ev.empty:
    st.warning("该 run 暂无复评文件，请先到「复评中心」更新复评。")
    st.stop()
ev = annotate_with_selection_tags(ev)

with st.expander("卖出规则参数", expanded=True):
    c1, c2, c3 = st.columns(3)
    stop_loss = c1.number_input("止损线(%)", value=-5.0, step=0.5)
    tp1 = c2.number_input("第一止盈线(%)", value=8.0, step=0.5)
    tp2 = c3.number_input("第二止盈线(%)", value=12.0, step=0.5)

    c4, c5, c6 = st.columns(3)
    mdd = c4.number_input("高回撤阈值(%)", value=-6.0, step=0.5)
    max_holding = int(c5.slider("最大持有天数（到期退出）", 3, 10, 5))
    risk_reduce = c6.checkbox("追涨风险触发减仓", value=True)
    block_reduce = st.checkbox("交易受限触发减仓", value=True)

cfg = SellRuleConfig(
    stop_loss_pct=float(stop_loss),
    tp1_pct=float(tp1),
    tp2_pct=float(tp2),
    high_drawdown_pct=float(mdd),
    max_holding_days=max_holding,
    risk_tag_reduce_enabled=bool(risk_reduce),
    block_reduce_enabled=bool(block_reduce),
)
advice = apply_sell_rules(ev, cfg)

summary = sell_action_summary(advice)
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("止损", summary.get("止损", 0))
k2.metric("减仓", summary.get("减仓", 0))
k3.metric("分批止盈", summary.get("分批止盈", 0))
k4.metric("到期退出", summary.get("到期退出", 0))
k5.metric("持有观察", summary.get("持有观察", 0))

f1, f2 = st.columns(2)
high_only = f1.checkbox("仅看高优先级（执行清单）", value=False)
action_filter = f2.multiselect(
    "按卖出动作筛选",
    options=["止损", "减仓", "分批止盈", "到期退出", "持有观察"],
    default=[],
)

priority_order = pd.Categorical(
    advice.get("sell_priority", pd.Series([], dtype=str)),
    categories=["高", "中", "低"],
    ordered=True,
)
advice = advice.assign(_priority=priority_order).sort_values(["_priority", "total_score"], ascending=[True, False])
if high_only:
    advice = advice[advice["sell_priority"] == "高"].copy()
if action_filter:
    advice = advice[advice["sell_action"].isin(action_filter)].copy()

show_cols = [
    "symbol",
    "name",
    "total_score",
    "selection_tags",
    "selection_alert",
    "status",
    "trade_block_reason",
    "ret_close_t1_net",
    "ret_close_t3_net",
    "ret_close_t5_net",
    "ret_close_t1",
    "ret_close_t3",
    "ret_close_t5",
    "max_drawdown_t3",
    "sell_action",
    "sell_reason",
    "sell_priority",
]
show_cols = [c for c in show_cols if c in advice.columns]
view = advice[show_cols].copy()

def _priority_style(v: object) -> str:
    if str(v) == "高":
        return "background-color: #ffe6e6; color: #b00020; font-weight: 700;"
    if str(v) == "中":
        return "background-color: #fff5e6; color: #8a5a00; font-weight: 600;"
    return "background-color: #eef7ee; color: #1f6f3f;"

if "sell_priority" in view.columns:
    styled = view.style.map(_priority_style, subset=["sell_priority"])
    st.dataframe(styled, use_container_width=True)
else:
    st.dataframe(view, use_container_width=True)

csv = advice.drop(columns=["_priority"], errors="ignore").to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
st.download_button(
    "导出卖出建议 CSV",
    data=csv,
    file_name=f"sell_advice_{choice}.csv",
    mime="text/csv",
)

