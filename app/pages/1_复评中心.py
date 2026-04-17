from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.evaluation import evaluate_multi_horizon, layer_summary, portfolio_summary
from core.selection_tags import annotate_with_selection_tags
from core.sell_rules import SellRuleConfig, apply_sell_rules, sell_action_summary
from core.run_store import (
    append_strategy_report,
    list_runs,
    load_run_candidates,
    load_run_evaluation,
    run_evaluation_path,
)

DISPLAY_RET = [
    ("ret_close_t1", "T+1 收盘收益%"),
    ("ret_close_t2", "T+2"),
    ("ret_close_t3", "T+3"),
    ("ret_close_t5", "T+5"),
    ("ret_close_t10", "T+10"),
]


def _display_name(col: str) -> str:
    base_map = {
        "symbol": "代码",
        "name": "名称",
        "total_score": "综合得分",
        "status": "状态",
        "ret_close_t1": "T+1 收盘收益%",
        "ret_close_t2": "T+2 收盘收益%",
        "ret_close_t3": "T+3 收盘收益%",
        "ret_close_t5": "T+5 收盘收益%",
        "ret_close_t10": "T+10 收盘收益%",
        "ret_close_t1_net": "T+1 净收益%(交易约束后)",
        "ret_close_t2_net": "T+2 净收益%(交易约束后)",
        "ret_close_t3_net": "T+3 净收益%(交易约束后)",
        "ret_close_t5_net": "T+5 净收益%(交易约束后)",
        "ret_close_t10_net": "T+10 净收益%(交易约束后)",
        "t0_trade_date": "T0交易日",
        "t0_close": "T0收盘价",
        "t1_trade_date": "T+1交易日",
        "t2_trade_date": "T+2交易日",
        "t3_trade_date": "T+3交易日",
        "t5_trade_date": "T+5交易日",
        "t10_trade_date": "T+10交易日",
        "t1_close": "T+1收盘价",
        "t2_close": "T+2收盘价",
        "t3_close": "T+3收盘价",
        "t5_close": "T+5收盘价",
        "t10_close": "T+10收盘价",
        "hit_stop_loss": "是否触发止损",
        "hit_take_profit": "是否触发止盈",
        "trade_block_reason": "交易受限原因",
        "sell_action": "卖出动作",
        "sell_reason": "卖出原因",
        "sell_priority": "优先级",
        "selection_tags": "选股标签",
        "selection_alert": "提醒级别",
        "selection_tooltip": "触发说明",
        "n": "样本数",
        "mean_ret": "平均收益%",
        "median_ret": "中位收益%",
        "win_rate": "胜率",
        "profit_factor_approx": "近似盈亏比",
        "layer": "分层",
    }
    if col in base_map:
        return base_map[col]
    if col.startswith("max_drawdown_t"):
        n = col.split("t")[-1]
        return f"T+{n} 最大回撤%"
    if col.startswith("max_runup_t"):
        n = col.split("t")[-1]
        return f"T+{n} 最大冲高%"
    return col


def _rename_for_display(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: _display_name(c) for c in df.columns})


def _run_needs_eval(run_id: str) -> bool:
    ev = load_run_evaluation(run_id)
    if ev is None or ev.empty:
        return True
    if "status" not in ev.columns:
        return True
    return bool((ev["status"].astype(str) != "已完成").any())


st.set_page_config(page_title="复评中心", page_icon="📊", layout="wide")
st.title("复评中心")
st.caption("选择历史 run，拉取多周期真实表现并查看组合与分层统计。")

runs_df = list_runs()
if runs_df.empty:
    st.info("暂无运行记录。请先在「程序化选股助手」页完成一次筛选。")
    st.stop()

run_ids = runs_df["run_id"].astype(str).tolist()
choice = st.selectbox("选择 run_id", run_ids, index=len(run_ids) - 1)

row = runs_df[runs_df["run_id"] == choice].iloc[0]
st.write(
    f"创建时间：**{row.get('created_at', '')}** ｜ 候选数：**{row.get('candidate_count', '')}** ｜ "
    f"当时健康分：**{row.get('health_score', '')}** ｜ 源：**{row.get('source_used', '')}** ｜ "
    f"类型：**{row.get('data_source', '')}**"
)

c1, c2, c3 = st.columns(3)
with c1:
    do_eval = st.button("更新复评数据（当前 run）", type="primary")
with c2:
    do_batch = st.button("更新全部未完成 run", help="对缺复评或状态未完成的 run 逐个拉行情（较慢）")
with c3:
    use_net_for_summary = st.checkbox("统计使用净收益", value=True, help="勾选后组合统计/分层表现优先使用交易约束后的净收益列。")

with st.expander("交易约束参数（基础版）", expanded=False):
    fee_bps = st.number_input("单边手续费(bps)", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
    slippage_bps = st.number_input("单边滑点(bps)", min_value=0.0, max_value=100.0, value=8.0, step=0.5)
    block_limit_up_at_t1 = st.checkbox("T+1 涨停视作难买入（净收益记空）", value=True)
    limit_up_threshold_pct = st.number_input("涨停判定阈值(%)", min_value=0.0, max_value=30.0, value=9.8, step=0.1)

if do_eval:
    cands = load_run_candidates(choice)
    if cands is None or cands.empty:
        st.error("该 run 无候选快照或文件缺失。")
    else:
        bar = st.progress(0, text="正在多周期复评（逐票请求行情）...")
        try:
            result = evaluate_multi_horizon(
                cands,
                fee_bps=float(fee_bps),
                slippage_bps=float(slippage_bps),
                block_limit_up_at_t1=bool(block_limit_up_at_t1),
                limit_up_threshold_pct=float(limit_up_threshold_pct),
            )
            bar.progress(100, text="写入文件...")
            out_p = run_evaluation_path(choice)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(out_p, index=False, encoding="utf-8-sig")
            bar.empty()
            st.success(f"已写入：{out_p}")
            if "ret_close_t1" in result.columns:
                t1_valid = int(pd.to_numeric(result["ret_close_t1"], errors="coerce").notna().sum())
                if t1_valid == 0:
                    st.warning("当前 run 还没有可用的 T+1 收盘数据。若是今日/近两日 run，请等待后续交易日再更新复评。")
            summ = portfolio_summary(result, "ret_close_t1")
            if summ:
                append_strategy_report(
                    {
                        "run_id": choice,
                        "eval_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "horizon": "t1_summary",
                        **{k: summ.get(k) for k in summ},
                    }
                )
        except Exception as exc:
            st.error(str(exc))

if do_batch:
    todo = [r for r in run_ids if _run_needs_eval(r)]
    if not todo:
        st.success("没有需要更新的 run。")
    else:
        prog = st.progress(0, text=f"待处理 {len(todo)} 个 run...")
        for i, rid in enumerate(todo):
            cands = load_run_candidates(rid)
            if cands is None or cands.empty:
                continue
            try:
                result = evaluate_multi_horizon(
                    cands,
                    fee_bps=float(fee_bps),
                    slippage_bps=float(slippage_bps),
                    block_limit_up_at_t1=bool(block_limit_up_at_t1),
                    limit_up_threshold_pct=float(limit_up_threshold_pct),
                )
                out_p = run_evaluation_path(rid)
                result.to_csv(out_p, index=False, encoding="utf-8-sig")
            except Exception:
                pass
            prog.progress((i + 1) / len(todo), text=f"已完成 {i + 1}/{len(todo)}")
        st.success("批量更新结束（失败项已跳过）。")

ev_df = load_run_evaluation(choice)
if ev_df is None or ev_df.empty:
    st.warning("暂无复评结果文件。请点击「更新复评数据」。")
    st.stop()

ev_df = annotate_with_selection_tags(ev_df)
sell_cfg = SellRuleConfig()
sell_df = apply_sell_rules(ev_df, sell_cfg)
sell_sum = sell_action_summary(sell_df)

st.subheader("卖出建议卡（MVP）")
s1, s2, s3, s4, s5 = st.columns(5)
s1.metric("止损", sell_sum.get("止损", 0))
s2.metric("减仓", sell_sum.get("减仓", 0))
s3.metric("分批止盈", sell_sum.get("分批止盈", 0))
s4.metric("到期退出", sell_sum.get("到期退出", 0))
s5.metric("持有观察", sell_sum.get("持有观察", 0))
only_high_sell = st.checkbox("单票表现仅看高优先级卖出建议", value=False)
if st.button("打开卖出决策中心"):
    try:
        st.switch_page("pages/4_卖出决策中心.py")
    except Exception:
        st.info("当前环境不支持自动跳转，请在左侧菜单打开「卖出决策中心」。")

st.subheader("单票表现")
show_cols = [
    "symbol",
    "name",
    "total_score",
    "selection_tags",
    "selection_alert",
    "selection_tooltip",
    "status",
    "trade_block_reason",
    "t0_trade_date",
    "t0_close",
]
for col, lab in DISPLAY_RET:
    if col in ev_df.columns:
        show_cols.append(col)
    net_col = f"{col}_net"
    if net_col in ev_df.columns:
        show_cols.append(net_col)
    date_col = col.replace("ret_close_", "").replace("t", "t") + "_trade_date"
    close_col = col.replace("ret_close_", "").replace("t", "t") + "_close"
    if date_col in ev_df.columns:
        show_cols.append(date_col)
    if close_col in ev_df.columns:
        show_cols.append(close_col)
for col in ev_df.columns:
    if col.startswith("max_drawdown") or col.startswith("max_runup"):
        show_cols.append(col)
if "hit_stop_loss" in ev_df.columns:
    show_cols.append("hit_stop_loss")
if "hit_take_profit" in ev_df.columns:
    show_cols.append("hit_take_profit")
show_cols += ["sell_action", "sell_reason", "sell_priority"]
show_cols = [c for c in show_cols if c in sell_df.columns]
view_sell = sell_df.copy()
if only_high_sell and "sell_priority" in view_sell.columns:
    view_sell = view_sell[view_sell["sell_priority"] == "高"].copy()

def _priority_style(v: object) -> str:
    if str(v) == "高":
        return "background-color: #ffe6e6; color: #b00020; font-weight: 700;"
    if str(v) == "中":
        return "background-color: #fff5e6; color: #8a5a00; font-weight: 600;"
    return "background-color: #eef7ee; color: #1f6f3f;"

display_df = _rename_for_display(view_sell[show_cols])
prio_col = _display_name("sell_priority")
if prio_col in display_df.columns:
    st.dataframe(display_df.style.map(_priority_style, subset=[prio_col]), use_container_width=True)
else:
    st.dataframe(display_df, use_container_width=True)

st.subheader("组合统计（等权看待选列表）")
cols_m = []
for col, lab in DISPLAY_RET:
    chosen = col
    net_col = f"{col}_net"
    if use_net_for_summary and net_col in ev_df.columns:
        chosen = net_col
    if chosen in ev_df.columns:
        s = portfolio_summary(ev_df, chosen)
        if s:
            cols_m.append({"周期": lab, **{k: s.get(k) for k in s}})

if cols_m:
    pm = pd.DataFrame(cols_m)
    st.dataframe(_rename_for_display(pm), use_container_width=True)

st.subheader("分层表现（按综合得分排序）")
for col, lab in DISPLAY_RET:
    chosen = col
    net_col = f"{col}_net"
    if use_net_for_summary and net_col in ev_df.columns:
        chosen = net_col
    if chosen not in ev_df.columns:
        continue
    ls = layer_summary(ev_df, ret_col=chosen)
    if not ls.empty:
        st.markdown(f"**{lab}**")
        st.dataframe(_rename_for_display(ls), use_container_width=True)
