from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.evaluation import evaluate_multi_horizon, layer_summary, portfolio_summary
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
    f"当时健康分：**{row.get('health_score', '')}** ｜ 源：**{row.get('source_used', '')}**"
)

c1, c2, c3 = st.columns(3)
with c1:
    do_eval = st.button("更新复评数据（当前 run）", type="primary")
with c2:
    do_batch = st.button("更新全部未完成 run", help="对缺复评或状态未完成的 run 逐个拉行情（较慢）")
with c3:
    pass

if do_eval:
    cands = load_run_candidates(choice)
    if cands is None or cands.empty:
        st.error("该 run 无候选快照或文件缺失。")
    else:
        bar = st.progress(0, text="正在多周期复评（逐票请求行情）...")
        try:
            result = evaluate_multi_horizon(cands)
            bar.progress(100, text="写入文件...")
            out_p = run_evaluation_path(choice)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(out_p, index=False, encoding="utf-8-sig")
            bar.empty()
            st.success(f"已写入：{out_p}")
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
                result = evaluate_multi_horizon(cands)
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

st.subheader("单票表现")
show_cols = ["symbol", "name", "total_score", "status"]
for col, lab in DISPLAY_RET:
    if col in ev_df.columns:
        show_cols.append(col)
for col in ev_df.columns:
    if col.startswith("max_drawdown") or col.startswith("max_runup"):
        show_cols.append(col)
if "hit_stop_loss" in ev_df.columns:
    show_cols.append("hit_stop_loss")
if "hit_take_profit" in ev_df.columns:
    show_cols.append("hit_take_profit")
show_cols = [c for c in show_cols if c in ev_df.columns]
st.dataframe(ev_df[show_cols], use_container_width=True)

st.subheader("组合统计（等权看待选列表）")
cols_m = []
for col, lab in DISPLAY_RET:
    if col in ev_df.columns:
        s = portfolio_summary(ev_df, col)
        if s:
            cols_m.append({"周期": lab, **{k: s.get(k) for k in s}})

if cols_m:
    pm = pd.DataFrame(cols_m)
    st.dataframe(pm, use_container_width=True)

st.subheader("分层表现（按综合得分排序）")
for col, lab in DISPLAY_RET:
    if col not in ev_df.columns:
        continue
    ls = layer_summary(ev_df, ret_col=col)
    if not ls.empty:
        st.markdown(f"**{lab}**")
        st.dataframe(ls, use_container_width=True)
