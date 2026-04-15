from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.run_store import (
    COLLECTION_HISTORY_CSV,
    DEFAULT_WEIGHTS_JSON,
    STRATEGY_REPORT_CSV,
    list_runs,
    load_run_evaluation,
)
from core.walk_forward import WalkForwardConfig, walk_forward_report

st.set_page_config(page_title="系统驾驶舱", page_icon="🧭", layout="wide")
st.title("系统驾驶舱")
st.caption("每日只看这一页：采集健康、复评表现、样本外稳定性、当前默认权重。")


def _pick_ret_col(ev: pd.DataFrame) -> str | None:
    for c in ("ret_close_t5_net", "ret_close_t3_net", "ret_close_t1_net", "ret_close_t5", "ret_close_t3", "ret_close_t1"):
        if c in ev.columns:
            return c
    return None


def _eval_run_snapshot(run_id: str) -> dict | None:
    ev = load_run_evaluation(run_id)
    if ev is None or ev.empty:
        return None
    ret_col = _pick_ret_col(ev)
    if not ret_col:
        return None
    s = pd.to_numeric(ev[ret_col], errors="coerce").dropna()
    if s.empty:
        return None
    return {
        "run_id": run_id,
        "ret_col": ret_col,
        "n": int(len(s)),
        "mean_ret": float(s.mean()),
        "median_ret": float(s.median()),
        "win_rate": float((s > 0).mean()),
    }


runs = list_runs()
if runs.empty:
    st.info("暂无 run 数据。请先在实时页或历史页运行一次筛选。")
    st.stop()

runs = runs.sort_values("created_at").reset_index(drop=True)
recent_runs = runs.tail(20).copy()

recent_eval_rows: list[dict] = []
for rid in recent_runs["run_id"].astype(str).tolist():
    item = _eval_run_snapshot(rid)
    if item:
        recent_eval_rows.append(item)

recent_eval = pd.DataFrame(recent_eval_rows)

c1, c2, c3, c4 = st.columns(4)
c1.metric("累计 run 数", int(len(runs)))
c2.metric("近20次平均健康分", f"{pd.to_numeric(recent_runs['health_score'], errors='coerce').mean():.1f}")
c3.metric("近20次已复评 run", int(len(recent_eval)))
if not recent_eval.empty:
    c4.metric("近20次平均收益(按run均值)", f"{recent_eval['mean_ret'].mean():.3f}%")
else:
    c4.metric("近20次平均收益(按run均值)", "—")

st.subheader("采集健康（近20次）")
left, right = st.columns(2)
with left:
    if COLLECTION_HISTORY_CSV.exists():
        h = pd.read_csv(COLLECTION_HISTORY_CSV)
        if not h.empty:
            tail = h.tail(20).copy()
            succ = tail["success"].astype(bool)
            st.metric("采集成功率", f"{succ.mean() * 100:.1f}%")
            src = tail["source"].astype(str).value_counts().rename_axis("source").reset_index(name="count")
            st.dataframe(src, use_container_width=True)
        else:
            st.info("collection_history 为空。")
    else:
        st.info("暂无 collection_history.csv。")

with right:
    if "source_used" in recent_runs.columns:
        src_runs = (
            recent_runs["source_used"]
            .astype(str)
            .value_counts()
            .rename_axis("source_used")
            .reset_index(name="count")
        )
        st.metric("近20次备源降级次数", int(pd.to_numeric(recent_runs.get("degraded"), errors="coerce").fillna(0).astype(bool).sum()))
        st.dataframe(src_runs, use_container_width=True)

st.subheader("复评表现（近20次）")
if recent_eval.empty:
    st.warning("近20次 run 尚无可用复评收益列。")
else:
    show = recent_eval.copy()
    show = show.rename(
        columns={
            "run_id": "run_id",
            "ret_col": "收益口径",
            "n": "样本数",
            "mean_ret": "平均收益%",
            "median_ret": "中位收益%",
            "win_rate": "胜率",
        }
    )
    st.dataframe(show, use_container_width=True)
    st.caption("说明：每个 run 的收益统计来自对应复评文件中的优先口径（优先 net 列）。")
    dist = pd.DataFrame({"mean_ret": recent_eval["mean_ret"]})
    st.bar_chart(dist)

st.subheader("滚动样本外（Walk-Forward）")
wf_col1, wf_col2, wf_col3, wf_col4 = st.columns(4)
wf_ret_col = wf_col1.selectbox("目标列", ["ret_close_t5_net", "ret_close_t3_net", "ret_close_t1_net", "ret_close_t5", "ret_close_t3", "ret_close_t1"], index=0)
wf_top_n = int(wf_col2.slider("Top N", 3, 15, 5))
wf_min_train = int(wf_col3.slider("最小训练run", 1, 12, 3))
wf_n_random = int(wf_col4.slider("随机组数", 20, 300, 120, 10))

if st.button("刷新 Walk-Forward 结果", type="primary"):
    cfg = WalkForwardConfig(
        ret_col=wf_ret_col,
        top_n=wf_top_n,
        min_train_runs=wf_min_train,
        mode="random",
        n_random=wf_n_random,
        seed=42,
    )
    with st.spinner("计算中..."):
        st.session_state["dashboard_wf"] = walk_forward_report(runs, eval_exists=load_run_evaluation, cfg=cfg)

wf = st.session_state.get("dashboard_wf")
if wf is None:
    st.info("点击上方按钮计算最新 Walk-Forward。")
elif wf.empty:
    st.warning("当前样本不足，暂时无法形成有效 Walk-Forward。")
else:
    st.metric("OOS 平均收益", f"{pd.to_numeric(wf['oos_mean_ret'], errors='coerce').mean():.4f}%")
    st.dataframe(wf, use_container_width=True)
    curve = wf[["test_run_id", "cum_oos_ret"]].copy().set_index("test_run_id")
    st.line_chart(curve)

st.subheader("当前默认权重与策略摘要")
w_col1, w_col2 = st.columns(2)
with w_col1:
    if DEFAULT_WEIGHTS_JSON.exists():
        data = pd.read_json(DEFAULT_WEIGHTS_JSON, typ="series")
        st.json(data.to_dict())
    else:
        st.info("暂无默认权重文件（ui_default_weights.json）。")
with w_col2:
    if STRATEGY_REPORT_CSV.exists():
        sr = pd.read_csv(STRATEGY_REPORT_CSV)
        st.dataframe(sr.tail(20), use_container_width=True)
    else:
        st.info("暂无 strategy_report.csv。")
