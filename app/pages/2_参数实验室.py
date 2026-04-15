from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.run_store import RUNS_CSV, list_runs, load_run_evaluation, save_default_weights
from core.weight_experiment import run_experiment

st.set_page_config(page_title="参数实验室", page_icon="🧪", layout="wide")
st.title("参数实验室")
st.caption("在历史已复评的 run 上回放不同权重组合，按风险调整收益排序。")

if not RUNS_CSV.exists() or list_runs().empty:
    st.info("暂无 run 或复评数据。请先完成筛选与复评中心的多周期更新。")
    st.stop()

runs = list_runs()
run_ids_all = runs["run_id"].astype(str).tolist()
ret_col = st.selectbox(
    "目标收益列（回放排序依据）",
    ["ret_close_t5", "ret_close_t3", "ret_close_t1", "ret_close_t10"],
    index=0,
)
top_n = st.slider("模拟组合容量（取重算得分前 N）", 3, 15, 5)
mode = st.selectbox("搜索方式", ["random", "grid", "both"], index=0)
n_random = st.slider("随机组数（mode 含 random 时）", 20, 200, 80, 10)

usable: list[str] = []
for rid in run_ids_all:
    ev = load_run_evaluation(rid)
    if ev is not None and not ev.empty and ret_col in ev.columns:
        usable.append(rid)

st.write(f"可用 run 数（含 `{ret_col}`）：**{len(usable)}** / {len(run_ids_all)}")

if len(usable) < 1:
    st.warning("没有已复评且含目标列的 run。请到复评中心先更新复评。")
    st.stop()

if st.button("运行权重实验", type="primary"):
    with st.spinner("计算中..."):
        df = run_experiment(
            usable,
            ret_col=ret_col,
            top_n=top_n,
            mode=mode,
            n_random=n_random,
            seed=42,
        )
    if df.empty:
        st.error("无有效结果。")
        st.session_state.pop("weight_leaderboard", None)
    else:
        st.session_state["weight_leaderboard"] = df

if "weight_leaderboard" in st.session_state:
    df = st.session_state["weight_leaderboard"]
    st.subheader("权重组合排行榜（按 risk_adjusted_score）")
    st.dataframe(df.head(30), use_container_width=True)
    top = df.iloc[0]
    st.success(
        f"推荐：题材 {top['w_theme']:.3f} / 板块 {top['w_sector']:.3f} / "
        f"个股 {top['w_stock']:.3f} / 资金 {top['w_capital']:.3f}"
    )
    if st.button("将推荐权重写入筛选页默认值（下次打开首页生效）"):
        save_default_weights(
            float(top["w_theme"]),
            float(top["w_sector"]),
            float(top["w_stock"]),
            float(top["w_capital"]),
        )
        st.success("已保存到 data/ui_default_weights.json，请返回首页刷新页面。")
