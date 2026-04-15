from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.run_store import RUNS_CSV, list_runs, load_run_evaluation, save_default_weights
from core.walk_forward import WalkForwardConfig, walk_forward_report
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

st.divider()
st.subheader("滚动样本外验证（Walk-Forward）")
st.caption("按时间顺序滚动：前 N 个 run 训练权重，第 N+1 个 run 做样本外验证。")

wf_ret_col = st.selectbox(
    "Walk-Forward 目标收益列",
    ["ret_close_t5_net", "ret_close_t3_net", "ret_close_t1_net", "ret_close_t5", "ret_close_t3", "ret_close_t1"],
    index=0,
)
wf_top_n = st.slider("Walk-Forward 组合容量（Top N）", 3, 15, 5)
wf_min_train = st.slider("最小训练 run 数", 1, 12, 3)
wf_mode = st.selectbox("Walk-Forward 搜索方式", ["random", "grid", "both"], index=0)
wf_n_random = st.slider("Walk-Forward 随机组数", 20, 300, 120, 10)

if st.button("运行 Walk-Forward 验证", type="secondary"):
    cfg = WalkForwardConfig(
        ret_col=wf_ret_col,
        top_n=wf_top_n,
        min_train_runs=wf_min_train,
        mode=wf_mode,
        n_random=wf_n_random,
        seed=42,
    )
    with st.spinner("正在执行滚动样本外验证..."):
        wf_df = walk_forward_report(runs, eval_exists=load_run_evaluation, cfg=cfg)
    st.session_state["walk_forward_report"] = wf_df

if "walk_forward_report" in st.session_state:
    wf_df = st.session_state["walk_forward_report"]
    if wf_df is None or wf_df.empty:
        st.warning("当前数据不足以形成有效 walk-forward 结果，请先积累更多已复评 run。")
    else:
        st.write(
            f"有效步数：**{len(wf_df)}** ｜ "
            f"OOS 平均收益：**{wf_df['oos_mean_ret'].mean():.4f}%** ｜ "
            f"OOS 平均胜率：**{wf_df['oos_win_rate'].mean():.4f}**"
        )
        st.dataframe(wf_df, use_container_width=True)
        curve = wf_df[["test_run_id", "cum_oos_ret"]].copy().set_index("test_run_id")
        st.line_chart(curve)
