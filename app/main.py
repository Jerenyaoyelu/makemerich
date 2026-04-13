from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.data_provider import fetch_live_signals, load_sample_signals
from core.scoring import score_frame

SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "sample_stocks.csv"
LIVE_DATA_PATH = PROJECT_ROOT / "data" / "latest_signals.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"


def normalize_weights(w1: float, w2: float, w3: float, w4: float) -> tuple[float, float, float, float]:
    total = w1 + w2 + w3 + w4
    if total <= 0:
        return 0.25, 0.25, 0.25, 0.25
    return w1 / total, w2 / total, w3 / total, w4 / total


def main() -> None:
    st.set_page_config(page_title="程序化选股助手", page_icon="📈", layout="wide")
    st.title("程序化选股助手")
    st.caption("策略：事件驱动龙头的分歧转一致（自动采集 + 自动打分）")

    with st.sidebar:
        st.subheader("数据源")
        data_source = st.radio("选择数据来源", ["自动采集（推荐）", "样例数据（离线演示）"], index=0)
        refresh_limit = st.slider("自动采集股票数量上限", 100, 1000, 300, 50)
        refresh_btn = st.button("刷新实时数据")
        use_local_btn = st.button("手动加载本地缓存数据")

        st.subheader("评分参数")
        raw_theme = st.slider("题材强度权重", 0.0, 1.0, 0.30, 0.05)
        raw_sector = st.slider("板块联动权重", 0.0, 1.0, 0.25, 0.05)
        raw_stock = st.slider("个股强度权重", 0.0, 1.0, 0.25, 0.05)
        raw_capital = st.slider("资金承接权重", 0.0, 1.0, 0.20, 0.05)
        score_threshold = st.slider("最低入选分数", 0.0, 100.0, 70.0, 1.0)
        top_n = st.slider("展示候选数量", 1, 20, 10, 1)
        run_btn = st.button("运行筛选", type="primary")

    w_theme, w_sector, w_stock, w_capital = normalize_weights(
        raw_theme, raw_sector, raw_stock, raw_capital
    )

    st.write(
        f"当前归一化权重：题材 {w_theme:.2f} / 板块 {w_sector:.2f} / 个股 {w_stock:.2f} / 资金 {w_capital:.2f}"
    )

    if refresh_btn:
        try:
            live = fetch_live_signals(limit=refresh_limit)
            live.to_csv(LIVE_DATA_PATH, index=False, encoding="utf-8-sig")
            st.success(f"自动采集完成：{LIVE_DATA_PATH}（{len(live)} 条）")
        except Exception as exc:
            st.error(f"自动采集失败：{exc}")
            st.info("系统不会自动回退。你可以点击“手动加载本地缓存数据”或切换“样例数据（离线演示）”。")

    if use_local_btn:
        if LIVE_DATA_PATH.exists():
            st.success(f"已选择本地缓存数据：{LIVE_DATA_PATH}")
            st.session_state["force_local_file"] = True
        else:
            st.error(f"本地缓存文件不存在：{LIVE_DATA_PATH}")

    if run_btn:
        frame: pd.DataFrame
        if data_source == "自动采集（推荐）":
            force_local = st.session_state.get("force_local_file", False)
            if force_local:
                if not LIVE_DATA_PATH.exists():
                    st.error(f"你选择了本地缓存数据，但文件不存在：{LIVE_DATA_PATH}")
                    return
                frame = pd.read_csv(LIVE_DATA_PATH)
                st.info("当前使用本地缓存数据（手动选择）。")
                st.session_state["force_local_file"] = False
            else:
                try:
                    frame = fetch_live_signals(limit=refresh_limit)
                    frame.to_csv(LIVE_DATA_PATH, index=False, encoding="utf-8-sig")
                except Exception as exc:
                    st.error(f"实时采集失败：{exc}")
                    st.info("请先处理网络/代理问题，或点击“手动加载本地缓存数据”。")
                    return
        else:
            frame = load_sample_signals(SAMPLE_DATA_PATH)

        scored = score_frame(frame, w_theme, w_sector, w_stock, w_capital)
        filtered = scored[scored["total_score"] >= score_threshold].head(top_n)

        st.subheader("候选股结果")
        st.dataframe(filtered, use_container_width=True)
        st.metric("入选数量", value=len(filtered))

        if not filtered.empty:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            out_path = OUTPUT_DIR / f"candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filtered.to_csv(out_path, index=False, encoding="utf-8-sig")
            st.success(f"已导出结果：{out_path}")
        else:
            st.warning("当前阈值下无入选标的，可尝试降低分数阈值。")

        with st.expander("指标含义与来源（点击展开）", expanded=False):
            st.markdown(
                """
                - `theme_strength`：题材强度，来自行业平均涨幅 + 行业成交额（归一化到 0-100）  
                - `sector_linkage`：板块联动，来自行业平均涨幅 + 行业内上涨家数占比  
                - `stock_strength`：个股强度，来自涨跌幅 + 量比 + 振幅  
                - `capital_support`：资金承接，来自换手率 + 成交额 + 量比  
                - `risk_tag`：波动标签，根据振幅与换手率分为低/中/高波动  
                - 原始字段 `pct_chg`、`turnover`、`amount`、`volume_ratio`、`amplitude` 直接来自实时行情
                """
            )

    st.divider()
    st.markdown(
        """
        ### 使用建议
        - 第一阶段只做辅助决策，不要自动下单  
        - 每天先点“刷新实时数据”，再点“运行筛选”  
        - 每周复盘一次：高分股为什么成功/失败  
        - 按你的交易日志持续微调权重和阈值
        """
    )


if __name__ == "__main__":
    main()

