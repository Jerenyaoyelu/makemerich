from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.data_provider import load_sample_signals
from core.scoring import score_signals

SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "sample_stocks.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"


def normalize_weights(w1: float, w2: float, w3: float, w4: float) -> tuple[float, float, float, float]:
    total = w1 + w2 + w3 + w4
    if total <= 0:
        return 0.25, 0.25, 0.25, 0.25
    return w1 / total, w2 / total, w3 / total, w4 / total


def main() -> None:
    st.set_page_config(page_title="程序化选股助手", page_icon="📈", layout="wide")
    st.title("程序化选股助手（MVP）")
    st.caption("策略：事件驱动龙头的分歧转一致（辅助决策版）")

    with st.sidebar:
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

    if run_btn:
        signals = load_sample_signals(SAMPLE_DATA_PATH)
        scored = score_signals(signals, w_theme, w_sector, w_stock, w_capital)
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

    st.divider()
    st.markdown(
        """
        ### 使用建议
        - 第一阶段只做辅助决策，不要自动下单  
        - 每周复盘一次：高分股为什么成功/失败  
        - 逐步接入真实数据源替换样例 CSV
        """
    )


if __name__ == "__main__":
    main()

