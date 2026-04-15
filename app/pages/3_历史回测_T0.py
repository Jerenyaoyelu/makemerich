from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
APP_DIR = PROJECT_ROOT / "app"
for p in (PROJECT_ROOT, APP_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core.data_provider import fetch_historical_signals
from core.logger import get_logger, set_log_level
from core.run_store import load_default_weights
from core.scoring import score_frame
from ui_screener_shared import (
    SCORING_AND_PIPELINE_MARKDOWN,
    ensure_live_meta_columns,
    finalize_screener_run,
    normalize_weights,
)

OUTPUT_DIR = PROJECT_ROOT / "output"
logger = get_logger("ui.backtest_t0")

st.set_page_config(page_title="历史回测 T0", page_icon="📅", layout="wide")
st.title("历史回测 T0")
logger.info("打开页面: 历史回测 T0")
st.caption("用某一交易日的日线重建截面 → 与实时选股相同的打分逻辑 → 再去「复评中心」验 T+1、T+2…")

st.success(
    "**正确步骤（共 2 步）**：① 在下方侧边栏选好 **T0 日期** 与参数 → 点击 **「运行历史回测筛选」**。"
    " ② 到 **「复评中心」** 选中对应 run，点「更新复评数据」。"
    "**不要**去「实时选股」页点「刷新实时数据」——那条链路只服务最新行情。"
)

with st.expander("计算与打分原理（与实时选股相同）", expanded=False):
    st.markdown(SCORING_AND_PIPELINE_MARKDOWN)

defaults = load_default_weights()
w_def = defaults or {}

with st.sidebar:
    log_level = st.selectbox("日志级别", ["INFO", "DEBUG"], index=0)
    set_log_level(log_level)
    st.subheader("T0 与采集")
    historical_date = st.date_input(
        "回测交易日 T0（须为交易日）",
        value=date.today() - timedelta(days=5),
        help="系统会逐票请求该日的后复权日线，耗时与「扫描上限」成正比。",
    )
    hist_scan_cap = st.slider(
        "历史扫描上限（逐票请求次数上限）",
        min_value=100,
        max_value=8000,
        value=1200,
        step=100,
        help="上限越大，覆盖面越广但耗时越长；建议先用 800~1500。",
    )
    hist_request_delay = st.slider(
        "请求间隔（秒，略增可减轻断连）",
        min_value=0.03,
        max_value=0.35,
        value=0.1,
        step=0.01,
    )
    market_scope = st.selectbox(
        "板块范围",
        options=["全部A股", "沪深两市（主板）", "创业板", "科创板"],
        index=0,
    )
    _primary_src_labels = (
        "稳定优先（新浪→腾讯→东财）",
        "字段全优先（东财→新浪→腾讯）",
        "自动（按近期成功采集推断）",
    )
    primary_src_choice = st.radio(
        "日线主源顺序",
        _primary_src_labels,
        index=0,
        help="与实时页「主源策略」一致：自动读取 data/collection_history.csv。",
    )
    primary_source_strategy = {
        _primary_src_labels[0]: "stability_first",
        _primary_src_labels[1]: "completeness_first",
        _primary_src_labels[2]: "auto",
    }[primary_src_choice]
    refresh_limit = st.slider(
        "参与排序的股票数量上限",
        100,
        min(3000, int(hist_scan_cap)),
        min(300, int(hist_scan_cap)),
        50,
        help="这是最终进入排序池的数量（<= 扫描上限）。扫描上限更大可提升覆盖，但最终只保留前 N。",
    )

    st.subheader("评分参数")
    raw_theme = st.slider("题材强度权重", 0.0, 1.0, float(w_def.get("w_theme", 0.30)), 0.05)
    raw_sector = st.slider("板块联动权重", 0.0, 1.0, float(w_def.get("w_sector", 0.25)), 0.05)
    raw_stock = st.slider("个股强度权重", 0.0, 1.0, float(w_def.get("w_stock", 0.25)), 0.05)
    raw_capital = st.slider("资金承接权重", 0.0, 1.0, float(w_def.get("w_capital", 0.20)), 0.05)
    score_threshold = st.slider("最低入选分数", 0.0, 100.0, 70.0, 1.0)
    top_n = st.slider("展示候选数量", 1, 20, 10, 1)
    run_btn = st.button("运行历史回测筛选", type="primary")

w_theme, w_sector, w_stock, w_capital = normalize_weights(raw_theme, raw_sector, raw_stock, raw_capital)
st.write(
    f"当前归一化权重：题材 {w_theme:.2f} / 板块 {w_sector:.2f} / 个股 {w_stock:.2f} / 资金 {w_capital:.2f}"
)

if run_btn:
    logger.info(
        "点击按钮: 运行历史回测筛选 | T0=%s scope=%s scan_cap=%s delay=%.2f limit=%s threshold=%.2f top_n=%s",
        historical_date,
        market_scope,
        hist_scan_cap,
        hist_request_delay,
        refresh_limit,
        score_threshold,
        top_n,
    )
    progress = st.progress(0, text="准备拉取历史截面…")
    scope_map = {
        "全部A股": "all_a",
        "沪深两市（主板）": "hs_main",
        "创业板": "gem",
        "科创板": "star",
    }
    trade_d = historical_date.strftime("%Y-%m-%d")
    try:
        progress.progress(10, text=f"正在拉取 {trade_d} 历史截面（逐票日线）…")
        frame = fetch_historical_signals(
            trade_date=trade_d,
            limit=refresh_limit,
            market_scope=scope_map.get(market_scope, "all_a"),
            max_universe_scan=hist_scan_cap,
            request_delay_sec=hist_request_delay,
            primary_source_strategy=primary_source_strategy,
            progress_callback=lambda p, msg: progress.progress(max(10, min(90, int(p))), text=msg),
        )
        logger.info("历史截面采集成功: T0=%s rows=%s", trade_d, len(frame))
        st.info(f"已构建 T0={trade_d} 截面，共 {len(frame)} 条进入打分（截断上限 {refresh_limit}）。")
    except Exception as exc:
        progress.empty()
        logger.exception("历史截面采集失败: %s", exc)
        st.error(f"历史截面采集失败：{exc}")
        st.stop()

    frame = ensure_live_meta_columns(frame)
    progress.progress(92, text="正在计算评分与筛选…")
    scored = score_frame(frame, w_theme, w_sector, w_stock, w_capital)
    filtered = scored[scored["total_score"] >= score_threshold].head(top_n)
    logger.info("历史回测筛选完成: input_rows=%s filtered_rows=%s", len(scored), len(filtered))
    progress.progress(100, text="完成")

    ds_label = f"历史回测T0={trade_d}"
    finalize_screener_run(
        scored,
        filtered,
        w_theme=w_theme,
        w_sector=w_sector,
        w_stock=w_stock,
        w_capital=w_capital,
        score_threshold=score_threshold,
        top_n=top_n,
        market_scope=market_scope,
        refresh_limit=refresh_limit,
        data_source_label=ds_label,
        output_dir=OUTPUT_DIR,
    )
    logger.info("历史回测 run 已落盘并展示完成: T0=%s", trade_d)

st.divider()
st.markdown(
    """
    ### 说明
    - 快照时间固定为 **T0 日 15:00:00**，复评脚本以此为基准计算后续交易日表现。  
    - 日线无「量比」，与新浪备源一致按缺失处理。  
    - 若频繁 `RemoteDisconnected`，请增大请求间隔、暂时关闭代理，或降低扫描上限。
    """
)
