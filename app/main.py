from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
for p in (PROJECT_ROOT, APP_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core.data_provider import fetch_live_signals, load_sample_signals
from core.logger import get_logger, set_log_level
from core.run_store import append_collection_history, load_default_weights
from core.scoring import score_frame
from ui_screener_shared import (
    SCORING_AND_PIPELINE_MARKDOWN,
    ensure_live_meta_columns,
    finalize_screener_run,
    normalize_weights,
)

SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "sample_stocks.csv"
LIVE_DATA_PATH = PROJECT_ROOT / "data" / "latest_signals.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"
logger = get_logger("ui.main")


def main() -> None:
    logger.info("打开页面: 实时选股")
    st.set_page_config(page_title="实时选股（执行页）", page_icon="📈", layout="wide")
    if not st.session_state.get("_home_redirected_to_dashboard", False):
        st.session_state["_home_redirected_to_dashboard"] = True
        try:
            st.switch_page("pages/0_系统驾驶舱.py")
        except Exception:
            # 旧版 Streamlit 不支持 switch_page 时继续显示执行页
            pass

    st.title("实时选股（执行页）")
    st.caption("最新行情快照 → 打分 → 候选池（与「历史回测 T0」互不混用）")

    st.info(
        "**历史某日回测**（选过去交易日、直接拉日线做 T0，再去复评中心验 T+1…）"
        "请打开左侧页面 **「历史回测 T0」**，不要点本页的「刷新实时数据」。"
    )

    with st.expander("计算与打分原理（与 `core/data_provider.py`、`core/scoring.py` 一致）", expanded=False):
        st.markdown(SCORING_AND_PIPELINE_MARKDOWN)

    defaults = load_default_weights()
    w_def = defaults or {}

    with st.sidebar:
        if st.button("前往系统驾驶舱（首页）"):
            try:
                st.switch_page("pages/0_系统驾驶舱.py")
            except Exception:
                st.info("当前环境不支持页面跳转，请在左侧导航手动切换到「系统驾驶舱」。")
        log_level = st.selectbox("日志级别", ["INFO", "DEBUG"], index=0)
        set_log_level(log_level)
        st.subheader("数据源（仅实时 / 样例）")
        data_source = st.radio(
            "选择数据来源",
            ["自动采集（推荐）", "样例数据（离线演示）"],
            index=0,
            help="历史回测请切换页面「历史回测 T0」。",
        )
        market_scope = st.selectbox(
            "采集范围",
            options=["全部A股", "沪深两市（主板）", "创业板", "科创板"],
            index=0,
        )
        _primary_src_labels = (
            "稳定优先（新浪→东财）",
            "字段全优先（东财→新浪）",
            "自动（按近期成功采集推断）",
        )
        primary_src_choice = st.radio(
            "主源策略",
            _primary_src_labels,
            index=0,
            help="新浪通常更稳；东财字段更全。自动策略读取 data/collection_history.csv 中近期成功记录。",
        )
        primary_source_strategy = {
            _primary_src_labels[0]: "stability_first",
            _primary_src_labels[1]: "completeness_first",
            _primary_src_labels[2]: "auto",
        }[primary_src_choice]
        refresh_limit = st.slider("自动采集股票数量上限", 100, 1000, 300, 50)
        refresh_btn = st.button("刷新实时数据", help="写入 data/latest_signals.csv，供本次或下次筛选使用。")
        use_local_btn = st.button("手动加载本地缓存数据")

        st.subheader("评分参数")
        raw_theme = st.slider(
            "题材强度权重", 0.0, 1.0, float(w_def.get("w_theme", 0.30)), 0.05
        )
        raw_sector = st.slider(
            "板块联动权重", 0.0, 1.0, float(w_def.get("w_sector", 0.25)), 0.05
        )
        raw_stock = st.slider(
            "个股强度权重", 0.0, 1.0, float(w_def.get("w_stock", 0.25)), 0.05
        )
        raw_capital = st.slider(
            "资金承接权重", 0.0, 1.0, float(w_def.get("w_capital", 0.20)), 0.05
        )
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
        logger.info("点击按钮: 刷新实时数据 | scope=%s | limit=%s", market_scope, refresh_limit)
        progress = st.progress(0, text="准备刷新实时数据...")
        try:
            scope_map = {
                "全部A股": "all_a",
                "沪深两市（主板）": "hs_main",
                "创业板": "gem",
                "科创板": "star",
            }
            live = fetch_live_signals(
                limit=refresh_limit,
                market_scope=scope_map.get(market_scope, "all_a"),
                progress_callback=lambda p, msg: progress.progress(p, text=msg),
                primary_source_strategy=primary_source_strategy,
            )
            progress.progress(75, text="正在写入本地数据文件...")
            LIVE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            live.to_csv(LIVE_DATA_PATH, index=False, encoding="utf-8-sig")
            progress.progress(100, text="实时数据刷新完成")
            src = str(live["source"].iloc[0]) if "source" in live.columns else "unknown"
            append_collection_history(True, src)
            logger.info("刷新实时数据成功: source=%s rows=%s", src, len(live))
            st.success(f"自动采集完成：{LIVE_DATA_PATH}（{len(live)} 条）")
        except Exception as exc:
            progress.empty()
            append_collection_history(False, "error", str(exc))
            logger.exception("刷新实时数据失败: %s", exc)
            st.error(f"刷新失败：{exc}")
            st.info("可点击「手动加载本地缓存数据」或改用样例数据。")

    if use_local_btn:
        logger.info("点击按钮: 手动加载本地缓存数据")
        if LIVE_DATA_PATH.exists():
            st.success(f"已选择本地缓存数据：{LIVE_DATA_PATH}")
            st.session_state["force_local_file"] = True
        else:
            st.error(f"本地缓存文件不存在：{LIVE_DATA_PATH}")

    if run_btn:
        logger.info(
            "点击按钮: 运行筛选 | data_source=%s scope=%s refresh_limit=%s threshold=%.2f top_n=%s",
            data_source,
            market_scope,
            refresh_limit,
            score_threshold,
            top_n,
        )
        progress = st.progress(0, text="准备运行筛选...")
        frame: pd.DataFrame
        if data_source == "自动采集（推荐）":
            force_local = st.session_state.get("force_local_file", False)
            if force_local:
                if not LIVE_DATA_PATH.exists():
                    progress.empty()
                    st.error(f"你选择了本地缓存数据，但文件不存在：{LIVE_DATA_PATH}")
                    return
                progress.progress(30, text="正在加载本地缓存数据...")
                frame = ensure_live_meta_columns(pd.read_csv(LIVE_DATA_PATH))
                logger.info("使用本地缓存数据运行筛选: path=%s rows=%s", LIVE_DATA_PATH, len(frame))
                st.info("当前使用本地缓存数据（手动选择）。")
                st.session_state["force_local_file"] = False
            else:
                try:
                    scope_map = {
                        "全部A股": "all_a",
                        "沪深两市（主板）": "hs_main",
                        "创业板": "gem",
                        "科创板": "star",
                    }
                    progress.progress(20, text="正在获取筛选输入数据...")
                    frame = fetch_live_signals(
                        limit=refresh_limit,
                        market_scope=scope_map.get(market_scope, "all_a"),
                        progress_callback=lambda p, msg: progress.progress(p, text=f"筛选前数据准备：{msg}"),
                        primary_source_strategy=primary_source_strategy,
                    )
                    progress.progress(60, text="正在写入缓存文件...")
                    LIVE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
                    frame.to_csv(LIVE_DATA_PATH, index=False, encoding="utf-8-sig")
                    src = str(frame["source"].iloc[0]) if "source" in frame.columns else "unknown"
                    append_collection_history(True, src)
                    logger.info("运行筛选前实时采集成功: source=%s rows=%s", src, len(frame))
                except Exception as exc:
                    progress.empty()
                    append_collection_history(False, "error", str(exc))
                    logger.exception("运行筛选前实时采集失败: %s", exc)
                    st.error(f"实时采集失败：{exc}")
                    st.info("请先处理网络/代理问题，或点击「手动加载本地缓存数据」。")
                    return
        else:
            progress.progress(30, text="正在加载样例数据...")
            frame = ensure_live_meta_columns(load_sample_signals(SAMPLE_DATA_PATH))

        frame = ensure_live_meta_columns(frame)
        progress.progress(80, text="正在计算评分与筛选...")
        scored = score_frame(frame, w_theme, w_sector, w_stock, w_capital)
        filtered = scored[scored["total_score"] >= score_threshold].head(top_n)
        logger.info("筛选计算完成: input_rows=%s filtered_rows=%s", len(scored), len(filtered))
        progress.progress(100, text="筛选完成")

        ds_label = "自动采集" if data_source == "自动采集（推荐）" else "样例数据"
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
        logger.info("run 已落盘并展示完成")

    st.divider()
    st.markdown(
        """
        ### 使用建议（实时）
        - 需要最新行情时：可先 **刷新实时数据**，再 **运行筛选**（也可直接运行筛选，会现采现算）。  
        - **历史回测**请用左侧 **「历史回测 T0」**，勿与本页「刷新实时数据」混用。  
        - 每周复盘高分股得失，微调权重与阈值。
        """
    )


if __name__ == "__main__":
    main()
