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
from core.health_score import compute_health_score
from core.run_store import (
    append_collection_history,
    append_run,
    load_default_weights,
    new_run_id,
    recent_collection_success_rate,
    save_run_candidates,
)
from core.scoring import score_frame

SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "sample_stocks.csv"
LIVE_DATA_PATH = PROJECT_ROOT / "data" / "latest_signals.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"


def _ensure_live_meta_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """样例数据等缺少快照/来源字段时补齐，便于 run 持久化与健康度。"""
    out = frame.copy()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if "snapshot_time" not in out.columns:
        out["snapshot_time"] = now
    if "source" not in out.columns:
        out["source"] = "sample_offline"
    if "theme_source" not in out.columns:
        out["theme_source"] = "spot"
    if "turnover_source" not in out.columns:
        out["turnover_source"] = "spot"
    if "volume_ratio_source" not in out.columns:
        out["volume_ratio_source"] = "spot"
    return out

# 仅用于界面展示；导出 CSV 仍保留英文字段名，便于脚本与复盘对接
DISPLAY_COLUMN_LABELS: dict[str, str] = {
    "symbol": "代码",
    "name": "名称",
    "theme": "行业（题材代理）",
    "total_score": "综合得分",
    "theme_strength": "题材强度",
    "sector_linkage": "板块联动",
    "stock_strength": "个股强度",
    "capital_support": "资金承接",
    "risk_tag": "波动标签",
    "pct_chg": "涨跌幅(%)",
    "turnover": "换手率(%)",
    "amount": "成交额",
    "volume_ratio": "量比",
    "amplitude": "振幅(%)",
    "industry_mean_pct": "行业平均涨跌幅(%)",
    "industry_up_ratio": "行业上涨家数占比(%)",
    "snapshot_time": "快照时间",
    "source": "数据来源",
    "theme_source": "行业来源",
    "turnover_source": "换手来源",
    "volume_ratio_source": "量比来源",
}

DISPLAY_COLUMN_ORDER: list[str] = [
    "symbol",
    "name",
    "theme",
    "total_score",
    "theme_strength",
    "sector_linkage",
    "stock_strength",
    "capital_support",
    "risk_tag",
    "pct_chg",
    "turnover",
    "amount",
    "volume_ratio",
    "amplitude",
    "industry_mean_pct",
    "industry_up_ratio",
    "snapshot_time",
    "source",
    "theme_source",
    "turnover_source",
    "volume_ratio_source",
]

# 写入 CSV 仍为英文键；界面展示为中文说明
SOURCE_DISPLAY_LABELS: dict[str, str] = {
    "eastmoney_em": "东方财富（含换手、量比、振幅等，字段较全）",
    "sina_finance": "新浪财经（含换手；接口不含量比，表中显示「—」，打分中间量比按中性处理）",
    "stock_zh_a_spot": "新浪财经·旧版标识（换手/量比可能曾不准确，建议重新刷新数据）",
}
FIELD_SOURCE_LABELS: dict[str, str] = {
    "spot": "实时源原始字段",
    "industry_map_cache": "行业映射缓存回填",
    "unknown": "未知",
    "estimated_by_amount_nmc": "成交额/流通市值估算",
    "missing": "缺失",
}


def _format_cell_missing_number(val: object) -> object:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    if isinstance(val, (int, float)):
        return round(float(val), 4)
    return val


SCORING_AND_PIPELINE_MARKDOWN = """
### 一、整体在算什么

系统做的是**同一时刻快照上的相对排序**：先把行情与行业统计合成几个**原始特征**，再在**本次参与计算的全部股票范围内**做 **0–100 的线性归一化**，得到四个因子分；最后用你在侧边栏设的权重做加权平均得到**综合得分**。  
它不是「绝对基本面评分」，也不含分钟级「分歧转一致」触发逻辑；权重与阈值需要你用实盘结果慢慢校准。

---

### 二、数据从哪来、先做什么清洗

1. **主源**：东方财富 `stock_zh_a_spot_em`（字段较全，含行业、换手、量比、振幅等）。  
2. **备源**：新浪财经分页接口（与 AkShare 同源），**保留换手率**；**官方无量比**，导出为缺失，**打分内部**将量比按 **1.0** 参与合成（中性，避免所有人被同一个假常数绑死）。  
3. **清洗**：名称含 **ST** 的剔除；**成交额 ≤ 0** 的剔除。  
4. **行业**：用行情里的「所属行业」作为 `theme`；若数据源没有行业列，则全部为 **「未知行业」**（此时行业类因子会退化成「全市场一大组」的统计，解释力会弱很多）。

---

### 三、行业维度（分组对象 = `theme`）

对每只股票，按当日快照在**其所属行业**内聚合（与代码一致）：

| 中间变量 | 含义 |
|----------|------|
| **行业平均涨跌幅** `industry_mean_pct` | 该行业内所有样本涨跌幅的**算术平均**（%） |
| **行业上涨家数占比** `industry_up_ratio` | 该行业内 `涨跌幅 > 0` 的样本占比 × **100**（可理解为 0–100 的分数，不是小数） |
| **行业总成交额** `industry_amount` | 该行业内成交额之和 |

---

### 四、归一化（四个因子共用的「0–100」）

对任意一列原始合成值 `x`，在**本次快照里、清洗后保留的全部股票**上：

```
y = clip( (x - x.min) / (x.max - x.min) * 100, 0, 100 )
```

若全表几乎相等（`max - min` 极小），则该列所有人记为 **50**。

含义：**高分 = 在这一批股票里相对更突出**，换一批股票或换一个交易日，分数不可横向直接对比历史绝对水平。

---

### 五、四个因子（先算原始线性组合，再归一化）

记 `log1p(x)` = **ln(1 + x)**。**涨跌幅、振幅、换手率**等与行情源单位一致（通常为 **%**）；**成交额**为源数据绝对值。

**1. 题材强度 `theme_strength`**（偏「行业有多强 + 有多大钱」）

```
raw = industry_mean_pct * 0.6 + log1p(industry_amount) * 0.4
```

再对 `raw` 做上面的0–100 归一化。

**2. 板块联动 `sector_linkage`**（偏「行业普涨氛围」）

```
raw = industry_mean_pct * 0.5 + industry_up_ratio * 0.5
```

再归一化。

**3. 个股强度 `stock_strength`**（偏「自身涨得多不多 + 量能与波动」）

- 令 **量比参与打分** `vol_for_score` = 有则取源数据，**缺失则 1.0**（新浪备源）。

```
raw = pct_chg * 0.5 + vol_for_score * 22 + amplitude * 0.2
```

再归一化。

**4. 资金承接 `capital_support`**（偏「换手与成交额是否活跃」）

- 换手在合成里缺失时按 **0**；量比仍用上面的 `vol_for_score`。

```
raw = turnover * 0.35 + log1p(amount) * 6 + vol_for_score * 35
```

再归一化。

系数 **0.6/0.4、22、35** 等是**经验加权**，用来把不同量纲捏到同一数量级后再归一化；若要严肃迭代，应固定规则用日志回测再改。

---

### 六、综合得分（侧边栏权重）

侧边栏四个滑块会先**归一化**为 `w_theme + w_sector + w_stock + w_capital = 1`，再：

```
total_score = theme_strength * w_theme + sector_linkage * w_sector
            + stock_strength * w_stock + capital_support * w_capital
```

保留 **两位小数**，并按 `total_score` **降序**排序。

---

### 七、入选池是怎么截断的（容易忽略）

在算综合得分之前，程序会先把全市场结果按 **个股强度 `stock_strength`（未加权）** 排序，只保留前 **「自动采集股票数量上限」** 条，再在这批里算总分。  
因此：**综合得分只在这批「个股强度靠前」的票里排序**；若你认为龙头应在更大范围里比，需要把上限调大。

---

### 八、波动标签 `risk_tag`（不参与加权得分）

用 **振幅** 与 **换手率**（换手缺失时按 **0**）：

- **高波动**：振幅 ≥ **12** 或 换手 ≥ **18**  
- 否则若振幅 ≥ **7** 或 换手 ≥ **10** → **中波动**  
- 否则 **低波动**

---

### 九、使用上要注意什么

- **归一化是相对排名**，不是「这只股票永远值 80 分」。  
- **行业字段**缺失时，题材/联动类因子会失真，优先保证东财主源可用。  
- **新浪无量比**时，个股强度/资金承接里「量比」项退化为中性1.0，更依赖涨跌幅、振幅、成交额、换手。  
- 若要贴近你文档里的「分歧转一致」，需要后续加分钟线/触发条件；当前版本**只有基于单次全市场实时快照合成的因子**，没有分时形态识别。
"""


def dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    ordered = [c for c in DISPLAY_COLUMN_ORDER if c in df.columns]
    ordered += [c for c in df.columns if c not in ordered]
    out = df[ordered].copy()
    rename = {c: DISPLAY_COLUMN_LABELS[c] for c in ordered if c in DISPLAY_COLUMN_LABELS}
    out = out.rename(columns=rename)
    if "数据来源" in out.columns:
        out["数据来源"] = out["数据来源"].map(
            lambda x: SOURCE_DISPLAY_LABELS.get(str(x), str(x)) if pd.notna(x) else "—"
        )
    for zh_col in ("行业来源", "换手来源", "量比来源"):
        if zh_col in out.columns:
            out[zh_col] = out[zh_col].map(
                lambda x: FIELD_SOURCE_LABELS.get(str(x), str(x)) if pd.notna(x) else "—"
            )
    for zh_col in ("换手率(%)", "量比"):
        if zh_col in out.columns:
            out[zh_col] = out[zh_col].map(_format_cell_missing_number)
    return out


def normalize_weights(w1: float, w2: float, w3: float, w4: float) -> tuple[float, float, float, float]:
    total = w1 + w2 + w3 + w4
    if total <= 0:
        return 0.25, 0.25, 0.25, 0.25
    return w1 / total, w2 / total, w3 / total, w4 / total


def render_data_quality_panel(frame: pd.DataFrame) -> None:
    total = len(frame)
    if total == 0:
        st.warning("当前无可用数据，无法生成数据质量诊断。")
        return

    def pct(expr: pd.Series) -> float:
        return float(expr.mean() * 100)

    theme_real = pct(frame.get("theme_source", pd.Series([], dtype=str)) == "spot")
    theme_filled = pct(frame.get("theme_source", pd.Series([], dtype=str)) == "industry_map_cache")
    theme_unknown = pct(frame.get("theme_source", pd.Series([], dtype=str)) == "unknown")

    turnover_real = pct(frame.get("turnover_source", pd.Series([], dtype=str)) == "spot")
    turnover_est = pct(frame.get("turnover_source", pd.Series([], dtype=str)) == "estimated_by_amount_nmc")
    turnover_missing = pct(frame.get("turnover", pd.Series([], dtype=float)).isna())

    vr_real = pct(frame.get("volume_ratio_source", pd.Series([], dtype=str)) == "spot")
    vr_missing = pct(frame.get("volume_ratio", pd.Series([], dtype=float)).isna())

    c1, c2, c3 = st.columns(3)
    c1.metric("行业真实覆盖率", f"{theme_real:.1f}%")
    c2.metric("换手真实覆盖率", f"{turnover_real:.1f}%")
    c3.metric("量比真实覆盖率", f"{vr_real:.1f}%")

    with st.expander("数据可用性诊断（评分可信度）", expanded=True):
        st.markdown(
            f"""
            - **行业来源**：实时 `{theme_real:.1f}%` / 缓存回填 `{theme_filled:.1f}%` / 未知 `{theme_unknown:.1f}%`
            - **换手来源**：实时 `{turnover_real:.1f}%` / 估算 `{turnover_est:.1f}%` / 缺失 `{turnover_missing:.1f}%`
            - **量比来源**：实时 `{vr_real:.1f}%` / 缺失 `{vr_missing:.1f}%`
            """
        )

        warnings: list[str] = []
        if theme_unknown > 30:
            warnings.append("行业未知占比偏高，题材强度/板块联动解释力下降。")
        if turnover_est > 40:
            warnings.append("换手估算占比偏高，资金承接因子稳定性下降。")
        if vr_missing > 60:
            warnings.append("量比缺失占比很高，个股强度/资金承接对量比依赖已被弱化。")

        if warnings:
            for msg in warnings:
                st.warning(msg)
        else:
            st.success("当前字段覆盖率较好，评分可用性正常。")


def build_review_template(filtered: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["symbol", "name", "theme", "total_score", "snapshot_time", "source"] if c in filtered.columns]
    base = filtered[cols].copy()
    base["next_day_open"] = pd.NA
    base["next_day_close"] = pd.NA
    base["next_day_high"] = pd.NA
    base["next_day_low"] = pd.NA
    base["next_day_close_pct_vs_today_close"] = pd.NA
    base["hit_tag"] = ""
    base["review_notes"] = ""
    return base


def main() -> None:
    st.set_page_config(page_title="程序化选股助手", page_icon="📈", layout="wide")
    st.title("程序化选股助手")
    st.caption("策略：事件驱动龙头的分歧转一致（自动采集 + 自动打分）")

    with st.expander("计算与打分原理（与 `core/data_provider.py`、`core/scoring.py` 一致）", expanded=False):
        st.markdown(SCORING_AND_PIPELINE_MARKDOWN)

    defaults = load_default_weights()
    w_def = defaults or {}

    with st.sidebar:
        st.subheader("数据源")
        data_source = st.radio("选择数据来源", ["自动采集（推荐）", "样例数据（离线演示）"], index=0)
        market_scope = st.selectbox(
            "采集范围",
            options=["全部A股", "沪深两市（主板）", "创业板", "科创板"],
            index=0,
        )
        refresh_limit = st.slider("自动采集股票数量上限", 100, 1000, 300, 50)
        refresh_btn = st.button("刷新实时数据")
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
            )
            progress.progress(75, text="正在写入本地数据文件...")
            LIVE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            live.to_csv(LIVE_DATA_PATH, index=False, encoding="utf-8-sig")
            progress.progress(100, text="实时数据刷新完成")
            src = str(live["source"].iloc[0]) if "source" in live.columns else "unknown"
            append_collection_history(True, src)
            st.success(f"自动采集完成：{LIVE_DATA_PATH}（{len(live)} 条）")
        except Exception as exc:
            progress.empty()
            append_collection_history(False, "error", str(exc))
            st.info("系统不会自动回退。你可以点击“手动加载本地缓存数据”或切换“样例数据（离线演示）”。")

    if use_local_btn:
        if LIVE_DATA_PATH.exists():
            st.success(f"已选择本地缓存数据：{LIVE_DATA_PATH}")
            st.session_state["force_local_file"] = True
        else:
            st.error(f"本地缓存文件不存在：{LIVE_DATA_PATH}")

    if run_btn:
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
                frame = _ensure_live_meta_columns(pd.read_csv(LIVE_DATA_PATH))
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
                    )
                    progress.progress(60, text="正在写入缓存文件...")
                    LIVE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
                    frame.to_csv(LIVE_DATA_PATH, index=False, encoding="utf-8-sig")
                    src = str(frame["source"].iloc[0]) if "source" in frame.columns else "unknown"
                    append_collection_history(True, src)
                except Exception as exc:
                    progress.empty()
                    append_collection_history(False, "error", str(exc))
                    st.error(f"实时采集失败：{exc}")
                    st.info("请先处理网络/代理问题，或点击“手动加载本地缓存数据”。")
                    return
        else:
            progress.progress(30, text="正在加载样例数据...")
            frame = _ensure_live_meta_columns(load_sample_signals(SAMPLE_DATA_PATH))

        frame = _ensure_live_meta_columns(frame)
        progress.progress(80, text="正在计算评分与筛选...")
        scored = score_frame(frame, w_theme, w_sector, w_stock, w_capital)
        filtered = scored[scored["total_score"] >= score_threshold].head(top_n)
        progress.progress(100, text="筛选完成")

        if "theme_source" in scored.columns and (scored["theme_source"] == "unknown").mean() > 0.7:
            st.info("当前处于无行业模式（行业字段覆盖不足），建议降低题材强度/板块联动权重，先验证主流程。")

        source_used = str(scored["source"].iloc[0]) if "source" in scored.columns else "unknown"
        degraded = source_used != "eastmoney_em"
        recent_rate = recent_collection_success_rate()
        health_score, health_notes, _health_detail = compute_health_score(
            scored,
            source_used=source_used,
            degraded=degraded,
            recent_collection_rate=recent_rate,
        )

        render_data_quality_panel(scored)

        run_id = new_run_id()
        scope_label = market_scope
        ds_label = "自动采集" if data_source == "自动采集（推荐）" else "样例数据"
        append_run(
            run_id,
            w_theme=w_theme,
            w_sector=w_sector,
            w_stock=w_stock,
            w_capital=w_capital,
            score_threshold=score_threshold,
            top_n=top_n,
            market_scope=scope_label,
            refresh_limit=refresh_limit,
            data_source=ds_label,
            source_used=source_used,
            degraded=degraded,
            health_score=health_score,
            health_notes=health_notes,
            candidate_count=len(filtered),
            track_eval=True,
        )
        save_run_candidates(run_id, filtered)

        st.subheader("本次运行（闭环）")
        c_run1, c_run2, c_run3 = st.columns(3)
        c_run1.metric("run_id", run_id)
        c_run2.metric("数据健康分", f"{health_score}")
        c_run3.metric("备源降级", "是" if degraded else "否")
        st.caption(
            f"当前数据源：{SOURCE_DISPLAY_LABELS.get(source_used, source_used)} ｜ "
            f"持续复评：已纳入（候选已写入 data/run_candidates）｜ "
            f"健康说明：{health_notes}"
        )
        if degraded:
            st.warning("当前未使用东财主源，量比等字段可能缺失，评分已按规则降权；建议网络恢复后重采。")

        st.subheader("候选股结果")
        st.dataframe(dataframe_for_display(filtered), use_container_width=True)
        st.metric("入选数量", value=len(filtered))

        if not filtered.empty:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = OUTPUT_DIR / f"candidates_{ts}.csv"
            filtered.to_csv(out_path, index=False, encoding="utf-8-sig")
            review_path = OUTPUT_DIR / f"candidates_review_template_{ts}.csv"
            build_review_template(filtered).to_csv(review_path, index=False, encoding="utf-8-sig")
            st.success(f"已导出结果：{out_path}")
            st.info(f"已生成复评模板：{review_path}")
        else:
            st.warning("当前阈值下无入选标的，可尝试降低分数阈值。")

        with st.expander("表头字段速查（点击展开）", expanded=False):
            st.markdown(
                """
                详细公式见页面上方 **「计算与打分原理」**。

                - **题材强度 / 板块联动 / 个股强度 / 资金承接**：均为 0–100 的截面归一化因子（见上文第四节、第五节）  
                - **综合得分**：四项因子 × 侧边栏权重（见上文第六节）  
                - **波动标签**：振幅与换手阈值分档（见上文第八节），**不计入**综合得分  
                - **涨跌幅、换手、成交额、量比、振幅**：行情源字段；新浪无量比时表中为「—」  
                - **行业平均涨跌幅、行业上涨家数占比**：按「行业（题材代理）」分组统计  
                - **数据来源**：东方财富或新浪财经（见上文第二节）
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

