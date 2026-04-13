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
]

# 写入 CSV 仍为英文键；界面展示为中文说明
SOURCE_DISPLAY_LABELS: dict[str, str] = {
    "eastmoney_em": "东方财富（含换手、量比、振幅等，字段较全）",
    "sina_finance": "新浪财经（含换手；接口不含量比，表中显示「—」，打分中间量比按中性处理）",
    "stock_zh_a_spot": "新浪财经·旧版标识（换手/量比可能曾不准确，建议重新刷新数据）",
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
    for zh_col in ("换手率(%)", "量比"):
        if zh_col in out.columns:
            out[zh_col] = out[zh_col].map(_format_cell_missing_number)
    return out


def normalize_weights(w1: float, w2: float, w3: float, w4: float) -> tuple[float, float, float, float]:
    total = w1 + w2 + w3 + w4
    if total <= 0:
        return 0.25, 0.25, 0.25, 0.25
    return w1 / total, w2 / total, w3 / total, w4 / total


def main() -> None:
    st.set_page_config(page_title="程序化选股助手", page_icon="📈", layout="wide")
    st.title("程序化选股助手")
    st.caption("策略：事件驱动龙头的分歧转一致（自动采集 + 自动打分）")

    with st.expander("计算与打分原理（与 `core/data_provider.py`、`core/scoring.py` 一致）", expanded=False):
        st.markdown(SCORING_AND_PIPELINE_MARKDOWN)

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
            LIVE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
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
                    LIVE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
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
        st.dataframe(dataframe_for_display(filtered), use_container_width=True)
        st.metric("入选数量", value=len(filtered))

        if not filtered.empty:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            out_path = OUTPUT_DIR / f"candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filtered.to_csv(out_path, index=False, encoding="utf-8-sig")
            st.success(f"已导出结果：{out_path}")
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

