# Trading Selector UI

一个用于 A 股“事件驱动龙头分歧转一致”策略的程序化选股系统（自动采集版）。

## 功能

- 用户界面（Streamlit）
- 自动采集 A 股实时行情（akshare）
- 支持按交易板块采集：全部A股 / 沪深两市（主板）/ 创业板 / 科创板
- 自动计算核心指标（题材强度、板块联动、个股强度、资金承接）
- 自动打分并输出候选股列表
- 数据质量诊断面板（行业/换手/量比覆盖率与风险提示）
- 支持导出到 `output/`
- 自动生成“次日复评模板”CSV，便于回测验证评分有效性

## 闭环（run + 复评 + 权重实验）

- 每次在首页「运行筛选」会生成 `run_id`，并写入 `data/runs.csv` 与 `data/run_candidates/{run_id}.csv`
- 侧边栏进入 **复评中心**：选择 run →「更新复评数据」生成 `data/run_evaluations/{run_id}.csv`（多周期脚本见下）
- **参数实验室**：对已复评的多个 run 做权重回放，可将推荐权重保存为 `data/ui_default_weights.json`（首页侧边栏滑块默认值）

## 目录结构

- `app/main.py`：UI 入口（筛选与出池）
- `app/pages/`：复评中心、参数实验室（多页面）
- `core/models.py`：数据模型
- `core/scoring.py`：打分引擎
- `core/data_provider.py`：数据源与指标计算（自动采集 + 样例数据）
- `scripts/update_live_data.py`：命令行更新实时数据
- `data/sample_stocks.csv`：样例数据
- `data/latest_signals.csv`：自动采集后生成的数据
- `output/`：导出结果

## 快速开始

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 启动 UI

```bash
streamlit run app/main.py
```

3. 浏览器访问

- 默认地址：<http://localhost:8501>

## 命令行更新实时数据（可配合任务计划）

```bash
python scripts/update_live_data.py

若出现 `ProxyError`（代理连接失败），程序会自动执行：
1) 代理模式抓取  
2) 直连模式重试  
3) 若仍失败，UI 会直接报错，不自动回退  
4) 你可在 UI 点击“手动加载本地缓存数据”后再运行筛选
```

## 候选池次日复评（评分有效性验证）

UI 每次导出候选股时，会同时生成：

- `output/candidates_review_template_*.csv`

你也可以用脚本自动补次日行情字段：

```bash
python scripts/evaluate_candidates_next_day.py --input output/candidates_YYYYMMDD_HHMMSS.csv
```

会生成：

- `output/candidates_YYYYMMDD_HHMMSS_evaluated.csv`

多周期复评（T+1～T+10，与 UI 复评中心逻辑一致）：

```bash
python scripts/evaluate_candidates_multi_horizon.py --input data/run_candidates/run_xxx.csv --output data/run_evaluations/run_xxx.csv
```

权重实验（命令行）：

```bash
python scripts/experiment_weights.py --ret-col ret_close_t5 --mode random --n-random 100
```

## 指标来源

- `pct_chg`：涨跌幅（实时行情）
- `turnover`：换手率（实时行情；缺失时按 `成交额/流通市值*100` 估算）
- `amount`：成交额（实时行情）
- `volume_ratio`：量比（实时行情；缺失时显示为缺失并在打分中按可用性降权）
- `amplitude`：振幅（实时行情）
- `theme`：行业（实时字段优先；缺失时尝试行业映射缓存回填）
  - 行业映射构建优先级：Tushare `stock_basic` -> 同花顺分页抓取 -> 本地缓存

## 环境变量

- `TUSHARE_TOKEN`：用于拉取股票行业主数据（建议在 `.env` 配置）
- `theme_strength`：行业平均涨幅 + 行业成交额（归一化）
- `sector_linkage`：行业平均涨幅 + 行业内上涨占比（归一化）
- `stock_strength`：涨跌幅 + 量比 + 振幅（归一化）
- `capital_support`：换手率 + 成交额 + 量比（归一化）

界面会额外展示 `行业来源/换手来源/量比来源`，便于识别哪些值是原始字段、哪些是估算或缺失。

## 下一步扩展

- 加入分钟级别“分歧转一致”触发检测
- 对接你的交易日志做策略回测和迭代

## 闭环复盘文档

- `docs/18-盈利闭环操作系统.md`
- `docs/20-周度复盘模板.csv`
- `docs/21-调参实验记录.csv`
