# Trading Selector UI

一个用于 A 股“事件驱动龙头分歧转一致”策略的程序化选股系统（自动采集版）。

## 功能

- 用户界面（Streamlit）
- 自动采集 A 股实时行情（akshare）
- 自动计算核心指标（题材强度、板块联动、个股强度、资金承接）
- 自动打分并输出候选股列表
- 支持导出到 `output/`

## 目录结构

- `app/main.py`：UI 入口
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

## 指标来源

- `pct_chg`：涨跌幅（实时行情）
- `turnover`：换手率（实时行情）
- `amount`：成交额（实时行情）
- `volume_ratio`：量比（实时行情）
- `amplitude`：振幅（实时行情）
- `theme_strength`：行业平均涨幅 + 行业成交额（归一化）
- `sector_linkage`：行业平均涨幅 + 行业内上涨占比（归一化）
- `stock_strength`：涨跌幅 + 量比 + 振幅（归一化）
- `capital_support`：换手率 + 成交额 + 量比（归一化）

## 下一步扩展

- 加入分钟级别“分歧转一致”触发检测
- 对接你的交易日志做策略回测和迭代
