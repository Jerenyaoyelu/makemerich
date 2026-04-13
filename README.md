# Trading Selector UI

一个用于 A 股“事件驱动龙头分歧转一致”策略的程序化选股脚手架。

## 功能

- 简单用户界面（Streamlit）
- 可配置选股参数（题材强度、板块联动、个股强度、资金承接）
- 自动打分并输出候选股列表
- 支持导出到 `output/`

## 目录结构

- `app/main.py`：UI 入口
- `core/models.py`：数据模型
- `core/scoring.py`：打分引擎
- `core/data_provider.py`：数据源（当前为样例 CSV）
- `data/sample_stocks.csv`：样例数据
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

## 下一步扩展

- 用真实行情/新闻数据替换 `core/data_provider.py`
- 加入分钟级别“分歧转一致”触发检测
- 对接你的交易日志做策略回测和迭代
