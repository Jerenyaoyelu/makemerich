from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
for p in (PROJECT_ROOT, APP_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

st.set_page_config(page_title="选股工作台", page_icon="🧭", layout="wide")

dashboard = st.Page(
    "pages/0_系统驾驶舱.py",
    title="系统驾驶舱（首页）",
    icon="🧭",
    default=True,
)
screener = st.Page("live_screener.py", title="实时选股（执行页）", icon="📈")
backtest = st.Page("pages/3_历史回测_T0.py", title="历史回测 T0", icon="📅")
review = st.Page("pages/1_复评中心.py", title="复评中心", icon="📊")
sell = st.Page("pages/4_卖出决策中心.py", title="卖出决策中心", icon="🛟")
params = st.Page("pages/2_参数实验室.py", title="参数实验室", icon="🧪")

pg = st.navigation(
    [dashboard, screener, backtest, review, sell, params],
)
pg.run()
