from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _format_symbol_for_ak(symbol: str) -> str:
    s = str(symbol).strip()
    if s.startswith(("sh", "sz", "bj")):
        return s
    if s.startswith(("6", "9")):
        return f"sh{s}"
    if s.startswith(("8", "4")):
        return f"bj{s}"
    return f"sz{s}"


def _next_trading_row(hist: pd.DataFrame, trade_date: str) -> tuple[pd.Series | None, pd.Series | None]:
    # hist columns expected: 日期 开盘 收盘 最高 最低 ...
    d = pd.to_datetime(trade_date).date()
    hist = hist.copy()
    hist["日期"] = pd.to_datetime(hist["日期"]).dt.date
    today = hist[hist["日期"] == d]
    next_rows = hist[hist["日期"] > d].sort_values("日期")
    if today.empty or next_rows.empty:
        return None, None
    return today.iloc[-1], next_rows.iloc[0]


def evaluate_file(input_csv: Path, output_csv: Path) -> None:
    import akshare as ak

    df = pd.read_csv(input_csv, dtype={"symbol": str})
    if "snapshot_time" not in df.columns:
        raise ValueError("输入文件缺少 snapshot_time 字段。")

    result = df.copy()
    result["next_day_open"] = pd.NA
    result["next_day_close"] = pd.NA
    result["next_day_high"] = pd.NA
    result["next_day_low"] = pd.NA
    result["next_day_close_pct_vs_today_close"] = pd.NA

    for idx, row in result.iterrows():
        symbol = _format_symbol_for_ak(row["symbol"])
        snapshot_date = pd.to_datetime(row["snapshot_time"]).strftime("%Y%m%d")
        try:
            hist = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=snapshot_date,
                end_date=datetime.now().strftime("%Y%m%d"),
                adjust="qfq",
            )
        except Exception:
            continue
        if hist is None or hist.empty:
            continue

        today_row, next_row = _next_trading_row(hist, snapshot_date)
        if today_row is None or next_row is None:
            continue

        result.at[idx, "next_day_open"] = next_row["开盘"]
        result.at[idx, "next_day_close"] = next_row["收盘"]
        result.at[idx, "next_day_high"] = next_row["最高"]
        result.at[idx, "next_day_low"] = next_row["最低"]
        try:
            base_close = float(today_row["收盘"])
            next_close = float(next_row["收盘"])
            result.at[idx, "next_day_close_pct_vs_today_close"] = round((next_close / base_close - 1) * 100, 4)
        except Exception:
            pass

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"评估完成: {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="对候选池文件补充次日表现字段")
    parser.add_argument("--input", required=True, help="候选池 CSV 路径")
    parser.add_argument("--output", required=False, help="输出 CSV 路径")
    args = parser.parse_args()

    input_csv = Path(args.input).resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_csv}")
    output_csv = Path(args.output).resolve() if args.output else input_csv.with_name(f"{input_csv.stem}_evaluated.csv")
    evaluate_file(input_csv, output_csv)


if __name__ == "__main__":
    main()
