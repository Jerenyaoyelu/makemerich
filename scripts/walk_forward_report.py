from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.run_store import RUNS_CSV, list_runs, load_run_evaluation
from core.walk_forward import WalkForwardConfig, walk_forward_report


def main() -> None:
    parser = argparse.ArgumentParser(description="滚动样本外验证（walk-forward）报表")
    parser.add_argument("--ret-col", default="ret_close_t5_net", help="目标收益列")
    parser.add_argument("--top-n", type=int, default=5, help="组合容量")
    parser.add_argument("--min-train-runs", type=int, default=3, help="最小训练 run 数")
    parser.add_argument("--mode", choices=["grid", "random", "both"], default="random")
    parser.add_argument("--n-random", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="", help="输出 CSV 路径")
    args = parser.parse_args()

    if not RUNS_CSV.exists():
        print("尚无 runs.csv，请先运行筛选与复评。")
        sys.exit(1)
    runs = list_runs()
    if runs.empty:
        print("runs 为空，请先运行筛选与复评。")
        sys.exit(1)

    cfg = WalkForwardConfig(
        ret_col=args.ret_col,
        top_n=args.top_n,
        min_train_runs=args.min_train_runs,
        mode=args.mode,
        n_random=args.n_random,
        seed=args.seed,
    )
    report = walk_forward_report(runs, eval_exists=load_run_evaluation, cfg=cfg)
    if report.empty:
        print("无有效 walk-forward 结果：请确认 run 数足够且复评列存在。")
        sys.exit(1)

    out_path = (
        Path(args.output).resolve()
        if args.output
        else PROJECT_ROOT / "output" / f"walk_forward_{args.ret_col}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_path, index=False, encoding="utf-8-sig")

    s = pd.to_numeric(report["oos_mean_ret"], errors="coerce").dropna()
    wr = pd.to_numeric(report["oos_win_rate"], errors="coerce").dropna()
    print(f"已写入: {out_path}")
    print(
        "OOS 摘要:",
        {
            "steps": int(len(report)),
            "oos_mean_ret_avg": round(float(s.mean()), 4) if not s.empty else None,
            "oos_mean_ret_median": round(float(s.median()), 4) if not s.empty else None,
            "oos_win_rate_avg": round(float(wr.mean()), 4) if not wr.empty else None,
            "cum_oos_ret_last": round(float(report['cum_oos_ret'].iloc[-1]), 4),
        },
    )


if __name__ == "__main__":
    main()

