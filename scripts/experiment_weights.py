from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.run_store import RUNS_CSV, load_run_evaluation, list_runs
from core.weight_experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="历史 run 权重网格/随机回放，对比真实复评收益")
    parser.add_argument("--ret-col", default="ret_close_t5", help="用于排序的收益率列")
    parser.add_argument("--top-n", type=int, default=5, help="模拟组合取前 N 名")
    parser.add_argument("--mode", choices=["grid", "random", "both"], default="both")
    parser.add_argument("--n-random", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="", help="排行榜 CSV 路径")
    args = parser.parse_args()

    if not RUNS_CSV.exists():
        print("尚无 runs.csv，请先在筛选页生成 run。")
        sys.exit(1)
    runs = list_runs()
    if runs.empty:
        print("runs 为空")
        sys.exit(1)

    run_ids: list[str] = []
    for rid in runs["run_id"].astype(str).tolist():
        ev = load_run_evaluation(rid)
        if ev is None or ev.empty:
            continue
        if args.ret_col not in ev.columns:
            continue
        run_ids.append(rid)

    if not run_ids:
        print(f"没有含复评列 {args.ret_col} 的 run，请先完成复评。")
        sys.exit(1)

    df = run_experiment(
        run_ids,
        ret_col=args.ret_col,
        top_n=args.top_n,
        mode=args.mode,
        n_random=args.n_random,
        seed=args.seed,
    )
    if df.empty:
        print("无有效结果（检查候选与复评是否可对齐 symbol）。")
        sys.exit(1)

    out_path = Path(args.output) if args.output else PROJECT_ROOT / "output" / "weight_experiment_leaderboard.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"已写入排行榜: {out_path}")
    top = df.iloc[0]
    rec = {
        "w_theme": float(top["w_theme"]),
        "w_sector": float(top["w_sector"]),
        "w_stock": float(top["w_stock"]),
        "w_capital": float(top["w_capital"]),
    }
    print("推荐权重（风险调整收益最高的一行）:", json.dumps(rec, ensure_ascii=False))


if __name__ == "__main__":
    main()
