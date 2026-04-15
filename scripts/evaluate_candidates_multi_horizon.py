from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.evaluation import evaluate_csv_to_path


def main() -> None:
    parser = argparse.ArgumentParser(description="多周期复评（T+1~T+10 交易日）")
    parser.add_argument("--input", required=True, help="候选池 CSV（须含 symbol, snapshot_time）")
    parser.add_argument("--output", required=False, help="输出 CSV 路径")
    args = parser.parse_args()

    input_csv = Path(args.input).resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_csv}")
    output_csv = Path(args.output).resolve() if args.output else input_csv.with_name(f"{input_csv.stem}_eval.csv")
    evaluate_csv_to_path(input_csv, output_csv)
    print(f"多周期复评完成: {output_csv}")


if __name__ == "__main__":
    main()
