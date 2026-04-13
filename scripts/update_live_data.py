from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.data_provider import fetch_live_signals


def main() -> None:
    out_path = PROJECT_ROOT / "data" / "latest_signals.csv"
    frame = fetch_live_signals(limit=500)
    frame.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"已更新: {out_path} | 记录数: {len(frame)}")


if __name__ == "__main__":
    main()
