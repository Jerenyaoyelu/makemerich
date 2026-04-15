from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RUNS_CSV = DATA_DIR / "runs.csv"
RUN_CANDIDATES_DIR = DATA_DIR / "run_candidates"
RUN_EVAL_DIR = DATA_DIR / "run_evaluations"
STRATEGY_REPORT_CSV = DATA_DIR / "strategy_report.csv"
DEFAULT_WEIGHTS_JSON = DATA_DIR / "ui_default_weights.json"
COLLECTION_HISTORY_CSV = DATA_DIR / "collection_history.csv"


def ensure_data_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RUN_CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    RUN_EVAL_DIR.mkdir(parents=True, exist_ok=True)


def new_run_id() -> str:
    return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def run_candidate_path(run_id: str) -> Path:
    return RUN_CANDIDATES_DIR / f"{run_id}.csv"


def run_evaluation_path(run_id: str) -> Path:
    return RUN_EVAL_DIR / f"{run_id}.csv"


def _runs_columns() -> list[str]:
    return [
        "run_id",
        "created_at",
        "w_theme",
        "w_sector",
        "w_stock",
        "w_capital",
        "score_threshold",
        "top_n",
        "market_scope",
        "refresh_limit",
        "data_source",
        "source_used",
        "degraded",
        "health_score",
        "health_notes",
        "candidate_count",
        "track_eval",
    ]


def append_run(
    run_id: str,
    *,
    w_theme: float,
    w_sector: float,
    w_stock: float,
    w_capital: float,
    score_threshold: float,
    top_n: int,
    market_scope: str,
    refresh_limit: int,
    data_source: str,
    source_used: str,
    degraded: bool,
    health_score: float,
    health_notes: str,
    candidate_count: int,
    track_eval: bool = True,
) -> None:
    ensure_data_dirs()
    row = {
        "run_id": run_id,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "w_theme": w_theme,
        "w_sector": w_sector,
        "w_stock": w_stock,
        "w_capital": w_capital,
        "score_threshold": score_threshold,
        "top_n": top_n,
        "market_scope": market_scope,
        "refresh_limit": refresh_limit,
        "data_source": data_source,
        "source_used": source_used,
        "degraded": degraded,
        "health_score": health_score,
        "health_notes": health_notes,
        "candidate_count": candidate_count,
        "track_eval": track_eval,
    }
    df = pd.DataFrame([row])
    if RUNS_CSV.exists():
        df.to_csv(RUNS_CSV, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(RUNS_CSV, index=False, encoding="utf-8-sig")


def save_run_candidates(run_id: str, df: pd.DataFrame) -> Path:
    ensure_data_dirs()
    path = run_candidate_path(run_id)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def list_runs() -> pd.DataFrame:
    if not RUNS_CSV.exists():
        return pd.DataFrame(columns=_runs_columns())
    return pd.read_csv(RUNS_CSV, dtype={"run_id": str})


def load_run_candidates(run_id: str) -> pd.DataFrame | None:
    p = run_candidate_path(run_id)
    if not p.exists():
        return None
    return pd.read_csv(p, dtype={"symbol": str})


def load_run_evaluation(run_id: str) -> pd.DataFrame | None:
    p = run_evaluation_path(run_id)
    if not p.exists():
        return None
    return pd.read_csv(p, dtype={"symbol": str})


def append_collection_history(success: bool, source: str, message: str = "") -> None:
    ensure_data_dirs()
    row = {
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "success": success,
        "source": source,
        "message": message[:500],
    }
    df = pd.DataFrame([row])
    if COLLECTION_HISTORY_CSV.exists():
        df.to_csv(COLLECTION_HISTORY_CSV, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(COLLECTION_HISTORY_CSV, index=False, encoding="utf-8-sig")


def recent_collection_success_rate(last_n: int = 20) -> float | None:
    if not COLLECTION_HISTORY_CSV.exists():
        return None
    try:
        h = pd.read_csv(COLLECTION_HISTORY_CSV)
        if h.empty:
            return None
        tail = h.tail(last_n)
        return float(tail["success"].astype(bool).mean())
    except Exception:
        return None


def append_strategy_report(row: dict[str, Any]) -> None:
    ensure_data_dirs()
    df = pd.DataFrame([row])
    if STRATEGY_REPORT_CSV.exists():
        df.to_csv(STRATEGY_REPORT_CSV, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(STRATEGY_REPORT_CSV, index=False, encoding="utf-8-sig")


def save_default_weights(w_theme: float, w_sector: float, w_stock: float, w_capital: float) -> None:
    ensure_data_dirs()
    payload = {
        "w_theme": w_theme,
        "w_sector": w_sector,
        "w_stock": w_stock,
        "w_capital": w_capital,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    DEFAULT_WEIGHTS_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_default_weights() -> dict[str, float] | None:
    if not DEFAULT_WEIGHTS_JSON.exists():
        return None
    try:
        data = json.loads(DEFAULT_WEIGHTS_JSON.read_text(encoding="utf-8"))
        return {
            "w_theme": float(data["w_theme"]),
            "w_sector": float(data["w_sector"]),
            "w_stock": float(data["w_stock"]),
            "w_capital": float(data["w_capital"]),
        }
    except Exception:
        return None
