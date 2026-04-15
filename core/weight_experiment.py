from __future__ import annotations

import itertools
import random

import numpy as np
import pandas as pd

from core.evaluation import portfolio_summary
from core.run_store import load_run_candidates, load_run_evaluation
from core.scoring import score_frame


def _normalize(w: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    s = sum(w)
    if s <= 0:
        return 0.25, 0.25, 0.25, 0.25
    return w[0] / s, w[1] / s, w[2] / s, w[3] / s


def replay_weights_on_run(
    run_id: str,
    w_theme: float,
    w_sector: float,
    w_stock: float,
    w_capital: float,
    ret_col: str,
    top_n: int,
) -> dict | None:
    cands = load_run_candidates(run_id)
    ev = load_run_evaluation(run_id)
    if cands is None or ev is None:
        return None
    need = {"theme_strength", "sector_linkage", "stock_strength", "capital_support"}
    if not need.issubset(cands.columns):
        return None
    scored = score_frame(cands, w_theme, w_sector, w_stock, w_capital)
    top = scored.head(top_n)
    merged = top.merge(ev[["symbol"] + ([ret_col] if ret_col in ev.columns else [])], on="symbol", how="inner")
    if ret_col not in merged.columns:
        return None
    return portfolio_summary(merged, ret_col=ret_col)


def run_experiment(
    run_ids: list[str],
    *,
    ret_col: str,
    top_n: int,
    mode: str,
    n_random: int,
    seed: int,
) -> pd.DataFrame:
    grid = [0.2, 0.25, 0.3, 0.35]
    weights_pool: list[tuple[float, float, float, float]] = []
    if mode in ("grid", "both"):
        for a, b, c, d in itertools.product(grid, grid, grid, grid):
            weights_pool.append(_normalize((a, b, c, d)))
    if mode in ("random", "both"):
        rng = random.Random(seed)
        for _ in range(n_random):
            x = [rng.random() for _ in range(4)]
            weights_pool.append(_normalize(tuple(x)))

    seen: set[tuple[float, float, float, float]] = set()
    uniq: list[tuple[float, float, float, float]] = []
    for w in weights_pool:
        key = tuple(round(x, 4) for x in w)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(w)

    rows = []
    for w_theme, w_sector, w_stock, w_capital in uniq:
        means: list[float] = []
        wins: list[float] = []
        for rid in run_ids:
            stat = replay_weights_on_run(rid, w_theme, w_sector, w_stock, w_capital, ret_col, top_n)
            if stat and stat.get("mean_ret") is not None:
                means.append(stat["mean_ret"])
                wins.append(stat.get("win_rate") or 0.0)
        if not means:
            continue
        risk_adj = float(np.mean(means)) / (np.std(means) + 1e-6) if len(means) > 1 else float(np.mean(means))
        rows.append(
            {
                "w_theme": w_theme,
                "w_sector": w_sector,
                "w_stock": w_stock,
                "w_capital": w_capital,
                "mean_ret_across_runs": float(np.mean(means)),
                "std_ret_across_runs": float(np.std(means)),
                "mean_win_rate": float(np.mean(wins)),
                "n_runs_used": len(means),
                "risk_adjusted_score": risk_adj,
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values("risk_adjusted_score", ascending=False).reset_index(drop=True)
