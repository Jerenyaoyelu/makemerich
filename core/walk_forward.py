from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.weight_experiment import replay_weights_on_run, run_experiment


@dataclass
class WalkForwardConfig:
    ret_col: str = "ret_close_t5_net"
    top_n: int = 5
    min_train_runs: int = 3
    mode: str = "random"
    n_random: int = 80
    seed: int = 42


def _valid_runs_with_ret(runs_df: pd.DataFrame, ret_col: str, eval_exists: callable) -> list[str]:
    ordered = runs_df.sort_values("created_at")
    out: list[str] = []
    for rid in ordered["run_id"].astype(str).tolist():
        ev = eval_exists(rid)
        if ev is None or ev.empty:
            continue
        if ret_col not in ev.columns:
            continue
        out.append(rid)
    return out


def walk_forward_report(
    runs_df: pd.DataFrame,
    *,
    eval_exists: callable,
    cfg: WalkForwardConfig,
) -> pd.DataFrame:
    """
    基于历史 run 按时间顺序做扩展窗 walk-forward：
    - 前 N 个 run 作为训练集选权重
    - 第 N+1 个 run 做样本外验证
    - 迭代滚动
    """
    run_ids = _valid_runs_with_ret(runs_df, cfg.ret_col, eval_exists)
    rows: list[dict] = []
    if len(run_ids) <= cfg.min_train_runs:
        return pd.DataFrame()

    for i in range(cfg.min_train_runs, len(run_ids)):
        train_ids = run_ids[:i]
        test_id = run_ids[i]
        board = run_experiment(
            train_ids,
            ret_col=cfg.ret_col,
            top_n=cfg.top_n,
            mode=cfg.mode,
            n_random=cfg.n_random,
            seed=cfg.seed + i,
        )
        if board.empty:
            continue
        best = board.iloc[0]
        test_stat = replay_weights_on_run(
            test_id,
            float(best["w_theme"]),
            float(best["w_sector"]),
            float(best["w_stock"]),
            float(best["w_capital"]),
            cfg.ret_col,
            cfg.top_n,
        )
        if not test_stat or test_stat.get("mean_ret") is None:
            continue

        rows.append(
            {
                "train_runs": len(train_ids),
                "test_run_id": test_id,
                "ret_col": cfg.ret_col,
                "top_n": cfg.top_n,
                "w_theme": float(best["w_theme"]),
                "w_sector": float(best["w_sector"]),
                "w_stock": float(best["w_stock"]),
                "w_capital": float(best["w_capital"]),
                "train_mean_ret": float(best["mean_ret_across_runs"]),
                "train_risk_adjusted": float(best["risk_adjusted_score"]),
                "oos_mean_ret": float(test_stat["mean_ret"]),
                "oos_win_rate": float(test_stat.get("win_rate") or 0.0),
                "oos_n": int(test_stat.get("n") or 0),
            }
        )

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["cum_oos_ret"] = out["oos_mean_ret"].cumsum()
    return out

