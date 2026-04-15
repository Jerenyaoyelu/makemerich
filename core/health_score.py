from __future__ import annotations

from datetime import datetime

import pandas as pd


def compute_health_score(
    frame: pd.DataFrame,
    *,
    source_used: str,
    degraded: bool,
    recent_collection_rate: float | None = None,
    backtest_mode: bool = False,
) -> tuple[float, str, dict]:
    """
    返回 (0~100 分, 简短说明, 明细 dict)。
    degraded=True 表示未使用主源（东财）。
    backtest_mode=True：历史截面回测，不按「快照是否今天」扣分，也不按实时采集成功率扣分。
    """
    details: dict = {
        "source_used": source_used,
        "degraded": degraded,
        "field_coverage_pts": 0.0,
        "freshness_pts": 0.0,
        "collection_pts": 0.0,
        "deductions": [],
    }
    total = len(frame)
    if total == 0:
        return 0.0, "无数据", details

    def mean_or_zero(series: pd.Series) -> float:
        return float(series.fillna(False).mean()) if len(series) else 0.0

    theme_ok = mean_or_zero(
        frame.get("theme_source", pd.Series(["unknown"] * total)) != "unknown"
    )
    turn_ok = mean_or_zero(frame.get("turnover_source", pd.Series(["missing"] * total)) == "spot")
    vr_ok = mean_or_zero(frame.get("volume_ratio_source", pd.Series(["missing"] * total)) == "spot")

    coverage01 = theme_ok * 0.4 + turn_ok * 0.3 + vr_ok * 0.3
    details["theme_coverage"] = round(theme_ok * 100, 2)
    details["turnover_spot_coverage"] = round(turn_ok * 100, 2)
    details["volume_ratio_spot_coverage"] = round(vr_ok * 100, 2)
    field_pts = coverage01 * 50.0
    details["field_coverage_pts"] = round(field_pts, 2)

    if theme_ok < 0.5:
        details["deductions"].append("行业有效覆盖不足")
    if turn_ok < 0.5:
        details["deductions"].append("换手非实时字段占比高")
    if vr_ok < 0.3:
        details["deductions"].append("量比大量缺失")

    if backtest_mode:
        freshness_pts = 30.0
        details["freshness_pts"] = freshness_pts
        coll_pts = 20.0
        details["collection_pts"] = coll_pts
        details["mode"] = "historical_backtest"
    else:
        # 新鲜度：快照日期为今天则满分，否则递减
        freshness_pts = 25.0
        try:
            st = frame["snapshot_time"].iloc[0]
            snap = pd.to_datetime(st)
            now = datetime.now()
            if snap.date() == now.date():
                freshness_pts = 30.0
            elif (now - snap.to_pydatetime()).days <= 1:
                freshness_pts = 20.0
            elif (now - snap.to_pydatetime()).days <= 3:
                freshness_pts = 10.0
            else:
                freshness_pts = 5.0
                details["deductions"].append("快照偏旧")
        except Exception:
            freshness_pts = 15.0
            details["deductions"].append("无法解析快照时间")
        details["freshness_pts"] = freshness_pts

        coll_pts = 20.0
        if recent_collection_rate is not None:
            coll_pts = 20.0 * recent_collection_rate
            if recent_collection_rate < 0.7:
                details["deductions"].append("近期采集失败率偏高")
        details["collection_pts"] = round(coll_pts, 2)

    downgrade_penalty = 15.0 if degraded else 0.0
    if degraded:
        details["deductions"].append("当前为备源模式（非东财主源）")

    raw = field_pts + freshness_pts + coll_pts - downgrade_penalty
    score = max(0.0, min(100.0, raw))
    if details["deductions"]:
        notes = "; ".join(details["deductions"])
    elif backtest_mode:
        notes = "历史截面回测：未按实时快照/采集成功率扣分"
    else:
        notes = "字段与新鲜度正常"
    return round(score, 1), notes, details
