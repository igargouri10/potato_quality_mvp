from __future__ import annotations

from typing import Any

import pandas as pd


DEFAULT_THRESHOLDS = {
    "wet_breakdown_poor": 0,
    "greening_poor": 0,
    "total_defects_poor": 10,
    "asavg_warning_below": 70,
    "wsttogoal_warning_above": 0,
}


def _get(record: dict[str, Any], *names: str) -> Any:
    lowered = {str(key).lower(): value for key, value in record.items()}
    for name in names:
        if name in record:
            return record[name]
        value = lowered.get(name.lower())
        if value is not None:
            return value
    return None


def _number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().replace("%", "").replace(",", "")
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def classify_quality(record: dict[str, Any], thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    """Classify a QC row as Good, Warning, Poor, or Unknown with reasons."""
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    if not record:
        return {"quality_status": "Unknown", "reasons": ["No matched QC record was found."]}

    reasons: list[str] = []
    status = "Good"

    disposition = str(_get(record, "disposition", "usage_decision") or "").strip().lower()
    if any(word in disposition for word in ["reject", "rejected", "hold", "dump"]):
        status = "Poor"
        reasons.append("Disposition indicates the load was rejected or held.")

    tier = _number(_get(record, "tier"))
    if tier is not None:
        if tier in (1, 2):
            reasons.append("Tier indicates acceptable quality.")
        elif tier == 3:
            status = "Warning" if status != "Poor" else status
            reasons.append("Tier indicates warning level.")
        elif tier >= 4:
            status = "Poor"
            reasons.append("Tier indicates poor quality.")

    wet_breakdown = _number(_get(record, "wet_breakdown", "wb"))
    if wet_breakdown is not None and wet_breakdown > thresholds["wet_breakdown_poor"]:
        status = "Poor"
        reasons.append("Wet breakdown is above the preferred threshold.")

    greening = _number(_get(record, "greening"))
    if greening is not None and greening > thresholds["greening_poor"]:
        status = "Poor"
        reasons.append("Greening is above the preferred threshold.")

    total_defects = _number(_get(record, "total_defects"))
    if total_defects is not None and total_defects > thresholds["total_defects_poor"]:
        status = "Poor"
        reasons.append("Total defects are above the poor-quality threshold.")

    asavg = _number(_get(record, "asavg"))
    if asavg is not None and asavg < thresholds["asavg_warning_below"]:
        status = "Warning" if status == "Good" else status
        reasons.append("ASAVG is below the preferred threshold.")

    waste_to_goal = _number(_get(record, "wsttogoal", "wst_to_goal", "waste_to_goal"))
    if waste_to_goal is not None and waste_to_goal > thresholds["wsttogoal_warning_above"]:
        status = "Warning" if status == "Good" else status
        reasons.append("Waste to goal is above the preferred threshold.")

    if not reasons:
        reasons.append("No configured QC rule found a quality concern.")

    return {"quality_status": status, "reasons": reasons, "thresholds": thresholds}
