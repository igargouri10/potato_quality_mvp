from __future__ import annotations

from typing import Any

import pandas as pd


def _clean_id(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.lstrip("0") or text


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().upper()


def _metadata_values(metadata: dict[str, Any]) -> dict[str, str]:
    return {
        "batch": _clean_id(metadata.get("batch_number") or metadata.get("batch")),
        "po_number": _clean_id(metadata.get("purchase_order") or metadata.get("po_number")),
        "po_line": _clean_id(metadata.get("po_line") or metadata.get("line")),
        "plant_name": _clean_text(metadata.get("site") or metadata.get("plant") or metadata.get("plant_name")),
    }


def _row_value(row: pd.Series, field: str) -> str:
    if field == "plant_name":
        return _clean_text(row.get(field, ""))
    return _clean_id(row.get(field, ""))


def _score_row(row: pd.Series, target: dict[str, str]) -> tuple[int, list[str]]:
    weights = {"batch": 8, "po_number": 4, "po_line": 2, "plant_name": 1}
    score = 0
    matched_fields: list[str] = []
    for field, weight in weights.items():
        expected = target.get(field, "")
        if expected and _row_value(row, field) == expected:
            score += weight
            matched_fields.append(field)
    return score, matched_fields


def match_qc_record(metadata: dict[str, Any], qc_df: pd.DataFrame) -> dict[str, Any]:
    """Find the best deterministic QC row for extracted report metadata."""
    if qc_df.empty:
        return {
            "match_status": "none",
            "matched_row_index": None,
            "matched_fields": [],
            "qc_record": {},
            "message": "QC file has no rows.",
        }

    target = _metadata_values(metadata)
    best_index = None
    best_score = -1
    best_fields: list[str] = []

    for index, row in qc_df.iterrows():
        score, fields = _score_row(row, target)
        if score > best_score:
            best_index = index
            best_score = score
            best_fields = fields

    if best_score >= 15:
        status = "strong"
    elif {"batch", "po_number"}.issubset(best_fields):
        status = "medium"
    elif "batch" in best_fields:
        status = "weak"
    else:
        status = "none"

    record = qc_df.loc[best_index].to_dict() if best_index is not None and status != "none" else {}
    return {
        "match_status": status,
        "matched_row_index": None if status == "none" else int(best_index),
        "matched_fields": best_fields if status != "none" else [],
        "qc_record": record,
        "metadata_used": target,
        "score": max(best_score, 0),
    }
