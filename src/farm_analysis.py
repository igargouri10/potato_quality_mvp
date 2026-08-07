from __future__ import annotations

from typing import Any

import pandas as pd


DEFAULT_COMPARISON_FIELDS = [
    "total_defects",
    "greening",
    "wet_breakdown",
    "wb",
    "wsttogoal",
    "wsttogrn",
    "wsttogta",
    "ndw_total",
    "wgo_total",
    "wgr_total",
    "wgta_total",
    "specific_gravity",
    "temperature",
]


def _available_fields(df: pd.DataFrame, fields: list[str] | None = None) -> list[str]:
    candidates = fields or DEFAULT_COMPARISON_FIELDS
    return [field for field in candidates if field in df.columns]


def _clean_filter_values(values: list[Any] | None) -> set[str]:
    if not values:
        return set()
    return {str(value).strip().upper() for value in values if str(value).strip()}


def _format_batch_list(values: pd.Series, limit: int = 5) -> str:
    batches = [str(value) for value in values.dropna().astype(str).unique() if value]
    if not batches:
        return "unknown batches"
    shown = batches[:limit]
    suffix = "" if len(batches) <= limit else f" and {len(batches) - limit} more"
    return ", ".join(shown) + suffix


def _numeric_summary(df: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    rows = []
    for field in fields:
        values = pd.to_numeric(df[field], errors="coerce")
        if values.notna().any():
            rows.append({"field": field, "mean": float(values.mean()), "median": float(values.median())})
    return pd.DataFrame(rows)


def filter_qc_for_farm_analysis(
    df: pd.DataFrame,
    plant_names: list[Any] | None = None,
    po_numbers: list[Any] | None = None,
    start_date: Any | None = None,
    end_date: Any | None = None,
) -> pd.DataFrame:
    """Apply plant, PO, and scheduled-delivery filters for farm analysis."""
    filtered = df.copy()

    plant_set = _clean_filter_values(plant_names)
    if plant_set and "plant_name" in filtered.columns:
        filtered = filtered[filtered["plant_name"].astype(str).str.upper().isin(plant_set)]

    po_set = _clean_filter_values(po_numbers)
    if po_set and "po_number" in filtered.columns:
        filtered = filtered[filtered["po_number"].astype(str).str.upper().isin(po_set)]

    if "scheduled_delivery_date" in filtered.columns:
        dates = pd.to_datetime(filtered["scheduled_delivery_date"], errors="coerce")
        if start_date is not None:
            filtered = filtered[dates >= pd.to_datetime(start_date)]
            dates = pd.to_datetime(filtered["scheduled_delivery_date"], errors="coerce")
        if end_date is not None:
            filtered = filtered[dates <= pd.to_datetime(end_date)]

    return filtered


def analyze_farm_appearance(
    df: pd.DataFrame,
    low_threshold: float = 80,
    stronger_min_score: float | None = None,
    comparison_fields: list[str] | None = None,
) -> dict[str, Any]:
    """Compare low-appearance batches with stronger batches from the same farm and plant."""
    required = {"farm_label", "plant_name", "appearance_score"}
    missing = sorted(required - set(df.columns))
    if missing:
        return {
            "farm_summaries": [],
            "batch_comparisons": pd.DataFrame(),
            "message": f"Missing required column(s): {', '.join(missing)}",
        }

    working = df.copy()
    working["appearance_score"] = pd.to_numeric(working["appearance_score"], errors="coerce")
    working = working.dropna(subset=["farm_label", "plant_name", "appearance_score"])
    if working.empty:
        return {
            "farm_summaries": [],
            "batch_comparisons": pd.DataFrame(),
            "message": "No rows had farm, plant, and appearance score values.",
        }

    stronger_min_score = stronger_min_score if stronger_min_score is not None else max(70, low_threshold)
    fields = _available_fields(working, comparison_fields)
    farm_summaries: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    for (farm_label, plant_name), group in working.groupby(["farm_label", "plant_name"], dropna=True):
        low_group = group[group["appearance_score"] < low_threshold].copy()
        if low_group.empty:
            continue

        stronger_group = group[group["appearance_score"] >= stronger_min_score].copy()
        if stronger_group.empty:
            stronger_group = group[group["appearance_score"] > low_group["appearance_score"].max()].copy()

        low_mean = float(low_group["appearance_score"].mean())
        stronger_mean = float(stronger_group["appearance_score"].mean()) if not stronger_group.empty else None
        score_gap = stronger_mean - low_mean if stronger_mean is not None else None

        low_summary = _numeric_summary(low_group, fields)
        stronger_summary = _numeric_summary(stronger_group, fields) if not stronger_group.empty else pd.DataFrame()
        factor_notes: list[str] = []

        for _, low_row in low_summary.iterrows():
            field = low_row["field"]
            if stronger_summary.empty or "field" not in stronger_summary.columns:
                continue
            match = stronger_summary[stronger_summary["field"] == field]
            if match.empty:
                continue
            stronger_value = float(match.iloc[0]["mean"])
            low_value = float(low_row["mean"])
            diff = low_value - stronger_value
            if abs(diff) < 0.01:
                continue

            direction = "higher" if diff > 0 else "lower"
            factor_notes.append(
                f"{field.replace('_', ' ')} averaged {abs(diff):.1f} {direction} "
                f"than the stronger comparison batches"
            )

        low_batches = _format_batch_list(low_group.get("batch", pd.Series(dtype=str)))
        stronger_batches = _format_batch_list(stronger_group.get("batch", pd.Series(dtype=str)))

        if stronger_mean is None:
            paragraph = (
                f"For farm {farm_label} at plant {plant_name}, batches {low_batches} had appearance scores "
                f"below {low_threshold:g}, averaging {low_mean:.1f}. There were no stronger same-farm, same-plant "
                f"comparison batches in the current filters, so the analysis should be reviewed with a wider date, "
                f"PO, or plant selection."
            )
        else:
            factor_text = "; ".join(factor_notes[:4]) if factor_notes else "the available numeric QC factors did not show a clear difference"
            paragraph = (
                f"For farm {farm_label} at plant {plant_name}, lower-appearance batches {low_batches} averaged "
                f"{low_mean:.1f} appearance score versus {stronger_mean:.1f} for comparison batches {stronger_batches}. "
                f"The gap was {score_gap:.1f} points. The main measured differences were: {factor_text}."
            )

        summary = {
            "farm_label": farm_label,
            "plant_name": plant_name,
            "low_batch_count": int(len(low_group)),
            "comparison_batch_count": int(len(stronger_group)),
            "low_avg_appearance": round(low_mean, 2),
            "comparison_avg_appearance": round(stronger_mean, 2) if stronger_mean is not None else None,
            "appearance_gap": round(score_gap, 2) if score_gap is not None else None,
            "low_batches": low_batches,
            "comparison_batches": stronger_batches,
            "paragraph": paragraph,
        }
        farm_summaries.append(summary)

        for _, row in group.iterrows():
            score = float(row["appearance_score"])
            if score < low_threshold:
                group_label = "Lower appearance"
            elif score >= stronger_min_score:
                group_label = "Comparison"
            else:
                group_label = "Middle"

            comparison_rows.append(
                {
                    "farm_label": farm_label,
                    "plant_name": plant_name,
                    "batch": row.get("batch", ""),
                    "po_number": row.get("po_number", ""),
                    "scheduled_delivery_date": row.get("scheduled_delivery_date", pd.NaT),
                    "appearance_score": score,
                    "comparison_group": group_label,
                }
            )

    summaries = sorted(
        farm_summaries,
        key=lambda row: (row["appearance_gap"] is None, -(row["appearance_gap"] or 0), row["farm_label"]),
    )
    return {
        "farm_summaries": summaries,
        "batch_comparisons": pd.DataFrame(comparison_rows),
        "message": "",
    }
