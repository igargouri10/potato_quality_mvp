from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.load_qc_data import load_qc_data


DEFAULT_OUTPUT_DIR = Path("data/processed/matched_records")

FACTOR_CANDIDATES = [
    "tier",
    "total_defects",
    "nodbtwst",
    "wsttogoal",
    "wsttogrn",
    "wsttogta",
    "stdev",
    "wb",
    "wet_breakdown",
    "greening",
    "under_size",
    "undersize",
    "oversize",
    "other",
    "mud",
    "dirt_rcv",
    "tuber_size",
    "solids",
    "hunter_l",
    "hunter_a",
    "load_to_fry_difference",
    "high_temp",
    "low_temp",
    "wgo_ext_blk",
    "wgo_ext_brn",
    "wgo_int_blk",
    "wgo_int_brn",
    "wgr_ext_blk",
    "wgr_ext_brn",
    "wgr_int_blk",
    "wgr_int_brn",
    "wgta_ext_blk",
    "wgta_ext_brn",
    "wgta_int_blk",
    "wgta_int_brn",
]

HIGHER_IS_WORSE = {
    "tier",
    "total_defects",
    "nodbtwst",
    "wsttogoal",
    "wsttogrn",
    "wsttogta",
    "stdev",
    "wb",
    "wet_breakdown",
    "greening",
    "under_size",
    "undersize",
    "oversize",
    "other",
    "mud",
    "dirt_rcv",
    "load_to_fry_difference",
    "high_temp",
    "low_temp",
    "wgo_ext_blk",
    "wgo_ext_brn",
    "wgo_int_blk",
    "wgo_int_brn",
    "wgr_ext_blk",
    "wgr_ext_brn",
    "wgr_int_blk",
    "wgr_int_brn",
    "wgta_ext_blk",
    "wgta_ext_brn",
    "wgta_int_blk",
    "wgta_int_brn",
}

LOWER_IS_WORSE = {"solids", "hunter_l", "tuber_size"}


def _batch_list(values: pd.Series, limit: int = 8) -> str:
    batches = [str(value).lstrip("0") or str(value) for value in values.dropna().astype(str).unique() if str(value).strip()]
    if not batches:
        return "unknown batches"
    return ", ".join(batches[:limit]) + (f" and {len(batches) - limit} more" if len(batches) > limit else "")


def _date_range_text(group: pd.DataFrame) -> str:
    if "scheduled_delivery_date" not in group.columns or group["scheduled_delivery_date"].isna().all():
        return "the selected date range"
    dates = pd.to_datetime(group["scheduled_delivery_date"], errors="coerce").dropna()
    if dates.empty:
        return "the selected date range"
    return f"{dates.min().date()} to {dates.max().date()}"


def _factor_notes(low: pd.DataFrame, comparison: pd.DataFrame, fields: list[str]) -> list[str]:
    notes: list[tuple[float, str]] = []
    for field in fields:
        low_mean = low[field].mean()
        comparison_mean = comparison[field].mean()
        if pd.isna(low_mean) or pd.isna(comparison_mean):
            continue
        diff = low_mean - comparison_mean
        if abs(diff) < 0.05:
            continue

        adverse = (field in HIGHER_IS_WORSE and diff > 0) or (field in LOWER_IS_WORSE and diff < 0)
        if not adverse:
            continue

        direction = "higher" if diff > 0 else "lower"
        label = field.replace("_", " ")
        notes.append(
            (
                abs(diff),
                f"{label} averaged {abs(diff):.2f} {direction} ({low_mean:.2f} vs {comparison_mean:.2f})",
            )
        )

    notes.sort(reverse=True, key=lambda item: item[0])
    return [note for _, note in notes[:5]]


def _output_name(source_path: Path) -> str:
    stem = source_path.stem.lower().replace("data from ", "").replace(" ", "_")
    return f"supplier_site_appearance_analysis_{stem}.md"


def generate_supplier_site_report(
    source_path: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Generate supplier-site appearance comparison paragraphs from a daily QC export."""
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_qc_data(source_path)
    df = df.dropna(subset=["farm_label", "plant_name", "appearance_score"]).copy()
    df["appearance_score"] = pd.to_numeric(df["appearance_score"], errors="coerce")

    fields = [field for field in FACTOR_CANDIDATES if field in df.columns]
    for field in fields:
        df[field] = pd.to_numeric(df[field], errors="coerce")

    paragraphs: list[tuple[str, str, str]] = []
    summary_rows: list[dict[str, Any]] = []

    for (supplier_site, plant_name), group in df.groupby(["farm_label", "plant_name"], dropna=True):
        group = group.dropna(subset=["appearance_score"]).copy()
        if group.empty:
            continue

        overall_avg = group["appearance_score"].mean()
        if overall_avg >= 80:
            low_threshold = 80
            comparison_threshold = 80
            threshold_text = "below 80"
        else:
            low_threshold = 70
            comparison_threshold = 70
            threshold_text = "in the 50s/60s or below 70 because this supplier-site/plant average is already under 80"

        low = group[group["appearance_score"] < low_threshold]
        comparison = group[group["appearance_score"] >= comparison_threshold]
        if low.empty:
            continue

        low_avg = low["appearance_score"].mean()
        comparison_avg = comparison["appearance_score"].mean() if not comparison.empty else None
        low_batches = _batch_list(low["batch"]) if "batch" in low.columns else "unknown batches"
        comparison_batches = _batch_list(comparison["batch"]) if "batch" in comparison.columns else "unknown batches"
        date_text = _date_range_text(group)

        summary_rows.append(
            {
                "supplier_site": supplier_site,
                "plant_name": plant_name,
                "rows": len(group),
                "overall_avg_asavg": round(overall_avg, 2),
                "low_threshold": low_threshold,
                "low_batch_count": len(low),
                "comparison_batch_count": len(comparison),
                "low_avg_asavg": round(low_avg, 2),
                "comparison_avg_asavg": "" if comparison_avg is None else round(comparison_avg, 2),
            }
        )

        if comparison.empty:
            paragraph = (
                f"{supplier_site} | {plant_name}: From {date_text}, {len(low)} batch(es) were {threshold_text}, "
                f"averaging {low_avg:.1f} ASAVG across batches {low_batches}. This group averaged {overall_avg:.1f} "
                f"overall and did not have same supplier-site, same-plant comparison batches at or above "
                f"{comparison_threshold}, so the lower appearance scores cannot be compared fairly inside this "
                f"filtered dataset. Use a wider date range, additional PO numbers, or a nearby delivery window "
                f"before drawing a root-cause conclusion."
            )
        else:
            notes = _factor_notes(low, comparison, fields)
            factor_text = "; ".join(notes) if notes else "the selected numeric QC factors did not show a clear adverse difference"
            paragraph = (
                f"{supplier_site} | {plant_name}: From {date_text}, the lower-appearance group ({threshold_text}) "
                f"averaged {low_avg:.1f} ASAVG across {len(low)} batch(es): {low_batches}. The same supplier-site "
                f"and same plant had {len(comparison)} stronger comparison batch(es) at or above "
                f"{comparison_threshold}, averaging {comparison_avg:.1f} ASAVG: {comparison_batches}. The appearance "
                f"gap was {comparison_avg - low_avg:.1f} points. The strongest measured differences pointing toward "
                f"the lower scores were: {factor_text}."
            )

        paragraphs.append((supplier_site, plant_name, paragraph))

    paragraphs.sort(key=lambda item: (item[0], item[1]))
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(["supplier_site", "plant_name"])

    lines = ["# Supplier Site Appearance Analysis", "", f"Source: {source_path.name}", ""]
    if not summary.empty:
        lines.extend(["## Summary Table", summary.to_string(index=False), ""])
    lines.append("## Paragraphs")
    for supplier_site, plant_name, paragraph in paragraphs:
        lines.extend([f"### {supplier_site} | {plant_name}", paragraph, ""])

    output_path = output_dir / _output_name(source_path)
    output_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "output_path": output_path,
        "summary": summary,
        "paragraph_count": len(paragraphs),
        "row_count": len(df),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate supplier-site appearance analysis paragraphs.")
    parser.add_argument("source_path")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    result = generate_supplier_site_report(args.source_path, args.output_dir)
    print(f"Rows analyzed: {result['row_count']}")
    print(f"Paragraphs generated: {result['paragraph_count']}")
    print(f"Output: {result['output_path']}")
