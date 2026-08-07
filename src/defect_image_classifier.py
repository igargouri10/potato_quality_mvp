from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.extract_pdf import extract_images
from src.image_features import compute_image_features


DEFAULT_REPORTS_ROOT = Path(
    r"C:\Users\IsmailGargouri\OneDrive - Black Gold Farms\Desktop\Digital Defects Reports"
)
DEFAULT_OUTPUT_DIR = Path("data/processed/defect_image_classification")
SUPPORTED_REPORT_EXTENSIONS = {".pdf", ".pdf_", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


VISUAL_CLASSIFIER_THRESHOLDS = {
    "bad_dark_pixel_ratio": 0.055,
    "bad_brown_pixel_ratio": 0.035,
    "review_dark_pixel_ratio": 0.025,
    "review_brown_pixel_ratio": 0.015,
    "min_images_for_batch": 1,
}


def parse_report_path(path: str | Path, reports_root: str | Path = DEFAULT_REPORTS_ROOT) -> dict[str, str]:
    """Parse supplier, PO number, and likely batch number from the folder/file layout."""
    path = Path(path)
    reports_root = Path(reports_root)
    try:
        relative_parts = path.relative_to(reports_root).parts
    except ValueError:
        relative_parts = path.parts

    supplier = relative_parts[0] if len(relative_parts) >= 1 else ""
    po_number = relative_parts[1] if len(relative_parts) >= 2 else ""

    filename = path.name
    batch_match = re.search(r"_?(\d{6,8})(?:M)?[_\d]*\.pdf_?$", filename, flags=re.IGNORECASE)
    if not batch_match:
        batch_match = re.search(r"(\d{6,8})", filename)

    return {
        "supplier": supplier,
        "po_number": po_number,
        "batch": batch_match.group(1) if batch_match else "",
        "source_file": str(path),
    }


def classify_image_features(
    features: dict[str, Any],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Classify one extracted image from explainable color/brightness features."""
    thresholds = {**VISUAL_CLASSIFIER_THRESHOLDS, **(thresholds or {})}
    dark_ratio = float(features.get("dark_pixel_ratio") or 0)
    brown_ratio = float(features.get("brown_pixel_ratio") or 0)
    reasons: list[str] = []

    if dark_ratio >= thresholds["bad_dark_pixel_ratio"]:
        reasons.append(f"dark pixel ratio is high ({dark_ratio:.3f})")
    if brown_ratio >= thresholds["bad_brown_pixel_ratio"]:
        reasons.append(f"brown pixel ratio is high ({brown_ratio:.3f})")

    if reasons:
        status = "Likely Bad"
    else:
        if dark_ratio >= thresholds["review_dark_pixel_ratio"]:
            reasons.append(f"dark pixel ratio is elevated ({dark_ratio:.3f})")
        if brown_ratio >= thresholds["review_brown_pixel_ratio"]:
            reasons.append(f"brown pixel ratio is elevated ({brown_ratio:.3f})")
        status = "Review" if reasons else "Likely Good"

    if not reasons:
        reasons.append("dark and brown defect signals are below review thresholds")

    return {
        "image_status": status,
        "image_reasons": reasons,
    }


def classify_report_file(
    report_path: str | Path,
    reports_root: str | Path = DEFAULT_REPORTS_ROOT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> list[dict[str, Any]]:
    """Classify all relevant potato images from one PDF/image report file."""
    report_path = Path(report_path)
    output_dir = Path(output_dir)
    metadata = parse_report_path(report_path, reports_root)

    image_paths: list[str]
    if report_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
        image_paths = [str(report_path)]
    else:
        safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", report_path.stem)
        extract_dir = output_dir / "extracted_images" / safe_stem
        extracted = extract_images(report_path, extract_dir, filter_non_potato=True)
        image_paths = [image["path"] for image in extracted["saved_images"]]

    rows: list[dict[str, Any]] = []
    for image_path in image_paths:
        try:
            features = compute_image_features(image_path)
        except ValueError as exc:
            rows.append({**metadata, "image_file": image_path, "image_status": "Unreadable", "image_reasons": str(exc)})
            continue

        image_result = classify_image_features(features)
        rows.append(
            {
                **metadata,
                **features,
                "image_status": image_result["image_status"],
                "image_reasons": "; ".join(image_result["image_reasons"]),
            }
        )

    return rows


def summarize_batch_classifications(image_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate image-level classifications to supplier/PO/batch level."""
    if image_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    group_columns = ["supplier", "po_number", "batch", "source_file"]
    for keys, group in image_df.groupby(group_columns, dropna=False):
        supplier, po_number, batch, source_file = keys
        status_counts = group["image_status"].value_counts().to_dict()
        bad_count = int(status_counts.get("Likely Bad", 0))
        review_count = int(status_counts.get("Review", 0))
        good_count = int(status_counts.get("Likely Good", 0))
        image_count = int(len(group))

        if bad_count:
            batch_status = "Likely Bad"
        elif review_count:
            batch_status = "Review"
        elif image_count >= VISUAL_CLASSIFIER_THRESHOLDS["min_images_for_batch"]:
            batch_status = "Likely Good"
        else:
            batch_status = "No Images"

        rows.append(
            {
                "supplier": supplier,
                "po_number": po_number,
                "batch": batch,
                "source_file": source_file,
                "batch_status": batch_status,
                "image_count": image_count,
                "good_image_count": good_count,
                "review_image_count": review_count,
                "bad_image_count": bad_count,
                "avg_dark_pixel_ratio": round(float(group["dark_pixel_ratio"].mean()), 4),
                "avg_brown_pixel_ratio": round(float(group["brown_pixel_ratio"].mean()), 4),
                "max_dark_pixel_ratio": round(float(group["dark_pixel_ratio"].max()), 4),
                "max_brown_pixel_ratio": round(float(group["brown_pixel_ratio"].max()), 4),
                "main_reasons": "; ".join(group["image_reasons"].dropna().astype(str).unique()[:4]),
            }
        )

    return pd.DataFrame(rows).sort_values(["supplier", "po_number", "batch"])


def scan_digital_defects_reports(
    reports_root: str | Path = DEFAULT_REPORTS_ROOT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    limit_files: int | None = None,
) -> dict[str, Any]:
    """Scan the daily reports folder and classify all report images."""
    reports_root = Path(reports_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = [
        path
        for path in reports_root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_REPORT_EXTENSIONS
    ]
    files = sorted(files)
    if limit_files is not None:
        files = files[:limit_files]

    image_rows: list[dict[str, Any]] = []
    for report_file in files:
        image_rows.extend(classify_report_file(report_file, reports_root, output_dir))

    image_df = pd.DataFrame(image_rows)
    batch_df = summarize_batch_classifications(image_df)

    image_output = output_dir / "image_classifications.csv"
    batch_output = output_dir / "batch_classifications.csv"
    image_df.to_csv(image_output, index=False)
    batch_df.to_csv(batch_output, index=False)

    return {
        "reports_root": str(reports_root),
        "files_scanned": len(files),
        "images_classified": int(len(image_df)),
        "batch_count": int(len(batch_df)),
        "image_output": str(image_output),
        "batch_output": str(batch_output),
        "image_classifications": image_df,
        "batch_classifications": batch_df,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Classify digital defects report images by visible defect signals.")
    parser.add_argument("--reports-root", default=str(DEFAULT_REPORTS_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--limit-files", type=int, default=None)
    args = parser.parse_args()

    result = scan_digital_defects_reports(args.reports_root, args.output_dir, args.limit_files)
    print(f"Scanned {result['files_scanned']} files.")
    print(f"Classified {result['images_classified']} images across {result['batch_count']} batches.")
    print(f"Batch output: {result['batch_output']}")
