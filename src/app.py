from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classify_rules import classify_quality
from src.defect_image_classifier import DEFAULT_REPORTS_ROOT, scan_digital_defects_reports
from src.extract_pdf import extract_pdf_report
from src.farm_analysis import analyze_farm_appearance, filter_qc_for_farm_analysis
from src.image_features import compute_image_features
from src.load_qc_data import load_qc_data
from src.match_records import match_qc_record


EXTRACTED_IMAGE_DIR = PROJECT_ROOT / "data" / "processed" / "extracted_images"


def _save_upload(uploaded_file, suffix: str) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="potato_quality_"))
    path = temp_dir / f"upload{suffix}"
    path.write_bytes(uploaded_file.getbuffer())
    return path


def _metadata_table(metadata: dict) -> pd.DataFrame:
    display_fields = {
        "Batch Number": metadata.get("batch_number"),
        "PO Number": metadata.get("purchase_order"),
        "PO Line": metadata.get("po_line"),
        "Plant/Site": metadata.get("site"),
        "Expected Date": metadata.get("expected_date"),
        "Image Timestamps": ", ".join(metadata.get("image_timestamps") or []),
        "PDF File": metadata.get("pdf_file"),
    }
    return pd.DataFrame(
        [{"Field": field, "Value": value or ""} for field, value in display_fields.items()]
    )


def _show_status(status: str) -> None:
    if status == "Good":
        st.success("Quality Status: Good")
    elif status == "Warning":
        st.warning("Quality Status: Warning")
    elif status == "Poor":
        st.error("Quality Status: Poor")
    else:
        st.info("Quality Status: Unknown")


def _single_report_workflow() -> None:
    st.header("Upload Files")
    left, right = st.columns(2)
    with left:
        report_file = st.file_uploader("Defect PDF report", type=["pdf"])
    with right:
        qc_file = st.file_uploader("QC Excel or CSV file", type=["csv", "xlsx", "xls"])

    if not report_file or not qc_file:
        st.info("Upload a PDF report and a QC file to run the matching workflow.")
        return

    with st.spinner("Extracting report data and matching QC record..."):
        report_path = _save_upload(report_file, ".pdf")
        metadata = extract_pdf_report(report_path, EXTRACTED_IMAGE_DIR)
        qc_df = load_qc_data(qc_file)
        match = match_qc_record(metadata, qc_df)
        assessment = classify_quality(match.get("qc_record", {}))

        image_features = []
        for image_path in metadata.get("extracted_images", []):
            try:
                image_features.append(compute_image_features(image_path))
            except ValueError:
                continue

    st.header("Extracted PDF Metadata")
    st.dataframe(_metadata_table(metadata), hide_index=True, use_container_width=True)

    st.header("Quality Assessment")
    _show_status(assessment["quality_status"])
    for reason in assessment["reasons"]:
        st.write(f"- {reason}")

    if image_features:
        defect_count = sum(1 for row in image_features if row["possible_defect_flag"])
        if defect_count:
            st.warning(f"{defect_count} extracted image(s) have simple visual defect flags.")

    st.header("Matched Batch Record")
    if match["match_status"] == "none":
        st.warning("No QC row matched the extracted metadata.")
    else:
        st.caption(
            f"Match status: {match['match_status']} | "
            f"Matched fields: {', '.join(match['matched_fields'])}"
        )
        st.dataframe(pd.DataFrame([match["qc_record"]]), use_container_width=True)

    st.header("Extracted Images")
    image_paths = metadata.get("extracted_images", [])
    skipped_images = metadata.get("skipped_images", [])
    if skipped_images:
        st.caption(f"Ignored {len(skipped_images)} embedded image(s) that did not look like potato inspection photos.")

    if image_paths:
        columns = st.columns(min(3, len(image_paths)))
        for index, image_path in enumerate(image_paths):
            with columns[index % len(columns)]:
                st.image(image_path, caption=Path(image_path).name, use_container_width=True)
    else:
        st.info("No embedded images were found in this PDF.")

    if image_features:
        st.subheader("Basic Image Features")
        st.dataframe(pd.DataFrame(image_features), use_container_width=True)

    with st.expander("Debug Matching Details"):
        st.json(match)
        st.subheader("Cleaned QC Columns")
        st.write(list(qc_df.columns))

    if skipped_images:
        with st.expander("Ignored PDF Images"):
            st.dataframe(pd.DataFrame(skipped_images), use_container_width=True)


def _farm_analysis_workflow() -> None:
    st.header("Farm Appearance Comparison")
    qc_file = st.file_uploader(
        "QC Excel or CSV file",
        type=["csv", "xlsx", "xls"],
        key="farm_qc_file",
    )

    if not qc_file:
        st.info("Upload a QC file to compare lower-appearance batches against same-farm, same-plant batches.")
        return

    qc_df = load_qc_data(qc_file)
    required_columns = {"farm_label", "plant_name", "appearance_score"}
    missing_columns = sorted(required_columns - set(qc_df.columns))
    if missing_columns:
        st.error(
            "The QC file is missing required mapped field(s): "
            + ", ".join(missing_columns)
            + ". Farm mapping can come from Supplier Site, Supplier Site Label, Farm, Field, or Grower. "
            + "Appearance score can come from ASAVG or Appearance Score."
        )
        st.subheader("Detected Columns")
        st.write(list(qc_df.columns))
        return

    with st.sidebar:
        st.subheader("Farm Analysis Filters")
        plants = sorted([value for value in qc_df["plant_name"].dropna().astype(str).unique() if value])
        selected_plants = st.multiselect("Plant Name", plants)

        po_values = sorted([value for value in qc_df.get("po_number", pd.Series(dtype=str)).dropna().astype(str).unique() if value])
        selected_pos = st.multiselect("PO Number", po_values)

        start_date = None
        end_date = None
        if "scheduled_delivery_date" in qc_df.columns and qc_df["scheduled_delivery_date"].notna().any():
            min_date = qc_df["scheduled_delivery_date"].min().date()
            max_date = qc_df["scheduled_delivery_date"].max().date()
            date_range = st.date_input("Scheduled Delivery Date", value=(min_date, max_date))
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date, end_date = date_range

        low_threshold = st.number_input("Low appearance below", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
        comparison_threshold = st.number_input(
            "Comparison appearance at least",
            min_value=0.0,
            max_value=100.0,
            value=max(70.0, low_threshold),
            step=1.0,
        )

    filtered = filter_qc_for_farm_analysis(
        qc_df,
        plant_names=selected_plants,
        po_numbers=selected_pos,
        start_date=start_date,
        end_date=end_date,
    )

    st.caption(f"Analyzing {len(filtered):,} rows after filters.")
    result = analyze_farm_appearance(
        filtered,
        low_threshold=low_threshold,
        stronger_min_score=comparison_threshold,
    )

    if result["message"]:
        st.warning(result["message"])
        return

    summaries = result["farm_summaries"]
    comparisons = result["batch_comparisons"]
    if not summaries:
        st.info("No lower-appearance batches were found for the current filters and threshold.")
        return

    summary_df = pd.DataFrame(summaries)
    st.subheader("Farm Summary")
    st.dataframe(
        summary_df[
            [
                "farm_label",
                "plant_name",
                "low_batch_count",
                "comparison_batch_count",
                "low_avg_appearance",
                "comparison_avg_appearance",
                "appearance_gap",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Farm-Level Explanation Paragraphs")
    for summary in summaries:
        st.markdown(f"**{summary['farm_label']} | {summary['plant_name']}**")
        st.write(summary["paragraph"])

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.subheader("Average Appearance Gap")
        chart_df = summary_df[["farm_label", "plant_name", "low_avg_appearance", "comparison_avg_appearance"]].copy()
        chart_df["farm_plant"] = chart_df["farm_label"] + " | " + chart_df["plant_name"]
        st.bar_chart(
            chart_df.set_index("farm_plant")[["low_avg_appearance", "comparison_avg_appearance"]],
            use_container_width=True,
        )

    with chart_right:
        st.subheader("Batch-Level Appearance")
        if not comparisons.empty:
            scatter_df = comparisons.copy()
            scatter_df["farm_plant"] = scatter_df["farm_label"] + " | " + scatter_df["plant_name"]
            st.scatter_chart(
                scatter_df,
                x="farm_plant",
                y="appearance_score",
                color="comparison_group",
                use_container_width=True,
            )

    st.subheader("Batch Comparison Data")
    st.dataframe(comparisons, hide_index=True, use_container_width=True)


def _defect_image_workflow() -> None:
    st.header("Digital Defects Image Classification")
    reports_root = st.text_input("Digital Defects Reports folder", value=str(DEFAULT_REPORTS_ROOT))
    limit_files = st.number_input("File limit for test run", min_value=0, value=25, step=5)
    run_scan = st.button("Scan And Classify Images")

    if not run_scan:
        st.info("Run a small test first, then set the file limit to 0 to scan the full folder.")
        return

    limit = None if limit_files == 0 else int(limit_files)
    with st.spinner("Scanning reports and classifying visible defect signals..."):
        result = scan_digital_defects_reports(reports_root=reports_root, limit_files=limit)

    st.success(
        f"Scanned {result['files_scanned']} file(s), classified {result['images_classified']} image(s), "
        f"and summarized {result['batch_count']} batch(es)."
    )
    st.caption(f"Batch output: {result['batch_output']}")
    st.caption(f"Image output: {result['image_output']}")

    batch_df = result["batch_classifications"]
    image_df = result["image_classifications"]
    if not batch_df.empty:
        st.subheader("Batch Classification Summary")
        st.dataframe(batch_df, hide_index=True, use_container_width=True)
        status_counts = batch_df["batch_status"].value_counts().rename_axis("status").reset_index(name="batches")
        st.bar_chart(status_counts.set_index("status"), use_container_width=True)

    if not image_df.empty:
        st.subheader("Image Classification Detail")
        st.dataframe(image_df, hide_index=True, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Potato Quality MVP", layout="wide")
    st.title("Potato Batch Quality Intelligence")

    report_tab, farm_tab, defect_tab = st.tabs(
        ["Single PDF Match", "Farm Appearance Analysis", "Defect Image Classifier"]
    )
    with report_tab:
        _single_report_workflow()
    with farm_tab:
        _farm_analysis_workflow()
    with defect_tab:
        _defect_image_workflow()


if __name__ == "__main__":
    main()
