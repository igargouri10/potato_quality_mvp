from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from src.chip_segmentation import create_chip_labeling_dataset, segment_chip_pieces
from src.classify_rules import classify_quality
from src.defect_image_classifier import classify_image_features, parse_report_path
from src.evaluate_defect_classifier import evaluate_defect_classifier
from src.extract_pdf import parse_metadata
from src.farm_analysis import analyze_farm_appearance, filter_qc_for_farm_analysis
from src.image_features import analyze_potato_image_relevance
from src.load_qc_data import clean_qc_dataframe
from src.match_records import match_qc_record


def test_parse_known_pdf_metadata_text():
    text = """
    Purchase Order: 4506845317
    Line: 750
    Batch Number: 4405357
    Site: Killingly
    Expected Need By Date: 02-Jul-26
    Image timestamps: 20:19, 20:20, 20:21
    """

    metadata = parse_metadata(text, "sample_report.pdf")

    assert metadata["purchase_order"] == "4506845317"
    assert metadata["po_line"] == "750"
    assert metadata["batch_number"] == "4405357"
    assert metadata["site"] == "Killingly"
    assert metadata["image_timestamps"] == ["20:19", "20:20", "20:21"]


def test_clean_and_match_known_record():
    qc = pd.DataFrame(
        [
            {
                "PO Concat Info": "4506845317-00750-0004405357",
                "Batch": "4405357",
                "PO Number": "4506845317.0",
                "PO Line Number": "750",
                "PLANT NAME": " killingly ",
                "TIER": "3",
                "ASAVG": "65",
            }
        ]
    )
    cleaned = clean_qc_dataframe(qc)
    metadata = {
        "purchase_order": "4506845317",
        "po_line": "750",
        "batch_number": "4405357",
        "site": "Killingly",
    }

    match = match_qc_record(metadata, cleaned)

    assert match["match_status"] == "strong"
    assert match["matched_row_index"] == 0
    assert set(match["matched_fields"]) == {"batch", "po_number", "po_line", "plant_name"}


def test_classify_warning_with_reasons():
    result = classify_quality({"tier": "3", "asavg": "65"})

    assert result["quality_status"] == "Warning"
    assert any("Tier" in reason for reason in result["reasons"])
    assert any("ASAVG" in reason for reason in result["reasons"])


def test_potato_image_relevance_rejects_blank_report_artifact():
    image = Image.new("RGB", (320, 240), "white")

    result = analyze_potato_image_relevance(image)

    assert result["is_relevant"] is False
    assert "blank" in result["reason"].lower()


def test_potato_image_relevance_accepts_chip_colored_photo():
    image = Image.new("RGB", (320, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((35, 35, 135, 125), fill=(205, 165, 82))
    draw.ellipse((155, 60, 270, 175), fill=(190, 130, 62))
    draw.ellipse((80, 145, 210, 220), fill=(220, 185, 95))

    result = analyze_potato_image_relevance(image)

    assert result["is_relevant"] is True
    assert result["potato_pixel_ratio"] >= 0.08


def test_supplier_site_maps_to_farm_label_and_appearance_score():
    qc = pd.DataFrame(
        [
            {
                "Supplier Site Label": "  Farm A  ",
                "PLANT NAME": "killingly",
                "ASAVG": "72",
                "SCHEDULED_DELIVERY_DATE": "2026-07-02",
            }
        ]
    )

    cleaned = clean_qc_dataframe(qc)

    assert cleaned.loc[0, "farm_label"] == "FARM A"
    assert cleaned.loc[0, "plant_name"] == "KILLINGLY"
    assert cleaned.loc[0, "appearance_score"] == 72
    assert str(cleaned.loc[0, "scheduled_delivery_date"].date()) == "2026-07-02"


def test_farm_appearance_analysis_compares_same_farm_and_plant():
    qc = pd.DataFrame(
        [
            {
                "Batch": "100",
                "PO Number": "4501",
                "Supplier Site Label": "Farm A",
                "PLANT NAME": "KILLINGLY",
                "ASAVG": "58",
                "Total Defects": "14",
                "GREENING": "3",
                "SCHEDULED_DELIVERY_DATE": "2026-07-02",
            },
            {
                "Batch": "101",
                "PO Number": "4501",
                "Supplier Site Label": "Farm A",
                "PLANT NAME": "KILLINGLY",
                "ASAVG": "78",
                "Total Defects": "5",
                "GREENING": "0",
                "SCHEDULED_DELIVERY_DATE": "2026-07-03",
            },
            {
                "Batch": "200",
                "PO Number": "4502",
                "Supplier Site Label": "Farm A",
                "PLANT NAME": "OTHER",
                "ASAVG": "82",
                "Total Defects": "1",
                "GREENING": "0",
                "SCHEDULED_DELIVERY_DATE": "2026-07-03",
            },
        ]
    )
    cleaned = clean_qc_dataframe(qc)
    filtered = filter_qc_for_farm_analysis(cleaned, plant_names=["KILLINGLY"], po_numbers=["4501"])

    result = analyze_farm_appearance(filtered, low_threshold=70, stronger_min_score=75)

    assert result["message"] == ""
    assert len(result["farm_summaries"]) == 1
    summary = result["farm_summaries"][0]
    assert summary["farm_label"] == "FARM A"
    assert summary["plant_name"] == "KILLINGLY"
    assert summary["low_batches"] == "100"
    assert summary["comparison_batches"] == "101"
    assert summary["appearance_gap"] == 20
    assert "total defects" in summary["paragraph"]


def test_parse_digital_defects_report_path():
    path = (
        r"C:\Reports\Arkansas\4506752607\_3740404_120260624140707.pdf_"
    )

    result = parse_report_path(path, r"C:\Reports")

    assert result["supplier"] == "Arkansas"
    assert result["po_number"] == "4506752607"
    assert result["batch"] == "3740404"


def test_visual_classifier_labels_good_review_and_bad():
    good = classify_image_features({"dark_pixel_ratio": 0.005, "brown_pixel_ratio": 0.004})
    review = classify_image_features({"dark_pixel_ratio": 0.03, "brown_pixel_ratio": 0.004})
    bad = classify_image_features({"dark_pixel_ratio": 0.004, "brown_pixel_ratio": 0.04})

    assert good["image_status"] == "Likely Good"
    assert review["image_status"] == "Review"
    assert bad["image_status"] == "Likely Bad"


def test_evaluate_defect_classifier_metrics(tmp_path):
    labels_path = tmp_path / "labels.csv"
    predictions_path = tmp_path / "predictions.csv"
    output_path = tmp_path / "metrics.csv"

    pd.DataFrame(
        [
            {"batch": "100", "true_label": "Good"},
            {"batch": "101", "true_label": "Bad"},
            {"batch": "102", "true_label": "Bad"},
        ]
    ).to_csv(labels_path, index=False)
    pd.DataFrame(
        [
            {"batch": "100", "batch_status": "Likely Good"},
            {"batch": "101", "batch_status": "Likely Bad"},
            {"batch": "102", "batch_status": "Likely Good"},
        ]
    ).to_csv(predictions_path, index=False)

    result = evaluate_defect_classifier(labels_path, predictions_path, output_path)

    assert result["metrics"]["records_evaluated"] == 3
    assert result["metrics"]["accuracy"] == 0.6667
    assert result["metrics"]["precision_bad"] == 1.0
    assert result["metrics"]["recall_bad"] == 0.5
    assert result["metrics"]["f1_bad"] == 0.6667


def test_segment_chip_pieces_creates_crops(tmp_path):
    image_path = tmp_path / "chips.jpg"
    image = Image.new("RGB", (280, 180), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((25, 35, 85, 95), fill=(218, 184, 91))
    draw.ellipse((150, 55, 225, 125), fill=(205, 160, 76))
    image.save(image_path)

    rows = segment_chip_pieces(image_path, tmp_path / "crops", min_area=250)

    assert len(rows) >= 2
    assert all(Path(row["chip_crop_path"]).exists() for row in rows)
    assert set(row["weak_label"] for row in rows).issubset({"Clean", "Black", "Brown", "Green", "Other"})


def test_create_chip_labeling_dataset_from_image(tmp_path):
    image_path = tmp_path / "chips.jpg"
    image = Image.new("RGB", (220, 160), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((30, 30, 95, 95), fill=(215, 178, 82))
    image.save(image_path)

    result = create_chip_labeling_dataset(image_path, tmp_path / "dataset")

    assert result["report_image_count"] == 1
    assert result["chip_count"] >= 1
    assert Path(result["manifest_path"]).exists()
    html_path = Path(result["html_path"])
    assert html_path.exists()
    html = html_path.read_text(encoding="utf-8")
    assert "ID" in html
    assert "Crop" in html
    assert "Judgement" in html
