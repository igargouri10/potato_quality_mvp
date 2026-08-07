from __future__ import annotations

import re
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd

from src.extract_pdf import extract_images
from src.defect_image_classifier import parse_report_path


DEFAULT_OUTPUT_DIR = Path("data/processed/chip_segments")
CHIP_LABELS = ["Clean", "Black", "Brown", "Green", "Other"]


def _safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)


def extract_image_times_by_page(pdf_path: str | Path) -> dict[int, list[str]]:
    """Read the fry times printed above image groups on each PDF page."""
    import pdfplumber

    times_by_page: dict[int, list[str]] = {}
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            times = re.findall(r"\b(?:[01]\d|2[0-3]):[0-5]\d\b", text)
            if times:
                times_by_page[page_number] = times
    return times_by_page


def _add_image_times(report_images: list[dict[str, Any]], source_path: Path) -> list[dict[str, Any]]:
    if source_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
        return report_images

    try:
        times_by_page = extract_image_times_by_page(source_path)
    except Exception:
        times_by_page = {}

    image_counts_by_page: dict[int, int] = {}
    enriched: list[dict[str, Any]] = []
    for image_info in report_images:
        page = int(image_info.get("page") or 0)
        image_counts_by_page[page] = image_counts_by_page.get(page, 0) + 1
        page_position = image_counts_by_page[page] - 1
        page_times = times_by_page.get(page, [])
        enriched.append(
            {
                **image_info,
                "timestamp": page_times[page_position] if page_position < len(page_times) else "",
            }
        )
    return enriched


def _relative_web_path(path: str | Path, html_dir: Path) -> str:
    path = Path(path)
    if not path.is_absolute():
        path = path.resolve()
    relative = Path(os.path.relpath(path, html_dir.resolve()))
    return quote(relative.as_posix(), safe="/._-")


def segment_chip_pieces(
    image_path: str | Path,
    output_dir: str | Path,
    min_area: int = 450,
    padding: int = 8,
) -> list[dict[str, Any]]:
    """Segment likely individual chip pieces from one white-background image."""
    import cv2
    import numpy as np

    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # The report photos use a pale white/blue background. Chip pieces are usually
    # warmer, slightly saturated, and/or visibly darker than that background.
    warm_chip_mask = (
        (hue >= 8)
        & (hue <= 42)
        & (saturation >= 18)
        & (value <= 252)
    )
    shadow_edge_mask = (gray < 232) & (saturation >= 8)
    mask = (warm_chip_mask | shadow_edge_mask).astype("uint8") * 255

    kernel = np.ones((3, 3), dtype="uint8")
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rows: list[dict[str, Any]] = []
    chip_number = 1
    for contour in sorted(contours, key=lambda c: cv2.boundingRect(c)[1:3]):
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w < 12 or h < 12:
            continue
        if w > width * 0.95 or h > height * 0.95:
            continue

        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(width, x + w + padding)
        y1 = min(height, y + h + padding)

        crop = image[y0:y1, x0:x1]
        crop_name = f"{image_path.stem}_chip_{chip_number:04d}.jpg"
        crop_path = output_dir / crop_name
        cv2.imwrite(str(crop_path), crop)

        weak_label, label_reason = weak_label_chip_crop(crop_path)
        rows.append(
            {
                "parent_image": str(image_path),
                "chip_id": f"{image_path.stem}_chip_{chip_number:04d}",
                "chip_crop_path": str(crop_path),
                "x": x0,
                "y": y0,
                "width": x1 - x0,
                "height": y1 - y0,
                "contour_area": round(area, 2),
                "weak_label": weak_label,
                "weak_label_reason": label_reason,
                "true_label": "",
            }
        )
        chip_number += 1

    return rows


def weak_label_chip_crop(image_path: str | Path) -> tuple[str, str]:
    """Generate a rough starter label for sorting; human labels remain the source of truth."""
    import cv2
    import numpy as np

    image = cv2.imread(str(image_path))
    if image is None:
        return "Other", "crop could not be read"

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    dark_ratio = float(np.mean(gray < 75))
    brown_ratio = float(np.mean((hue >= 5) & (hue <= 25) & (saturation > 45) & (value < 190)))
    green_ratio = float(np.mean((hue >= 35) & (hue <= 85) & (saturation > 35) & (value < 235)))

    if dark_ratio > 0.035:
        return "Black", f"dark ratio {dark_ratio:.3f}"
    if brown_ratio > 0.04:
        return "Brown", f"brown ratio {brown_ratio:.3f}"
    if green_ratio > 0.025:
        return "Green", f"green ratio {green_ratio:.3f}"
    return "Clean", "no strong black/brown/green color signal"


def create_chip_labeling_dataset(
    source_path: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Extract report images, segment chips, and write a chip-level labeling manifest."""
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_stem = _safe_stem(source_path)
    image_dir = output_dir / source_stem / "report_images"
    crop_dir = output_dir / source_stem / "chip_crops"

    if source_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
        report_images = [{"path": str(source_path), "page": "", "image_index": 1, "timestamp": ""}]
    else:
        extracted = extract_images(source_path, image_dir, filter_non_potato=True)
        report_images = _add_image_times(extracted["saved_images"], source_path)

    rows: list[dict[str, Any]] = []
    report_metadata = parse_report_path(source_path)
    for image_info in report_images:
        image_path = Path(image_info["path"])
        chip_rows = segment_chip_pieces(image_path, crop_dir)
        for row in chip_rows:
            row.update(
                {
                    **report_metadata,
                    "source_file": str(source_path),
                    "page": image_info.get("page", ""),
                    "image_index": image_info.get("image_index", ""),
                    "timestamp": image_info.get("timestamp", ""),
                    "allowed_labels": "|".join(CHIP_LABELS),
                }
            )
        rows.extend(chip_rows)

    manifest = pd.DataFrame(rows)
    manifest_path = output_dir / source_stem / "chip_labeling_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    html_path = output_dir / source_stem / "chip_review.html"
    generate_chip_review_html(manifest, html_path)

    return {
        "source_file": str(source_path),
        "report_image_count": len(report_images),
        "chip_count": len(manifest),
        "manifest_path": str(manifest_path),
        "html_path": str(html_path),
        "crop_dir": str(crop_dir),
        "manifest": manifest,
    }


def generate_chip_review_html(manifest: pd.DataFrame | str | Path, output_path: str | Path) -> str:
    """Create a local web page showing each parent image and its cropped chip table."""
    if not isinstance(manifest, pd.DataFrame):
        manifest = pd.read_csv(manifest, dtype=str).fillna("")
    else:
        manifest = manifest.fillna("")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html_dir = output_path.parent

    if manifest.empty:
        source_file = ""
        batch = ""
    else:
        source_file = str(manifest.iloc[0].get("source_file", ""))
        batch = str(manifest.iloc[0].get("batch", ""))

    sections: list[str] = []
    for parent_image, group in manifest.groupby("parent_image", sort=False):
        first = group.iloc[0]
        timestamp = str(first.get("timestamp", ""))
        page = str(first.get("page", ""))
        image_index = str(first.get("image_index", ""))
        parent_src = _relative_web_path(parent_image, html_dir)

        id_cells = []
        crop_cells = []
        judgement_cells = []
        for _, row in group.iterrows():
            crop_src = _relative_web_path(row["chip_crop_path"], html_dir)
            chip_id = str(row["chip_id"])
            judgement = str(row.get("true_label") or row.get("weak_label") or "")
            id_cells.append(f"<th>{chip_id}</th>")
            crop_cells.append(f'<td><img class="chip" src="{crop_src}" alt="{chip_id}"></td>')
            judgement_cells.append(f'<td><span class="label label-{judgement.lower()}">{judgement}</span></td>')

        section = f"""
        <section class="image-section">
          <div class="section-heading">
            <div>
              <div class="eyebrow">Batch</div>
              <h2>{batch or "Unknown Batch"}</h2>
            </div>
            <div class="meta-grid">
              <div><span>Time</span><strong>{timestamp or "Unknown"}</strong></div>
              <div><span>Page</span><strong>{page or "N/A"}</strong></div>
              <div><span>Image</span><strong>{image_index or "N/A"}</strong></div>
              <div><span>Chips</span><strong>{len(group)}</strong></div>
            </div>
          </div>
          <div class="whole-image-wrap">
            <img class="whole-image" src="{parent_src}" alt="Whole chip image">
          </div>
          <div class="table-scroll">
            <table>
              <tbody>
                <tr class="id-row"><th class="row-label">ID</th>{''.join(id_cells)}</tr>
                <tr><th class="row-label">Crop</th>{''.join(crop_cells)}</tr>
                <tr><th class="row-label">Judgement</th>{''.join(judgement_cells)}</tr>
              </tbody>
            </table>
          </div>
        </section>
        """
        sections.append(section)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Chip Review - Batch {batch}</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: #f6f7f8;
      color: #1f2933;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 5;
      background: #ffffff;
      border-bottom: 1px solid #d9dee5;
      padding: 16px 24px;
    }}
    h1, h2 {{ margin: 0; }}
    .source {{ margin-top: 6px; color: #627184; font-size: 13px; overflow-wrap: anywhere; }}
    main {{ padding: 20px 24px 40px; }}
    .image-section {{
      background: #ffffff;
      border: 1px solid #d9dee5;
      border-radius: 8px;
      margin-bottom: 22px;
      overflow: hidden;
    }}
    .section-heading {{
      display: flex;
      justify-content: space-between;
      gap: 20px;
      padding: 16px;
      border-bottom: 1px solid #e3e7ec;
      align-items: center;
    }}
    .eyebrow {{ font-size: 12px; text-transform: uppercase; color: #627184; font-weight: 700; }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(80px, 1fr));
      gap: 10px;
      min-width: 420px;
    }}
    .meta-grid div {{
      border: 1px solid #e3e7ec;
      border-radius: 6px;
      padding: 8px 10px;
      background: #fafbfc;
    }}
    .meta-grid span {{ display: block; font-size: 11px; color: #627184; }}
    .meta-grid strong {{ display: block; margin-top: 3px; font-size: 15px; }}
    .whole-image-wrap {{
      padding: 16px;
      border-bottom: 1px solid #e3e7ec;
      background: #fbfcfd;
    }}
    .whole-image {{
      display: block;
      max-width: 760px;
      max-height: 520px;
      width: auto;
      height: auto;
      border: 1px solid #d9dee5;
      background: white;
    }}
    .table-scroll {{ overflow-x: auto; }}
    table {{
      border-collapse: collapse;
      min-width: 100%;
      table-layout: fixed;
    }}
    th, td {{
      border: 1px solid #e1e5ea;
      padding: 8px;
      text-align: center;
      vertical-align: middle;
      min-width: 116px;
      max-width: 160px;
      font-size: 12px;
      overflow-wrap: anywhere;
    }}
    .row-label {{
      position: sticky;
      left: 0;
      z-index: 2;
      background: #eef2f5;
      min-width: 92px;
      font-weight: 700;
    }}
    .id-row th:not(.row-label) {{ background: #f7f9fb; font-weight: 600; }}
    .chip {{
      display: block;
      margin: 0 auto;
      max-width: 112px;
      max-height: 96px;
      object-fit: contain;
      border: 1px solid #d9dee5;
      background: white;
    }}
    .label {{
      display: inline-block;
      min-width: 62px;
      padding: 5px 8px;
      border-radius: 5px;
      font-weight: 700;
      border: 1px solid #cad2dc;
      background: #f7f9fb;
    }}
    .label-clean {{ background: #e8f6ee; color: #166534; border-color: #b7e0c4; }}
    .label-black {{ background: #e5e7eb; color: #111827; border-color: #9ca3af; }}
    .label-brown {{ background: #f8eadc; color: #8a4b16; border-color: #e5c29e; }}
    .label-green {{ background: #e5f5dc; color: #3f6212; border-color: #b8dd9b; }}
    .label-other {{ background: #fff4d6; color: #854d0e; border-color: #f3d37b; }}
    @media (max-width: 780px) {{
      .section-heading {{ display: block; }}
      .meta-grid {{ margin-top: 14px; min-width: 0; grid-template-columns: repeat(2, minmax(80px, 1fr)); }}
      .whole-image {{ max-width: 100%; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Chip Review - Batch {batch or "Unknown"}</h1>
    <div class="source">{source_file}</div>
  </header>
  <main>
    {''.join(sections)}
  </main>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return str(output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Segment individual chip pieces for labeling/model training.")
    parser.add_argument("source_path", help="PDF report or chip image.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    result = create_chip_labeling_dataset(args.source_path, args.output_dir)
    print(f"Report images processed: {result['report_image_count']}")
    print(f"Chip crops created: {result['chip_count']}")
    print(f"Manifest: {result['manifest_path']}")
    print(f"Review page: {result['html_path']}")
    print(f"Crop folder: {result['crop_dir']}")
