from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path
from typing import Any

from src.image_features import analyze_potato_image_relevance

DEFAULT_IMAGE_DIR = Path("data/processed/extracted_images")


FIELD_PATTERNS = {
    "purchase_order": [
        r"purchase\s+order\s*:?\s*(\d{6,})",
        r"\bpo(?:\s+number)?\s*:?\s*(\d{6,})",
    ],
    "po_line": [
        r"\bline\s*:?\s*0*(\d{1,6})\b",
        r"\bpo\s+line(?:\s+number)?\s*:?\s*0*(\d{1,6})\b",
    ],
    "batch_number": [
        r"batch(?:\s+number)?\s*:?\s*0*(\d{4,})",
        r"\bbatch\s*:?\s*0*(\d{4,})",
    ],
    "site": [
        r"\bsite\s*:?\s*([A-Za-z][A-Za-z\s\-]+?)(?=\s+(?:expected|purchase|line|batch|image|po)\b|\n|$)",
        r"\bplant\s*:?\s*([A-Za-z][A-Za-z\s\-]+?)(?=\s+(?:expected|purchase|line|batch|image|po)\b|\n|$)",
    ],
    "expected_date": [
        r"expected\s+need\s+by\s+date\s*:?\s*([0-9]{1,2}[-/][A-Za-z]{3}[-/][0-9]{2,4})",
        r"expected\s+date\s*:?\s*([0-9]{1,2}[-/][A-Za-z]{3}[-/][0-9]{2,4})",
    ],
}


def extract_text(pdf_path: str | Path) -> str:
    """Extract text from all pages in a PDF."""
    import pdfplumber

    pages: list[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(text)
    return "\n".join(pages)


def _first_match(text: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip().rstrip(" .,:;")
    return None


def parse_metadata(text: str, pdf_file: str | None = None) -> dict[str, Any]:
    """Parse common report fields from extracted PDF text."""
    metadata: dict[str, Any] = {"pdf_file": pdf_file}
    for field, patterns in FIELD_PATTERNS.items():
        metadata[field] = _first_match(text, patterns)

    metadata["image_timestamps"] = sorted(set(re.findall(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", text)))
    metadata["image_timestamps"] = [f"{hour.zfill(2)}:{minute}" for hour, minute in metadata["image_timestamps"]]
    return metadata


def extract_images(
    pdf_path: str | Path,
    output_dir: str | Path = DEFAULT_IMAGE_DIR,
    filter_non_potato: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Extract relevant embedded PDF images and return saved/skipped details."""
    import fitz

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_images: list[dict[str, Any]] = []
    skipped_images: list[dict[str, Any]] = []
    document = fitz.open(str(pdf_path))
    try:
        for page_index in range(len(document)):
            page = document[page_index]
            for image_index, image_info in enumerate(page.get_images(full=True), start=1):
                xref = image_info[0]
                image = document.extract_image(xref)
                image_bytes = image["image"]
                ext = image.get("ext", "png")
                stem = pdf_path.stem
                output_path = output_dir / f"{stem}_page{page_index + 1}_image{image_index}.{ext}"

                relevance = {"is_relevant": True, "reason": "Filtering disabled."}
                if filter_non_potato:
                    try:
                        relevance = analyze_potato_image_relevance(BytesIO(image_bytes))
                    except Exception as exc:
                        relevance = {
                            "is_relevant": False,
                            "reason": f"Image could not be inspected: {exc}",
                        }

                if not relevance["is_relevant"]:
                    skipped_images.append(
                        {
                            "page": page_index + 1,
                            "image_index": image_index,
                            "reason": relevance["reason"],
                            "details": relevance,
                        }
                    )
                    continue

                output_path.write_bytes(image_bytes)
                saved_images.append(
                    {
                        "path": str(output_path),
                        "page": page_index + 1,
                        "image_index": image_index,
                        "details": relevance,
                    }
                )
    finally:
        document.close()

    return {"saved_images": saved_images, "skipped_images": skipped_images}


def extract_pdf_report(pdf_path: str | Path, output_dir: str | Path = DEFAULT_IMAGE_DIR) -> dict[str, Any]:
    """Extract text metadata and images from a defect report PDF."""
    pdf_path = Path(pdf_path)
    text = extract_text(pdf_path)
    metadata = parse_metadata(text, pdf_file=pdf_path.name)
    metadata["raw_text"] = text
    image_result = extract_images(pdf_path, output_dir)
    metadata["extracted_images"] = [image["path"] for image in image_result["saved_images"]]
    metadata["image_extraction_details"] = image_result["saved_images"]
    metadata["skipped_images"] = image_result["skipped_images"]
    return metadata


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Extract metadata and images from a potato QC PDF.")
    parser.add_argument("pdf_path")
    parser.add_argument("--output-dir", default=str(DEFAULT_IMAGE_DIR))
    args = parser.parse_args()

    print(json.dumps(extract_pdf_report(args.pdf_path, args.output_dir), indent=2))
