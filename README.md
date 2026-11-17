# Potato Batch Quality Intelligence MVP

This MVP connects potato defect report PDFs to batch-level QC data, finds the matching QC row, and returns a rule-based quality status with reasons.

The first version is intentionally not an image-only AI model. It proves the repeatable workflow:

```text
PDF report -> metadata and image extraction -> QC file matching -> rule classification -> dashboard
```

## What It Includes

- PDF text extraction for PO, line, batch, plant/site, expected date, and image timestamps
- Embedded image extraction from PDF reports
- CSV/XLS/XLSX QC loading with cleaned column names and string-safe IDs
- Deterministic record matching by batch, PO number, PO line, and plant
- Rule-based quality labels: `Good`, `Warning`, `Poor`, and `Unknown`
- Basic image features such as brightness, dark pixels, pale pixels, and brown-region flags
- A Streamlit dashboard for uploading a PDF and QC file

## Project Structure

```text
potato_quality_mvp/
  data/
    raw/
      pdfs/
      images/
      excel/
    processed/
      extracted_images/
      matched_records/
  notebooks/
  src/
    __init__.py
    app.py
    classify_rules.py
    extract_pdf.py
    image_features.py
    load_qc_data.py
    match_records.py
  tests/
  README.md
  requirements.txt
  .gitignore
```

Raw and processed company data are ignored by Git so confidential files stay local.

## Install

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run The App

```powershell
streamlit run src/app.py
```

Upload:

- A PDF defect report
- A QC CSV, XLSX, or XLS file

The app will show extracted metadata, extracted PDF images, the matched QC row, quality status, classification reasons, and debug matching details.

## Expected Data

PDF reports should ideally contain fields like:

- Purchase Order
- Line or PO Line
- Batch Number
- Site or Plant
- Expected Need By Date
- Image timestamps

QC files can use common exported column names such as:

- `Batch` or `Batch Number`
- `PO Number`
- `PO Line Number`
- `PLANT NAME`
- `Grower` or `Vendor Name`
- `VARIETY`
- `FIELD`
- `TIER`
- `Disposition`
- `Total Defects`
- `GREENING`
- `Wet Breakdown`
- `ASAVG`
- `WSTTOGOAL`

## Matching Logic

The matcher scores each QC row using extracted PDF metadata:

- Strong match: batch, PO number, PO line, and plant all match
- Medium match: batch and PO number match
- Weak match: batch only
- No match: no reliable batch match

IDs are cleaned as strings to avoid Excel scientific notation and `.0` suffix issues.

## Quality Classification

The rule engine uses configurable thresholds and returns both a label and reasons.

Examples:

- Rejected or held disposition -> `Poor`
- Tier 1 or 2 -> usually `Good`
- Tier 3 -> `Warning`
- Tier 4 or 5 -> `Poor`
- Wet breakdown, greening, or high total defects -> `Poor`
- Low ASAVG or waste above goal -> `Warning`

For this MVP, QC result columns are allowed because the goal is reporting and explanation. Later predictive models should avoid leakage-prone result fields as inputs.

## Test

```powershell
pytest
```

The included tests cover the known first case:

```text
Batch: 4405357
PO Number: 4506845317
PO Line: 750
Plant/Site: Killingly
```

## Future ML Improvements

After the matching pipeline is reliable across several PDFs and QC exports:

- Build a labeled image dataset from extracted images
- Train a lightweight image classifier such as MobileNet, ResNet, or EfficientNet
- Combine image features with tabular fields
- Keep leakage-prone QC outcome columns as labels or evaluation targets, not predictive inputs

## Local recording workflow (October 2026)

A recording workflow has been prepared in the parent workspace's `DEMO_GUIDE.md`.
These are bounded demonstrations, not a claim that every experimental workflow has been reproduced.
Keep AI-assisted implementation attribution and the documented limitations in public descriptions.
