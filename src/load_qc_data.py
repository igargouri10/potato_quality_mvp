from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


COLUMN_ALIASES = {
    "batch": ["batch", "batch_number", "batch_no"],
    "po_number": ["po_number", "purchase_order", "purchase_order_number", "po"],
    "po_line": ["po_line", "po_line_number", "line", "line_number"],
    "plant_name": ["plant_name", "plant", "site", "plant_site"],
    "grower": ["grower", "vendor_name", "vendor", "supplier"],
    "variety": ["variety", "variety_name"],
    "field": ["field", "farm", "field_farm"],
    "farm_label": [
        "farm_label",
        "supplier_site",
        "supplier_site_label",
        "supplier_site_name",
        "field",
        "farm",
        "field_farm",
        "grower",
    ],
    "scheduled_delivery_date": [
        "scheduled_delivery_date",
        "scheduled_deliv_date",
        "scheduled_delivery",
        "goods_rcpt_date",
        "actual_unload_date",
        "delivery_date",
    ],
    "appearance_score": [
        "appearance_score",
        "asavg",
        "appearance",
        "avg_appearance",
        "appearance_avg",
    ],
}

ID_COLUMNS = ["batch", "batch_number", "po_number", "purchase_order", "po_line", "po_line_number"]


def normalize_column_name(name: object) -> str:
    value = str(name).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def _coerce_identifier(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text


def _add_alias_columns(df: pd.DataFrame) -> pd.DataFrame:
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            if alias in df.columns:
                df[canonical] = df[alias]
                break
    return df


def clean_qc_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns and clean common identifier fields."""
    df = df.copy()
    df.columns = [normalize_column_name(column) for column in df.columns]
    df = _add_alias_columns(df)

    for column in df.columns:
        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
            df[column] = df[column].astype("string").str.strip()

    for column in set(ID_COLUMNS + ["batch", "po_number", "po_line"]):
        if column in df.columns:
            df[column] = df[column].map(_coerce_identifier)

    if "plant_name" in df.columns:
        df["plant_name"] = df["plant_name"].astype(str).str.upper().str.strip()

    if "farm_label" in df.columns:
        df["farm_label"] = df["farm_label"].astype(str).str.upper().str.replace(r"\s+", " ", regex=True).str.strip()

    if "scheduled_delivery_date" in df.columns:
        df["scheduled_delivery_date"] = pd.to_datetime(
            df["scheduled_delivery_date"],
            errors="coerce",
        )

    if "appearance_score" in df.columns:
        df["appearance_score"] = pd.to_numeric(df["appearance_score"], errors="coerce")

    return df


def load_qc_data(path_or_buffer: str | Path | object) -> pd.DataFrame:
    """Load and clean CSV/XLS/XLSX quality data."""
    name = getattr(path_or_buffer, "name", str(path_or_buffer)).lower()
    if name.endswith(".csv"):
        df = pd.read_csv(path_or_buffer, dtype=str)
    elif name.endswith((".xlsx", ".xls")):
        sheets = pd.read_excel(path_or_buffer, sheet_name=None, dtype=str)
        df = _select_best_excel_sheet(sheets)
    else:
        raise ValueError("QC file must be a .csv, .xlsx, or .xls file.")
    return clean_qc_dataframe(df)


def _select_best_excel_sheet(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Choose the sheet most likely to contain row-level QC records."""
    if not sheets:
        raise ValueError("Excel workbook did not contain any sheets.")

    best_sheet = None
    best_score = -1
    required_hints = {
        "batch",
        "po_number",
        "supplier_site",
        "supplier_site_label",
        "asavg",
        "plant_name",
        "scheduled_delivery_date",
    }

    for sheet_name, sheet_df in sheets.items():
        normalized_columns = {normalize_column_name(column) for column in sheet_df.columns}
        score = len(required_hints & normalized_columns) * 10 + len(sheet_df.columns)
        if "raw" in sheet_name.lower():
            score += 25
        if score > best_score:
            best_sheet = sheet_df
            best_score = score

    return best_sheet if best_sheet is not None else next(iter(sheets.values()))
