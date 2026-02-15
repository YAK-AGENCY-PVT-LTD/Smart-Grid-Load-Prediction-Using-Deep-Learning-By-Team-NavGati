"""Extract monthly tables from power grid PDF reports.

Reads PDF files from data/raw/<year>/<month>/, infers year/month from the
PDF name or path, and writes cleaned CSVs to data/extracted/<year>/<month>/.
"""

from __future__ import annotations

import calendar
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import camelot
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"
EXTRACTED_DIR = BASE_DIR / "data" / "extracted"


_ORIG_RMTREE = shutil.rmtree


def _quiet_rmtree(path: str, *args, **kwargs) -> None:
    try:
        _ORIG_RMTREE(path, *args, **kwargs)
    except PermissionError:
        return


shutil.rmtree = _quiet_rmtree


def log_info(message: str) -> None:
    print(f"[INFO] {message}")


def log_status(message: str) -> None:
    print(f"[STATUS] {message}")


def log_success(message: str) -> None:
    print(f"[SUCCESS] {message}")


def log_warning(message: str) -> None:
    print(f"[WARNING] {message}")


@dataclass(frozen=True)
class TableSpec:
    name: str
    page: str
    drop_last_rows: int
    drop_columns: Tuple[int, ...]
    replace_dashes_in: Tuple[str, ...] = ()


TABLE_SPECS = (
    TableSpec(
        name="daily_maximum_demand",
        page="23",
        drop_last_rows=6,
        drop_columns=(0, 7, 7, 7),
    ),
    TableSpec(
        name="wind_generation",
        page="21",
        drop_last_rows=8,
        drop_columns=(0, 7),
        replace_dashes_in=("ER", "NER"),
    ),
    TableSpec(
        name="solar_generation",
        page="22",
        drop_last_rows=8,
        drop_columns=(0, 7),
    ),
    TableSpec(
        name="hydro_generation",
        page="19",
        drop_last_rows=8,
        drop_columns=(0, 7),
    ),
    TableSpec(
        name="region_wise_capacity",
        page="12",
        drop_last_rows=0,
        drop_columns=(),
    ),
)


def find_pdfs(raw_root: Path) -> Iterable[Path]:
    return sorted(raw_root.glob("**/*.pdf"))


def infer_year_month(path: Path) -> Optional[Tuple[int, int, str]]:
    month_names = "|".join(calendar.month_name[1:])
    match = re.search(rf"({month_names})[^0-9]*(\d{{4}})", path.stem, re.IGNORECASE)
    if match:
        month_name = match.group(1).capitalize()
        month_num = list(calendar.month_name).index(month_name)
        return int(match.group(2)), month_num, month_name

    year = None
    month_name = None
    for part in path.parts:
        if year is None and re.fullmatch(r"\d{4}", part):
            year = int(part)
        if month_name is None:
            match = re.search(rf"({month_names})", part, re.IGNORECASE)
            if match:
                month_name = match.group(1).capitalize()

    if year and month_name:
        month_num = list(calendar.month_name).index(month_name)
        return year, month_num, month_name
    return None


def month_folder_name(month_num: int, month_name: str) -> str:
    return f"{month_num:02d}_{month_name}"


def clean_table(df: pd.DataFrame, spec: TableSpec) -> pd.DataFrame:
    df = df.copy()

    if spec.name == "region_wise_capacity":
        df = df.drop(index=0, errors="ignore")
        if not df.empty:
            df.columns = df.iloc[0].values
        df = df.drop(index=[1, 2], errors="ignore")
        if len(df.columns) >= 3:
            df = df[[df.columns[2], df.columns[-2]]]
        df.columns = ["Region", "Grand_Total"]
        df = df.iloc[:-3]
        return df

    df = df.drop(index=0)
    df.columns = df.iloc[0].values
    df = df.drop(index=1)

    if spec.name == "daily_maximum_demand":
        if len(df.columns) > 0:
            df = df.drop(columns=df.columns[0])
        for _ in range(3):
            if len(df.columns) > 7:
                df = df.drop(columns=df.columns[7])
    elif spec.name == "wind_generation":
        if len(df.columns) > 0:
            df = df.drop(columns=df.columns[0])
        if len(df.columns) > 7:
            df = df.drop(columns=df.columns[7])
    else:
        for drop_index in sorted(spec.drop_columns, reverse=True):
            if 0 <= drop_index < len(df.columns):
                df = df.drop(columns=df.columns[drop_index])

    empty_headers = df.columns.isna() | (df.columns.astype(str).str.strip() == "")
    if empty_headers.any():
        df = df.loc[:, ~empty_headers]
    df = df.dropna(axis=1, how="all")

    if len(df.columns) > 7:
        df = df.iloc[:, :7]

    if len(df.columns) == 7:
        df.columns = ["Date", "NR", "WR", "SR", "ER", "NER", "All India"]
    elif len(df.columns) == 6:
        df.columns = ["Date", "NR", "WR", "SR", "ER", "All India"]
    else:
        raise ValueError(
            f"Unexpected column count after cleanup: {len(df.columns)}."
        )
    if spec.drop_last_rows:
        df = df.iloc[:-spec.drop_last_rows]

    for column in spec.replace_dashes_in:
        if column in df.columns:
            df[column] = df[column].replace("----------", 0)

    return df


def extract_table(pdf_path: Path, spec: TableSpec) -> pd.DataFrame:
    log_status(f"Extracting {spec.name} from page {spec.page} in {pdf_path.name}.")
    tables = camelot.read_pdf(
        str(pdf_path),
        pages=spec.page,
        flavor="lattice",
    )
    if not tables:
        raise ValueError(f"No tables found on page {spec.page} of {pdf_path.name}.")
    return clean_table(tables[0].df, spec)


def process_pdf(pdf_path: Path, output_root: Path) -> None:
    inferred = infer_year_month(pdf_path)
    if not inferred:
        raise ValueError(f"Could not infer year/month from {pdf_path.name}.")

    year, month_num, month_name = inferred
    month_folder = month_folder_name(month_num, month_name)
    output_dir = output_root / str(year) / month_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_outputs = [
        output_dir / f"{year}_{month_name}_{spec.name}.csv" for spec in TABLE_SPECS
    ]
    if all(path.exists() for path in expected_outputs):
        log_info(f"Skipping {pdf_path.name}; outputs already exist.")
        return
    log_info(f"Processing {pdf_path.name} -> {output_dir}.")

    for spec in TABLE_SPECS:
        cleaned = extract_table(pdf_path, spec)
        output_file = output_dir / f"{year}_{month_name}_{spec.name}.csv"
        cleaned.to_csv(output_file, index=False)
        log_success(f"Saved {output_file.name}.")


def main() -> None:
    pdfs = list(find_pdfs(RAW_DIR))
    if not pdfs:
        log_warning(f"No PDFs found under {RAW_DIR}.")
        raise SystemExit(1)

    log_info(f"Found {len(pdfs)} PDF(s) under {RAW_DIR}.")

    for pdf_path in pdfs:
        process_pdf(pdf_path, EXTRACTED_DIR)

    log_success("Extraction complete.")


if __name__ == "__main__":
    main()

