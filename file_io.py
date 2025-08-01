import os
import re
import logging, pathlib, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd





# -----------------------------------------------------------------------------
# Logging – everything goes to log.txt, nothing on the console
# -----------------------------------------------------------------------------
logger = logging.getLogger("spectrum")

if not logger.handlers:
    logger.setLevel(logging.INFO)
    _fh = logging.FileHandler("log.txt", mode="w", encoding="utf-8")
    _fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _fh.setFormatter(_fmt)
    logger.addHandler(_fh)
    logger.propagate = False  # keep root logger silent

# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------
#_file_format: str = ".hist"  # default – updated automatically per‑file

# Public helpers ---------------------------------------------------------------
global file_format


def set_file_format(fmt: str) -> None:
    """Set the global spectrum file‑format (.hist / .csv) if it changes."""
    file_format = os.path.splitext(spectrum_file_path)[-1].lower()
    file_format = fmt.lower()
    logger.debug("File format set to %s", file_format)


def get_file_format() -> str:
    return file_format


def auto_set_file_format_from_path(filepath: str) -> None:
    """Detect extension from *filepath* and call :pyfunc:`set_file_format`."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in (".csv", ".hist"):
        set_file_format(ext)
        logger.debug("Detected file extension: %s for %s", ext, filepath)
    else:
        raise ValueError(f"Unknown file format from extension: {ext}")


# -----------------------------------------------------------------------------
# Low‑level readers (.hist and .csv)
# -----------------------------------------------------------------------------

def read_hist_file(filepath: str, file_format = None) -> Tuple[np.ndarray, np.ndarray]:
    """Open *filepath* (either .hist or .csv) and return (counts, edges)."""
    if file_format is None:
        file_format = os.path.splitext(filepath)[-1].lower()
    if file_format == ".hist":
        return _read_hist_format(filepath)
    if file_format == ".csv":
        return _read_csv_format(filepath)
    raise ValueError(f"Unsupported format: {file_format}")


# -- helpers to parse .hist ----------------------------------------------------
_HDR_RE = re.compile(r"^\s*(Binwidth|EminLowEdge|EmaxLowEdge)\b", re.I)
_NUM_RE = re.compile(
    r"""
    ^\s*                  # leading spaces
    [+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?   # first number
    \s+                   # whitespace
    [+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?   # second number
    \s*$                  # end
""",
    re.X,
)


def _read_hist_format(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse Amptek‑style .hist text files."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
            lines = fh.readlines()

        # ---- meta (first three lines) ---------------------------------------
        meta: Dict[str, Any] = {}
        for ln in lines[:3]:
            parts = ln.strip().split()
            if len(parts) == 2 and _HDR_RE.match(ln):
                meta[parts[0].lower()] = float(parts[1])

        # ---- numeric spectrum lines ----------------------------------------
        numeric = [ln for ln in lines if _NUM_RE.match(ln)]
        if not numeric:
            raise ValueError("No numeric lines found in .hist file")

        data = np.loadtxt(numeric)  # shape (N, 2)
        bin_centers = data[:, 0]
        counts = data[:, 1]

        # derive bin width safely
        bin_width = meta.get(
            "binwidth",
            (bin_centers[1] - bin_centers[0]) if len(bin_centers) > 1 else 1.0,
        )
        edges = np.append(bin_centers - bin_width / 2, bin_centers[-1] + bin_width / 2)

        return counts, edges

    except Exception as exc:
        logger.error("Failed to read %s: %s", filepath, exc)
        raise


# -- CSV single‑spectrum -------------------------------------------------------

def _read_csv_format(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read simple two‑column CSV (Energy, Counts). Header is optional."""
    try:
        logger.debug("Reading CSV spectrum file: %s", filepath)
        data = np.loadtxt(filepath, delimiter=",", comments="#")
        edges = data[:, 0]
        counts = data[:, 1]
        # extend last edge for hist‑style plotting
        edges_ext = np.append(edges, edges[-1] + (edges[-1] - edges[-2]))
        return counts, edges_ext
    except Exception as exc:
        logger.error("Error reading CSV file %s: %s", filepath, exc)
        raise


# -----------------------------------------------------------------------------
# High‑level: load many spectra from a CSV manifest
# -----------------------------------------------------------------------------
HistTuple = Tuple[np.ndarray, np.ndarray]
_REQUIRED_COLS = {"x", "y", "rot", "spectrum_path"}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower‑case + map legacy column names to the required ones."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # legacy variants
    if "rot_idx" in df.columns and "rot_idx" not in df.columns:
        df.rename(columns={"rot_idx": "rot"}, inplace=True)
    if "spectrum_file_path" in df.columns and "spectrum_file_path" not in df.columns:
        df.rename(columns={"spectrum_file_path": "spectrum_file_path"}, inplace=True)
    return df


def read_histograms_from_files(csv_path: str, file_format = None, max_workers: int | None = None) -> Dict[Tuple[int, int, int], HistTuple]:
    """Load every spectrum referenced in *csv_path* concurrently.

    The CSV must contain the columns (x, y, rot|rot_idx, spectrum_path).
    A dict keyed by (x, y, rot) with values (counts, edges) is returned.
    """
    logger.info("Processing CSV file: %s", csv_path)
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    #df = _normalise_columns(df)
    required_cols = {"x", "y", "rot_idx", "spectrum_file_path"}

    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"CSV file is missing required columns: {missing}")

    results: Dict[Tuple[int, int, int], HistTuple] = {}
    max_workers = max_workers or (os.cpu_count() or 4)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {}
        for i, row in df.iterrows():
            key = (int(row["x"]), int(row["y"]), int(row["rot_idx"]))
            path = str(row["spectrum_file_path"]).strip()
            future_to_key[executor.submit(_safe_read_hist, path, get_file_format())] = key

        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                results[key] = future.result()
            except Exception as exc:
                logger.error("Failed to process %s: %s", key, exc)

    return results


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _safe_read_hist(path: str, file_format: str) -> HistTuple:
    """Wrapper around :pyfunc:`read_hist_file` with path validation."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    file_format = file_format or get_file_format()
    return read_hist_file(path, file_format)

