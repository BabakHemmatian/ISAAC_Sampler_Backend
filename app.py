# package and function imports
from __future__ import annotations

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr

import os
import sys
import uuid
import io
import csv
import time
import threading
import asyncio
import gzip
import shutil
import traceback
import gc
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Callable, Any, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import uvicorn
import numpy as np
import aiosmtplib
from email.message import EmailMessage
from dotenv import load_dotenv
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.csv as pacsv
import re
from calendar import monthrange

# load key-value pairs from .env file
load_dotenv()

# path variables
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR
METADATA_PATH = PROJECT_ROOT / "local_metadata.csv"
BULK_DIR = PROJECT_ROOT / "bulk"
DATASET_ROOT = BULK_DIR / "data"
LOCAL_METADATA_PATH = APP_DIR / "local_metadata.csv"
TEMP_DIR = BULK_DIR / "temp_files"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# CORS
ALLOWED_ORIGINS_ENV = os.getenv(
    "ALLOWED_ORIGINS",
    "https://isaac.psychology.illinois.edu,http://localhost:3000,http://localhost:3001",
)
ALLOWED_ORIGINS = [origin.strip() for origin in ALLOWED_ORIGINS_ENV.split(",") if origin.strip()]

# Email configuration
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
ISSUE_RECEIVER_EMAIL = os.getenv("ISSUE_RECEIVER_EMAIL", SMTP_USER)

# Performance / sizing
DEFAULT_MAX_WORKERS = int(os.getenv("MAX_WORKERS", str(max(2, os.cpu_count() or 4))))
ETA_EMA_ALPHA = float(os.getenv("ETA_EMA_ALPHA", "0.25"))
GZIP_COMPRESSLEVEL = int(os.getenv("GZIP_COMPRESSLEVEL", "1"))  # for parquet -> csv.gz and merged chunks
FULL_RANGE_CSV_FIRST = os.getenv("FULL_RANGE_CSV_FIRST", "1").lower() in ("1", "true", "yes")
TASK_CLEANUP_MAX_AGE_SECONDS = int(os.getenv("TASK_CLEANUP_MAX_AGE_SECONDS", "86400"))
TEMP_FILE_MAX_AGE_SECONDS = int(os.getenv("TEMP_FILE_MAX_AGE_SECONDS", str(48 * 3600)))
PROGRESS_UPDATE_INTERVAL_SECONDS = float(os.getenv("PROGRESS_UPDATE_INTERVAL_SECONDS", "0.5"))
CSV_PROGRESS_ROWS_PER_UNIT = int(os.getenv("CSV_PROGRESS_ROWS_PER_UNIT", "250000"))
ETA_MIN_SAMPLES = int(os.getenv("ETA_MIN_SAMPLES", "3"))
SAMPLING_PROGRESS_MIN_INTERVAL_SECONDS = float(os.getenv("SAMPLING_PROGRESS_MIN_INTERVAL_SECONDS", "1.0"))
# Relaxed from 0.02 -> 0.05: parquet random row-group access is significantly cheaper than
# scanning a whole gzipped CSV, so we should prefer it for a wider range of sample ratios.
PARQUET_SAMPLE_RATIO_THRESHOLD = float(os.getenv("PARQUET_SAMPLE_RATIO_THRESHOLD", "0.05"))
PARQUET_SAMPLE_MAX_ROWS = int(os.getenv("PARQUET_SAMPLE_MAX_ROWS", "200000"))
SMALL_FILE_ROWS_THRESHOLD = int(os.getenv("SMALL_FILE_ROWS_THRESHOLD", "300000"))
# Increased from 4MB -> 16MB. With ZIP_STORED-equivalent (tar) and no DEFLATE pass, throughput
# is closer to disk speed; bigger blocks mean fewer Python <-> C transitions.
STREAM_COPY_CHUNK_SIZE = int(os.getenv("STREAM_COPY_CHUNK_SIZE", str(16 * 1024 * 1024)))
PYARROW_CSV_BLOCK_SIZE = int(os.getenv("PYARROW_CSV_BLOCK_SIZE", str(8 * 1024 * 1024)))

### Utils

# regex for identifying monthly files in the local database
MONTH_FILE_RE = re.compile(
    r"^(?:RC|RS)_(\d{4})-(\d{2})(?:_part\d+_\d+)?\.(csv|csv\.gz|parquet|parq)$",
    re.IGNORECASE,
)

# Canonical column order for output CSVs. Used by parquet -> csv conversion, sampling readers,
# and the merge step. Keeping this as a flat list (no pandas dtype dict) since we now drive
# all CSV I/O through pyarrow.
READ_COLUMNS: List[str] = [
    "id", "parent id", "text", "author", "time",
    "subreddit", "score", "matched patterns", "source_row",
]

# create task containers
task_progress: Dict[str, Dict[str, Any]] = {}
task_stage: Dict[str, str] = {}
task_results: Dict[str, str] = {}
task_result_paths: Dict[str, str] = {}
task_meta: Dict[str, Dict[str, Any]] = {}
_progress_lock = threading.Lock()

_metadata_cache: Dict[str, Any] = {
    "mtime": None,
    "rows": None,
}

# time helper
def now() -> float:
    return time.time()

# cleans temp files from old requests
def cleanup_old_temp_files() -> None:
    current = time.time()
    for path in TEMP_DIR.glob("*"):
        try:
            if path.is_file():
                age = current - path.stat().st_mtime
                if age > TEMP_FILE_MAX_AGE_SECONDS:
                    path.unlink()
        except Exception as e:
            print(f"[WARN] Temp cleanup failed for {path}: {e}", file=sys.stderr)

# cleans up old tasks
def cleanup_old_tasks(max_age_seconds: int = TASK_CLEANUP_MAX_AGE_SECONDS) -> None:
    current = now()
    tasks_to_remove = []
    with _progress_lock:
        for task_id, meta in task_meta.items():
            start_time = float(meta.get("start_time", current))
            if current - start_time > max_age_seconds:
                tasks_to_remove.append(task_id)

        for task_id in tasks_to_remove:
            result_path = task_result_paths.pop(task_id, None)
            if result_path and os.path.exists(result_path):
                try:
                    os.remove(result_path)
                except Exception as e:
                    print(f"[WARN] Failed to remove old result {result_path}: {e}", file=sys.stderr)
            task_progress.pop(task_id, None)
            task_stage.pop(task_id, None)
            task_results.pop(task_id, None)
            task_meta.pop(task_id, None)

    if tasks_to_remove:
        print(f"[INFO] Cleaned up {len(tasks_to_remove)} old task(s)", file=sys.stderr)

# Time of last metadata modification
def _metadata_mtime() -> float:
    try:
        return LOCAL_METADATA_PATH.stat().st_mtime
    except FileNotFoundError:
        raise RuntimeError(f"Missing local metadata file: {LOCAL_METADATA_PATH}")

# load metadata from local storage
def _load_local_metadata_rows() -> List[Dict[str, Any]]:
    mtime = _metadata_mtime()
    if _metadata_cache["rows"] is not None and _metadata_cache["mtime"] == mtime:
        return _metadata_cache["rows"]

    rows: List[Dict[str, Any]] = []
    with open(LOCAL_METADATA_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            date_val = (raw.get("date") or "").strip()
            month_key = date_val[:7] if len(date_val) >= 7 else ""
            row = {
                "id": int(raw["id"]) if raw.get("id") else None,
                "social_group": (raw.get("social_group") or "").strip(),
                "date": date_val,
                "month_key": month_key,
                "file_path": (raw.get("file_path") or "").strip(),
                "num_rows": int(raw["num_rows"]) if raw.get("num_rows") not in (None, "",) else 0,
                "file_format": (raw.get("file_format") or "").strip().lower(),
                "size_bytes": int(raw["size_bytes"]) if raw.get("size_bytes") not in (None, "",) else 0,
                "row_groups": int(raw["row_groups"]) if raw.get("row_groups") not in (None, "",) else 0,
                "ts_min": (raw.get("ts_min") or "").strip() or None,
                "ts_max": (raw.get("ts_max") or "").strip() or None,
                "year": int(raw["year"]) if raw.get("year") not in (None, "",) else None,
                "month": int(raw["month"]) if raw.get("month") not in (None, "",) else None,
                "csv_file_path": (raw.get("csv_file_path") or "").strip(),
                "csv_available": str(raw.get("csv_available", "")).strip().lower() in ("1", "true", "yes"),
            }
            rows.append(row)

    _metadata_cache["mtime"] = mtime
    _metadata_cache["rows"] = rows
    return rows

# find local dataset
def _resolve_dataset_path(rel_or_abs: str) -> str:
    path = Path(rel_or_abs)
    if path.is_absolute():
        return str(path)
    return str((DATASET_ROOT / path).resolve())

# identify the months in the request range
def month_in_range(year: int, month: int, start_month: str, end_month: str) -> bool:
    start_key = int(start_month[:4]) * 100 + int(start_month[5:7])
    end_key = int(end_month[:4]) * 100 + int(end_month[5:7])
    this_key = year * 100 + month
    return start_key <= this_key <= end_key

### App creation

# create the FastAPI app object
app = FastAPI()

# Apply CORS to defined origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# create a class for issue reports
class IssueReport(BaseModel):
    email: EmailStr
    description: str

### ETA

# ETA report
def _fmt_eta(seconds: Optional[float]) -> Optional[str]:
    if seconds is None:
        return None
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"

# ETA progress evaluation
def _set_progress(
    task_id: str,
    *,
    stage: Optional[str] = None,
    total_units: Optional[int] = None,
    completed_units: Optional[int] = None,
    eta_seconds: Optional[float] = None,
) -> None:
    with _progress_lock:
        ts = now()
        meta = task_meta.setdefault(
            task_id,
            {
                "start_time": ts,
                "phase_start_ts": ts,
                "phase_start_units": 0,
                "total_units": 0,
                "completed_units": 0,
            },
        )

        if total_units is not None:
            meta["total_units"] = max(0, int(total_units))
        if completed_units is not None:
            meta["completed_units"] = max(0, int(completed_units))

        total = max(0, int(meta.get("total_units", 0) or 0))
        done = max(0, int(meta.get("completed_units", 0) or 0))
        percent = (done / total * 100.0) if total > 0 else 0.0

        prog = task_progress.setdefault(task_id, {})
        prog["percent"] = round(min(100.0, max(0.0, percent)), 2)
        prog["eta_seconds"] = None if eta_seconds is None else max(0.0, float(eta_seconds))
        prog["eta_human"] = _fmt_eta(prog["eta_seconds"])

        if stage is not None:
            task_stage[task_id] = stage

# ETA update
#
# Rewritten to use a phase-scoped average rate instead of an EMA over instantaneous
# inter-completion deltas. The old EMA approach failed badly here because:
#   (a) work per file is bimodal (.csv.gz no-op vs parquet conversion),
#   (b) parallel workers complete in bursts -- N completions clustered, then a
#       long gap -- which makes inst_rate swing between near-zero and absurdly high,
#   (c) the resulting EMA never converges within a single run, so the displayed
#       ETA tends to grow during the slow middle of the job.
#
# The new approach: rate = progressed_in_phase / elapsed_in_phase. This is the
# true wall-clock average of the current phase and is stable under bursty parallel
# completions. Phase boundaries call _reset_eta_smoothing(), which resets the
# phase baseline so the next phase computes its own rate from scratch.
def _update_eta_from_units(task_id: str) -> None:
    with _progress_lock:
        meta = task_meta.get(task_id)
        if not meta:
            return

        total = max(0, int(meta.get("total_units", 0) or 0))
        done = max(0, int(meta.get("completed_units", 0) or 0))
        if total <= 0 or done <= 0:
            task_progress.setdefault(task_id, {})["eta_seconds"] = None
            task_progress.setdefault(task_id, {})["eta_human"] = None
            return

        phase_start_ts = float(meta.get("phase_start_ts") or meta.get("start_time") or now())
        phase_start_units = int(meta.get("phase_start_units") or 0)
        elapsed = max(1e-6, now() - phase_start_ts)
        progressed = max(0, done - phase_start_units)
        phase_total = max(1, total - phase_start_units)
        phase_fraction = progressed / phase_total

        # Hide ETA until we have meaningful signal. With bursty parallel completions
        # the first few samples don't represent steady-state throughput.
        if phase_fraction < 0.005 or elapsed < 1.5:
            task_progress.setdefault(task_id, {})["eta_seconds"] = None
            task_progress.setdefault(task_id, {})["eta_human"] = None
            return

        rate = progressed / elapsed
        if rate <= 0:
            task_progress.setdefault(task_id, {})["eta_seconds"] = None
            task_progress.setdefault(task_id, {})["eta_human"] = None
            return

        remaining = max(0, total - done)
        eta = remaining / rate
        task_progress.setdefault(task_id, {})["eta_seconds"] = eta
        task_progress.setdefault(task_id, {})["eta_human"] = _fmt_eta(eta)

# Reset the phase baseline used by _update_eta_from_units. Call this before
# transitioning between phases of a multi-stage task (e.g., prepare -> bundle)
# so the next phase's ETA reflects its own throughput rather than the previous
# phase's. Briefly hides the ETA until the new phase has produced enough signal.
def _reset_eta_smoothing(task_id: str) -> None:
    with _progress_lock:
        meta = task_meta.get(task_id)
        if meta is None:
            return
        meta["phase_start_ts"] = now()
        meta["phase_start_units"] = int(meta.get("completed_units", 0) or 0)
        prog = task_progress.setdefault(task_id, {})
        prog["eta_seconds"] = None
        prog["eta_human"] = None

# initial progress metadata
def _init_progress_meta(task_id: str, total_units: int, stage: str) -> None:
    ts = now()
    with _progress_lock:
        task_meta[task_id] = {
            "start_time": ts,
            "phase_start_ts": ts,
            "phase_start_units": 0,
            "total_units": max(0, int(total_units)),
            "completed_units": 0,
        }
        task_progress[task_id] = {"percent": 0.0, "eta_seconds": None, "eta_human": None}
        task_stage[task_id] = stage

# incremental progress evaluation
def _increment_progress(task_id: str, units: int, stage: Optional[str] = None) -> None:
    if units <= 0:
        return

    with _progress_lock:
        ts = now()
        meta = task_meta.setdefault(
            task_id,
            {
                "start_time": ts,
                "phase_start_ts": ts,
                "phase_start_units": 0,
                "total_units": 0,
                "completed_units": 0,
            },
        )
        total_units = int(meta.get("total_units", 0) or 0)
        new_done = int(meta.get("completed_units", 0) or 0) + int(units)
        if total_units > 0:
            new_done = min(new_done, total_units)
        meta["completed_units"] = new_done

    _set_progress(task_id, stage=stage)
    _update_eta_from_units(task_id)

# fetch metadata from local storage
async def fetch_metadata(
    social_group: str,
    start_month: str,
    end_month: str,
    *,
    prefer_csv_for_full_range: bool = False,
):
    rows = _load_local_metadata_rows()
    selected: List[Dict[str, Any]] = []

    for row in rows:
        if row["social_group"] != social_group:
            continue
        month_key = row["month_key"]
        if not month_key:
            continue
        if month_key < start_month or month_key > end_month:
            continue

        file_path = row["file_path"]
        csv_file_path = row.get("csv_file_path") or ""
        use_csv = prefer_csv_for_full_range and FULL_RANGE_CSV_FIRST and row.get("csv_available") and csv_file_path

        chosen_path = csv_file_path if use_csv else file_path
        chosen_format = "csv" if use_csv else (row.get("file_format") or "")
        selected.append(
            {
                "file_path": _resolve_dataset_path(chosen_path),
                "num_rows": int(row.get("num_rows") or 0),
                "file_format": chosen_format,
                "csv_available": bool(row.get("csv_available")),
                "csv_file_path": _resolve_dataset_path(csv_file_path) if csv_file_path else "",
                "month_key": month_key,
                "ts_min": row.get("ts_min"),
                "ts_max": row.get("ts_max"),
                "row_groups": int(row.get("row_groups") or 0),
            }
        )

    selected.sort(key=lambda x: x["month_key"])
    for item in selected:
        yield item

# fetch file list
async def fetch_metadata_list(
    social_group: str,
    start_month: str,
    end_month: str,
    *,
    prefer_csv_for_full_range: bool = False,
):
    items = []
    async for item in fetch_metadata(
        social_group,
        start_month,
        end_month,
        prefer_csv_for_full_range=prefer_csv_for_full_range,
    ):
        items.append(item)
    return items

# estimate number of rows in a local data file
def estimate_num_rows_local(path: Path) -> int:
    lower = path.name.lower()
    try:
        if lower.endswith((".parquet", ".parq")):
            pf = pq.ParquetFile(path)
            return pf.metadata.num_rows if pf.metadata else 0
        opener = gzip.open if lower.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8", newline="") as f:
            rows = sum(1 for _ in f)
        return max(0, rows - 1)
    except Exception:
        return 0

# sort sample positions
def _build_sorted_sample_positions(
    total_rows: int,
    k: int,
    seed: Optional[Any] = None,
) -> np.ndarray:
    if total_rows <= 0 or k <= 0:
        return np.empty(0, dtype=np.int64)
    k = min(int(k), int(total_rows))
    if k == total_rows:
        return np.arange(total_rows, dtype=np.int64)
    # If a seed (int or numpy SeedSequence) is provided the RNG is fully
    # deterministic, which is what makes the random subset reproducible.
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total_rows, size=k, replace=False).astype(np.int64, copy=False))

### PyArrow CSV helpers
# Centralized so reader/writer behavior is consistent across the sampling and merge paths.

def _csv_read_options(use_threads: bool = True) -> pacsv.ReadOptions:
    return pacsv.ReadOptions(use_threads=use_threads, block_size=PYARROW_CSV_BLOCK_SIZE)

def _csv_string_convert_options() -> pacsv.ConvertOptions:
    """All target columns read as strings to avoid type-inference variance across files
    and to remove the cost of type conversion. Output is CSV anyway -- nothing downstream
    needs typed columns."""
    return pacsv.ConvertOptions(
        include_columns=READ_COLUMNS,
        include_missing_columns=True,
        column_types={c: pa.string() for c in READ_COLUMNS},
        strings_can_be_null=True,
    )

def _open_csv_binary(path_str: str):
    """Open a .csv or .csv.gz path for binary reading. PyArrow CSV consumes bytes."""
    if path_str.lower().endswith(".csv.gz"):
        return gzip.open(path_str, "rb")
    return open(path_str, "rb")

### Sampling: CSV path

# Read a (possibly gzipped) CSV with PyArrow streaming, picking up rows at the given
# sorted positions, and write them to a gzipped temp CSV. Replaces the previous
# pd.read_csv(chunksize=...) + df.iloc[]/df.to_csv() loop. PyArrow releases the GIL
# during parsing, so even single-file sampling benefits from internal threading, and
# the all-strings convert options skip the pandas type-conversion overhead.
def sample_csv_positions_to_temp_csv(
    path_str: str,
    positions: np.ndarray,
    tmp_path: str,
    *,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> int:
    if positions.size == 0:
        return 0

    pos_idx = 0
    row_base = 0
    rows_written = 0
    writer: Optional[pacsv.CSVWriter] = None

    src = _open_csv_binary(path_str)
    try:
        reader = pacsv.open_csv(
            src,
            _csv_read_options(use_threads=True),
            pacsv.ParseOptions(),
            _csv_string_convert_options(),
        )

        with gzip.open(tmp_path, "wb", compresslevel=GZIP_COMPRESSLEVEL) as out:
            try:
                for batch in reader:
                    chunk_len = batch.num_rows
                    if chunk_len <= 0:
                        continue

                    next_base = row_base + chunk_len
                    start = pos_idx
                    while pos_idx < positions.size and positions[pos_idx] < next_base:
                        pos_idx += 1

                    if pos_idx > start:
                        local_positions = positions[start:pos_idx] - row_base
                        sampled = batch.take(pa.array(local_positions))
                        if writer is None:
                            writer = pacsv.CSVWriter(out, sampled.schema)
                        writer.write_batch(sampled)
                        rows_written += sampled.num_rows

                    row_base = next_base
                    if progress_cb is not None:
                        progress_cb(chunk_len)

                    if pos_idx >= positions.size:
                        break
            finally:
                if writer is not None:
                    writer.close()
    finally:
        src.close()

    return rows_written

### Sampling: Parquet path

# extract row group sizes for parquet sampling
def _parquet_row_group_sizes(pf: pq.ParquetFile) -> List[int]:
    md = pf.metadata
    if md is None:
        return []
    return [md.row_group(i).num_rows for i in range(md.num_row_groups)]

# Read row groups containing the sampled positions and write the sampled rows to a
# gzipped temp CSV via pyarrow.csv.CSVWriter. Avoids the pandas detour entirely.
def sample_parquet_positions_to_temp_csv(
    path_str: str,
    positions: np.ndarray,
    tmp_path: str,
    *,
    columns: Optional[List[str]] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> int:
    if positions.size == 0:
        return 0

    pf = pq.ParquetFile(path_str)
    row_group_sizes = _parquet_row_group_sizes(pf)

    if not row_group_sizes:
        table = pf.read(columns=columns)
        if table.num_rows <= 0:
            return 0
        sampled = table.take(pa.array(positions))
        with gzip.open(tmp_path, "wb", compresslevel=GZIP_COMPRESSLEVEL) as out:
            pacsv.write_csv(sampled, out)
        return sampled.num_rows

    cumulative = np.cumsum(np.asarray(row_group_sizes, dtype=np.int64))
    group_ids = np.searchsorted(cumulative, positions, side="right")

    rows_written = 0
    start_idx = 0
    writer: Optional[pacsv.CSVWriter] = None

    with gzip.open(tmp_path, "wb", compresslevel=GZIP_COMPRESSLEVEL) as out:
        try:
            while start_idx < positions.size:
                rg_idx = int(group_ids[start_idx])
                end_idx = start_idx
                while end_idx < positions.size and int(group_ids[end_idx]) == rg_idx:
                    end_idx += 1

                row_group_start = 0 if rg_idx == 0 else int(cumulative[rg_idx - 1])
                local_positions = (positions[start_idx:end_idx] - row_group_start).astype(np.int64, copy=False)

                table = pf.read_row_group(rg_idx, columns=columns)
                sampled = table.take(pa.array(local_positions))

                if writer is None:
                    writer = pacsv.CSVWriter(out, sampled.schema)
                writer.write_table(sampled)
                rows_written += sampled.num_rows

                if progress_cb is not None:
                    progress_cb(int(row_group_sizes[rg_idx]))

                start_idx = end_idx
        finally:
            if writer is not None:
                writer.close()

    return rows_written

# identify the sampling path
def choose_sampling_path(file_info: Dict[str, Any], quota: int) -> str:
    primary_path = file_info["file_path"]
    csv_path = file_info.get("csv_file_path") or ""
    num_rows = max(0, int(file_info.get("num_rows") or 0))
    row_groups = max(0, int(file_info.get("row_groups") or 0))
    has_csv = bool(file_info.get("csv_available") and csv_path and os.path.exists(csv_path))
    primary_is_parquet = primary_path.lower().endswith((".parquet", ".parq"))

    if not has_csv:
        return primary_path

    if not primary_is_parquet:
        return csv_path

    if num_rows <= 0:
        return csv_path

    ratio = quota / max(1, num_rows)
    prefer_parquet = (
        row_groups > 0
        and quota <= PARQUET_SAMPLE_MAX_ROWS
        and ratio <= PARQUET_SAMPLE_RATIO_THRESHOLD
        and num_rows > SMALL_FILE_ROWS_THRESHOLD
    )
    return primary_path if prefer_parquet else csv_path

# sample any file format to a temp gzipped CSV
def sample_any_to_temp_csv(
    file_info: Dict[str, Any],
    k: int,
    *,
    progress_cb: Optional[Callable[[int], None]] = None,
    seed: Optional[Any] = None,
) -> Tuple[Optional[str], int, str]:
    if k <= 0:
        return None, 0, ""

    num_rows = max(0, int(file_info.get("num_rows") or 0))
    if num_rows <= 0:
        return None, 0, ""

    source_path = choose_sampling_path(file_info, k)
    positions = _build_sorted_sample_positions(num_rows, min(k, num_rows), seed=seed)
    if positions.size == 0:
        return None, 0, source_path

    # Per-file sampled output goes to a .csv.gz temp so we save disk space and
    # the merge step can stream gzipped bytes.
    tmp_name = f"sample_{uuid.uuid4().hex}.csv.gz"
    tmp_path = str(TEMP_DIR / tmp_name)

    try:
        if source_path.lower().endswith((".parquet", ".parq")):
            row_count = sample_parquet_positions_to_temp_csv(
                source_path,
                positions,
                tmp_path,
                columns=READ_COLUMNS,
                progress_cb=progress_cb,
            )
        else:
            row_count = sample_csv_positions_to_temp_csv(
                source_path,
                positions,
                tmp_path,
                progress_cb=progress_cb,
            )

        if row_count <= 0:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return None, 0, source_path

        return tmp_path, row_count, source_path
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

# identify number of sampled rows per data file
def compute_per_file_quotas(files: List[Dict[str, Any]], num_docs: int) -> List[int]:
    if not files:
        return []

    capacities = [max(0, int(f.get("num_rows") or 0)) for f in files]
    quotas = [0] * len(files)
    active = [i for i, cap in enumerate(capacities) if cap > 0]
    remaining = max(0, int(num_docs))

    while active and remaining > 0:
        base = max(1, remaining // len(active))
        next_active = []
        for idx in active:
            cap_left = capacities[idx] - quotas[idx]
            if cap_left <= 0:
                continue
            take = min(base, cap_left, remaining)
            quotas[idx] += take
            remaining -= take
            if quotas[idx] < capacities[idx]:
                next_active.append(idx)
            if remaining <= 0:
                break
        active = next_active

    if remaining > 0:
        for idx, cap in sorted(enumerate(capacities), key=lambda x: x[1] - quotas[x[0]], reverse=True):
            cap_left = cap - quotas[idx]
            if cap_left <= 0:
                continue
            take = min(cap_left, remaining)
            quotas[idx] += take
            remaining -= take
            if remaining <= 0:
                break

    return quotas

# compute sampling units for ETA evaluation
def compute_sampling_total_units(files: List[Dict[str, Any]], quotas: List[int]) -> int:
    total = 0
    for file_info, quota in zip(files, quotas):
        if quota <= 0:
            continue
        num_rows = max(1, int(file_info.get("num_rows") or 0))
        source_path = choose_sampling_path(file_info, quota)
        if source_path.lower().endswith((".parquet", ".parq")):
            total += min(num_rows, max(quota * 64, quota, 1))
        else:
            total += num_rows
    return max(total, 1)

### Sampling: merge step

def merge_sampled_to_single_csv(
    sampled_temp_files: List[Tuple[str, int, str]],
    output_path: str,
    *,
    rows_done_cb: Optional[Callable[[int], None]] = None,
) -> int:
    """Merge all per-file sampled temps into ONE plain (uncompressed) .csv with a
    single header row, and return the total rows written.

    This keeps a sample's on-disk shape identical to the full-month files served
    from Globus (a single CSV, same columns) — no tar, no gzip, no chunking — so
    the output a user gets opens directly in Excel and matches the bulk files."""
    canonical_schema = pa.schema([pa.field(c, pa.string()) for c in READ_COLUMNS])
    convert_options = _csv_string_convert_options()
    read_options = _csv_read_options(use_threads=False)
    parse_options = pacsv.ParseOptions()

    total_rows = 0
    out_file = open(output_path, "wb")
    writer = pacsv.CSVWriter(out_file, canonical_schema)  # writes the header row
    try:
        for tmp_csv_path, _row_count, _used_path in sampled_temp_files:
            with gzip.open(tmp_csv_path, "rb") as src:
                reader = pacsv.open_csv(src, read_options, parse_options, convert_options)
                for batch in reader:
                    if batch.num_rows == 0:
                        continue
                    writer.write_batch(batch)
                    total_rows += batch.num_rows
                    if rows_done_cb is not None:
                        rows_done_cb(batch.num_rows)
    finally:
        writer.close()
        out_file.close()
    return total_rows

### Sampling task

def background_sampling(
    task_id: str,
    social_group: str,
    start_date: str,
    end_date: str,
    num_docs: Optional[int] = None,
    random_seed: Optional[int] = None,
):
    try:
        overall_timeout = int(os.getenv("DB_OVERALL_TIMEOUT", "300"))
        _set_progress(task_id, stage="Fetching metadata")

        try:
            files = run_coro_in_new_loop(
                fetch_metadata_list(
                    social_group,
                    start_date,
                    end_date,
                    prefer_csv_for_full_range=False,
                ),
                overall_timeout=overall_timeout,
            )
        except Exception as e:
            _set_progress(task_id, stage=f"Error: fetch_metadata failed: {repr(e)}", eta_seconds=0)
            return

        if not files:
            _set_progress(task_id, stage="No files found", eta_seconds=0)
            return

        # ---- Sampling path: per-file random sampling -> one merged plain CSV ----
        total_available = sum((f.get("num_rows") or 0) for f in files)
        num_docs = max(0, min(int(num_docs), total_available))
        quotas = compute_per_file_quotas(files, num_docs)

        # When a seed is supplied, derive a deterministic per-file child seed via
        # numpy's SeedSequence. This guarantees reproducibility regardless of the
        # order in which ThreadPoolExecutor workers finish each file, while still
        # giving each file a statistically independent stream.
        if random_seed is not None:
            seed_sequence = np.random.SeedSequence(int(random_seed))
            file_seeds: List[Optional[Any]] = list(seed_sequence.spawn(len(files)))
        else:
            file_seeds = [None] * len(files)

        # All three phases use rows as the unit so phase rates stay comparable and
        # ETA resets between phases produce sensible values.
        # - Sampling phase: rows scanned across source files (compute_sampling_total_units).
        # - Merge phase: rows written to chunk files (~= num_docs).
        # - Bundle phase: tar.add of each chunk; weight per chunk reflects fast byte copy.
        total_sampling_units = compute_sampling_total_units(files, quotas)
        total_merge_units = max(1, num_docs)
        total_units = total_sampling_units + total_merge_units
        _init_progress_meta(task_id, total_units=total_units, stage="Sampling documents")

        short_id = task_id.replace("-", "")[:8]
        # Self-describing names: social group, month range, sample size, and (when
        # supplied) the seed — so a downloaded sample's parameters are obvious from
        # its filename and reproducible subsets are unambiguous. The sample is a
        # single plain CSV, matching the full-month files served from Globus.
        seed_suffix = f"_seed{int(random_seed)}" if random_seed is not None else ""
        base_name = f"ISAAC_{social_group}_{start_date}_to_{end_date}_n{num_docs}{seed_suffix}"
        csv_filename = f"{base_name}_{short_id}.csv"
        display_filename = f"{base_name}.csv"
        csv_path = str(TEMP_DIR / csv_filename)
        with _progress_lock:
            if task_id in task_meta:
                task_meta[task_id]["display_filename"] = display_filename

        total_sampled_rows = 0
        sampled_temp_files: List[Tuple[str, int, str]] = []

        progress_lock = threading.Lock()
        progress_state: Dict[str, Dict[str, float]] = {}

        def make_progress_cb(file_key: str) -> Callable[[int], None]:
            def _cb(rows_delta: int) -> None:
                if rows_delta <= 0:
                    return
                ts = now()
                units_to_emit = 0
                with progress_lock:
                    st = progress_state.setdefault(file_key, {"pending_units": 0, "last_emit_ts": 0.0})
                    st["pending_units"] += int(rows_delta)
                    if st["pending_units"] >= CSV_PROGRESS_ROWS_PER_UNIT or (ts - st["last_emit_ts"]) >= SAMPLING_PROGRESS_MIN_INTERVAL_SECONDS:
                        units_to_emit = int(st["pending_units"])
                        st["pending_units"] = 0
                        st["last_emit_ts"] = ts
                if units_to_emit > 0:
                    _increment_progress(task_id, units_to_emit, stage="Sampling documents")
            return _cb

        def flush_file_progress(file_key: str) -> None:
            leftover = 0
            with progress_lock:
                st = progress_state.get(file_key)
                if st:
                    leftover = int(st.get("pending_units", 0) or 0)
                    st["pending_units"] = 0
            if leftover > 0:
                _increment_progress(task_id, leftover, stage="Sampling documents")

        def process_one(file_info: Dict[str, Any], k: int, file_seed: Optional[Any]) -> Tuple[Optional[str], int, str]:
            file_key = file_info["file_path"]
            try:
                return sample_any_to_temp_csv(
                    file_info,
                    k,
                    progress_cb=make_progress_cb(file_key),
                    seed=file_seed,
                )
            finally:
                flush_file_progress(file_key)

        with ThreadPoolExecutor(max_workers=DEFAULT_MAX_WORKERS) as ex:
            futures = [
                ex.submit(process_one, f, q, s)
                for f, q, s in zip(files, quotas, file_seeds)
                if q > 0
            ]
            for fut in as_completed(futures):
                tmp_csv_path, row_count, used_path = fut.result()
                if tmp_csv_path and row_count > 0:
                    sampled_temp_files.append((tmp_csv_path, row_count, used_path))
                    total_sampled_rows += row_count
                    _set_progress(task_id, stage=f"Sampled {total_sampled_rows} rows so far")

        if total_sampled_rows == 0:
            # No rows matched the selection — emit an empty CSV (header only) so the
            # result is still a valid, openable file in the same format as a normal
            # sample (this previously produced an empty tar).
            empty_table = pa.table({c: pa.array([], type=pa.string()) for c in READ_COLUMNS})
            with open(csv_path, "wb") as out:
                pacsv.write_csv(empty_table, out)
            task_result_paths[task_id] = csv_path
            task_results[task_id] = f"/download/{task_id}"
            _set_progress(task_id, completed_units=total_units, stage="No matching rows found", eta_seconds=0)
            cleanup_old_tasks()
            return

        # Merge per-file samples into chunked .csv.gz outputs.
        # Reset the ETA baseline so the merge phase doesn't inherit the sampling
        # phase's throughput (which may be much faster or slower depending on the
        # mix of CSV-scan vs parquet-row-group sampling).
        _reset_eta_smoothing(task_id)
        _set_progress(task_id, stage="Merging sampled rows")

        def merge_progress_cb(rows_done: int) -> None:
            _increment_progress(task_id, max(1, int(rows_done)), stage="Merging sampled rows")

        try:
            merge_sampled_to_single_csv(
                sampled_temp_files,
                csv_path,
                rows_done_cb=merge_progress_cb,
            )
        finally:
            # Always remove per-file sampling temps.
            for tmp_csv_path, _, _ in sampled_temp_files:
                try:
                    os.remove(tmp_csv_path)
                except OSError:
                    pass

        if not os.path.exists(csv_path):
            _set_progress(task_id, stage="Error: No output generated", eta_seconds=0)
            return

        task_result_paths[task_id] = csv_path
        task_results[task_id] = f"/download/{task_id}"
        _set_progress(task_id, completed_units=total_units, stage="Done", eta_seconds=0)
        cleanup_old_tasks()

    except Exception as e:
        print(f"[ERROR] background_sampling crashed: {repr(e)}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        _set_progress(task_id, stage=f"Error: {repr(e)}", eta_seconds=0)

# run a coroutine in a new loop
def run_coro_in_new_loop(coro, overall_timeout: int):
    result_box: Dict[str, Any] = {}
    error_box: Dict[str, Exception] = {}
    done_evt = threading.Event()

    def _runner():
        try:
            result_box["value"] = asyncio.run(asyncio.wait_for(coro, timeout=overall_timeout))
        except Exception as e:
            error_box["exc"] = e
        finally:
            done_evt.set()

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    if not done_evt.wait(overall_timeout + 5):
        raise TimeoutError("Timed out waiting for async operation to finish")

    if "exc" in error_box:
        raise error_box["exc"]
    return result_box.get("value")

# Return status
@app.get("/")
def home():
    return {"message": "FastAPI is running"}

# Strip the on-disk uniqueness suffix (8-hex short-id, optionally followed by a
# legacy 14-digit timestamp + UUID) to produce a user-friendly download filename.
def _user_facing_filename(disk_name: str) -> str:
    # Legacy: ISAAC_..._<full-uuid>_<14-digit-ts>.<ext>
    name = re.sub(r"_[0-9a-fA-F]{8}-[0-9a-fA-F-]+_\d{14}(?=\.[^.]+$)", "", disk_name)
    # New: ISAAC_..._<8-hex-short-id>.<ext>
    name = re.sub(r"_[0-9a-fA-F]{8}(?=\.[^.]+$)", "", name)
    return name

# download button
@app.get("/download/{task_id}")
async def download_result(task_id: str):
    path = task_result_paths.get(task_id)

    # Look up the result file for this task. New tasks produce .tar with a short-id
    # suffix; legacy tasks may have .zip with a full UUID + timestamp.
    if (not path or not os.path.exists(path)) and TEMP_DIR.exists():
        short_id = task_id.replace("-", "")[:8]
        candidates: List[Path] = []
        for ext in (".csv", ".tar", ".zip"):
            candidates.extend(TEMP_DIR.glob(f"*_{short_id}{ext}"))
            candidates.extend(TEMP_DIR.glob(f"*_{task_id}_*{ext}"))  # legacy
        if candidates:
            candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            path = str(candidates[0])
            task_result_paths[task_id] = path
            task_results[task_id] = f"/download/{task_id}"

    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found or expired")

    # Prefer the explicit display filename stored in task_meta when available;
    # otherwise derive a clean one from the disk path (handles legacy zips too).
    display_name = None
    meta = task_meta.get(task_id) or {}
    if isinstance(meta.get("display_filename"), str):
        display_name = meta["display_filename"]
    if not display_name:
        display_name = _user_facing_filename(os.path.basename(path))

    lower = path.lower()
    if lower.endswith(".csv"):
        media_type = "text/csv"
    elif lower.endswith(".tar"):
        media_type = "application/x-tar"
    else:
        media_type = "application/zip"
    return FileResponse(path, media_type=media_type, filename=display_name)

# report issue function
@app.post("/report_issue")
async def report_issue(data: IssueReport):
    if not SMTP_HOST or not SMTP_USER or not SMTP_PASSWORD or not ISSUE_RECEIVER_EMAIL:
        raise HTTPException(status_code=500, detail="SMTP configuration missing")

    message = EmailMessage()
    message["From"] = SMTP_USER
    message["To"] = ISSUE_RECEIVER_EMAIL
    message["Subject"] = f"Issue Reported by {data.email}"
    message.set_content(f"User Email: {data.email}\n\nIssue:\n{data.description}")

    try:
        await aiosmtplib.send(
            message,
            hostname=SMTP_HOST,
            port=SMTP_PORT,
            start_tls=True,
            username=SMTP_USER,
            password=SMTP_PASSWORD,
        )
        return {"message": "Issue reported successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send email: {str(e)}")

# sampling task
@app.post("/sample")
async def get_sampled_data(payload: dict, background_tasks: BackgroundTasks):
    social_group = payload.get("social_group")
    start_date = payload.get("start_date")
    end_date = payload.get("end_date")
    num_docs = payload.get("num_docs")
    random_seed = payload.get("random_seed")

    if not social_group or not start_date or not end_date:
        raise HTTPException(status_code=400, detail="Missing required fields")

    # Whole-file (no-sample) downloads are now served directly from the public
    # Globus collection via links generated in the website UI — the VM no longer
    # bundles entire months on demand (that egress is what we moved off Radiant).
    # Reject a missing/blank num_docs so this path cannot be invoked from the API.
    if num_docs is None or (isinstance(num_docs, str) and num_docs.strip() == ""):
        raise HTTPException(
            status_code=400,
            detail=("Whole-file downloads are served via direct Globus links, not this "
                    "endpoint. Provide num_docs to request a random sample, or use the "
                    "Direct Download page for full monthly files."),
        )

    try:
        start_parts = start_date.split("-")
        end_parts = end_date.split("-")
        if len(start_parts) < 2 or len(end_parts) < 2:
            raise HTTPException(status_code=400, detail="Invalid date format. Expected YYYY-MM")

        start_year = int(start_parts[0])
        end_year = int(end_parts[0])
        if start_year < 2007 or end_year > 2023:
            raise HTTPException(status_code=400, detail="Date range must be between 2007 and 2023")
    except (ValueError, IndexError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format. Expected YYYY-MM: {str(e)}")

    if start_date > end_date:
        raise HTTPException(status_code=400, detail="Start date must be before end date")

    if num_docs is not None:
        try:
            num_docs = int(num_docs)
            if num_docs < 0:
                raise HTTPException(status_code=400, detail="num_docs must be non-negative")
            if num_docs > 100_000_000:
                raise HTTPException(status_code=400, detail="num_docs exceeds maximum allowed (100,000,000)")
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="num_docs must be a valid integer")

    # The seed is only meaningful when a sample size is requested. Silently drop
    # it for the full-range path so the user-facing tar filename stays clean.
    if random_seed is not None and num_docs is not None:
        try:
            random_seed = int(random_seed)
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="random_seed must be a valid integer")
        if random_seed < 0:
            raise HTTPException(status_code=400, detail="random_seed must be non-negative")
        # numpy's SeedSequence accepts arbitrarily large ints, but cap to a
        # 64-bit range so we don't silently overflow downstream JSON or filenames.
        if random_seed > 2**63 - 1:
            raise HTTPException(status_code=400, detail="random_seed exceeds maximum allowed (2^63 - 1)")
    else:
        random_seed = None

    cleanup_old_tasks()
    cleanup_old_temp_files()

    task_id = str(uuid.uuid4())
    with _progress_lock:
        task_meta[task_id] = {"start_time": now(), "total_units": 0, "completed_units": 0}
        task_progress[task_id] = {"percent": 0.0, "eta_seconds": None, "eta_human": None}
        task_stage[task_id] = "Queued"

    background_tasks.add_task(
        background_sampling,
        task_id,
        social_group,
        start_date,
        end_date,
        num_docs,
        random_seed,
    )
    return {"task_id": task_id}

# progress marker
@app.get("/progress/{task_id}")
async def get_progress(task_id: str):
    prog = task_progress.get(task_id, {})
    return {
        "stage": task_stage.get(task_id, "Initializing..."),
        "percent": prog.get("percent"),
        "eta_seconds": prog.get("eta_seconds"),
        "eta_human": prog.get("eta_human"),
        "download_link": task_results.get(task_id),
    }

# dataset stats evaluation
@app.get("/dataset-stats")
async def get_dataset_stats():
    try:
        rows = _load_local_metadata_rows()
        social_groups = sorted({row["social_group"] for row in rows if row["social_group"]})

        all_results: Dict[str, Any] = {}
        for group in social_groups:
            group_rows = [r for r in rows if r["social_group"] == group]
            year_buckets: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
            for row in group_rows:
                if row["year"] is not None:
                    year_buckets[int(row["year"])].append(row)

            year_data = []
            total_files = 0
            total_rows = 0
            all_file_sizes = []

            for year in sorted(year_buckets.keys()):
                bucket = year_buckets[year]
                file_count = len(bucket)
                rows_total = sum(int(r.get("num_rows") or 0) for r in bucket)
                file_sizes = [int(r.get("num_rows") or 0) for r in bucket]

                year_data.append(
                    {
                        "year": year,
                        "files": file_count,
                        "rows": rows_total,
                        "min_rows_per_file": min(file_sizes) if file_sizes else 0,
                        "max_rows_per_file": max(file_sizes) if file_sizes else 0,
                        "avg_rows_per_file": (rows_total / file_count) if file_count else 0.0,
                    }
                )
                total_files += file_count
                total_rows += rows_total
                all_file_sizes.extend(file_sizes)

            overall_dates = [r["date"] for r in group_rows if r.get("date")]
            distribution = defaultdict(int)
            for r in group_rows:
                distribution[int(r.get("num_rows") or 0)] += 1

            result = {
                "social_group": group,
                "total_files": total_files,
                "total_rows": total_rows,
                "year_range": f"{min(year_buckets)}-{max(year_buckets)}" if year_buckets else "No data",
                "earliest_date": min(overall_dates) if overall_dates else None,
                "latest_date": max(overall_dates) if overall_dates else None,
                "file_size_range": f"{min(all_file_sizes)}-{max(all_file_sizes)}" if all_file_sizes else "No data",
                "years_covered": len(year_buckets),
                "year_by_year": year_data,
                "file_size_distribution": [
                    {"rows_per_file": rows_per_file, "file_count": file_count}
                    for rows_per_file, file_count in sorted(distribution.items())
                ],
            }
            all_results[group] = result

        total_files = sum(data["total_files"] for data in all_results.values())
        total_rows = sum(data["total_rows"] for data in all_results.values())

        return {
            "summary": {
                "total_files": total_files,
                "total_rows": total_rows,
                "social_groups": len(all_results),
                "year_coverage": "2007-2023",
                "analysis_timestamp": datetime.now().isoformat(),
            },
            "by_social_group": all_results,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dataset stats: {str(e)}")

# make sure old temp files are cleared on start-up
cleanup_old_temp_files()

# run the app
if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)
