# package and function imports
from __future__ import annotations

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr

import os
import sys
import uuid
import zipfile
import io
import csv
import time
import threading
import asyncio
import gzip
import traceback
import gc
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Callable, Any, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import uvicorn
import pandas as pd
import numpy as np
import aiosmtplib
from email.message import EmailMessage
from dotenv import load_dotenv
import pyarrow as pa
import pyarrow.parquet as pq
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
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5000"))
ZIP_PART_SIZE_ROWS = int(os.getenv("ZIP_PART_SIZE_ROWS", "100000"))
DEFAULT_MAX_WORKERS = min(4, os.cpu_count() or 2)
ETA_EMA_ALPHA = float(os.getenv("ETA_EMA_ALPHA", "0.25"))
FULL_RANGE_ZIP_COMPRESSLEVEL = int(os.getenv("FULL_RANGE_ZIP_COMPRESSLEVEL", "1"))
FULL_RANGE_CSV_FIRST = os.getenv("FULL_RANGE_CSV_FIRST", "1").lower() in ("1", "true", "yes")
TASK_CLEANUP_MAX_AGE_SECONDS = int(os.getenv("TASK_CLEANUP_MAX_AGE_SECONDS", "86400"))
TEMP_FILE_MAX_AGE_SECONDS = int(os.getenv("TEMP_FILE_MAX_AGE_SECONDS", str(48 * 3600)))
PROGRESS_UPDATE_INTERVAL_SECONDS = float(os.getenv("PROGRESS_UPDATE_INTERVAL_SECONDS", "0.5"))
CSV_PROGRESS_ROWS_PER_UNIT = int(os.getenv("CSV_PROGRESS_ROWS_PER_UNIT", "250000"))
ETA_MIN_SAMPLES = int(os.getenv("ETA_MIN_SAMPLES", "3"))
SAMPLING_PROGRESS_MIN_INTERVAL_SECONDS = float(os.getenv("SAMPLING_PROGRESS_MIN_INTERVAL_SECONDS", "1.0"))
PARQUET_SAMPLE_RATIO_THRESHOLD = float(os.getenv("PARQUET_SAMPLE_RATIO_THRESHOLD", "0.02"))
PARQUET_SAMPLE_MAX_ROWS = int(os.getenv("PARQUET_SAMPLE_MAX_ROWS", "200000"))
SMALL_FILE_ROWS_THRESHOLD = int(os.getenv("SMALL_FILE_ROWS_THRESHOLD", "300000"))
ZIP_STREAM_CHUNK_SIZE = int(os.getenv("ZIP_STREAM_CHUNK_SIZE", str(4 * 1024 * 1024)))

### Utils

# regex for identifying monthly files in the local database
MONTH_FILE_RE = re.compile(
    r"^(?:RC|RS)_(\d{4})-(\d{2})(?:_part\d+_\d+)?\.(csv|csv\.gz|parquet|parq)$",
    re.IGNORECASE,
)

# CSV reader
READ_CSV_KW = dict(
    usecols=[
        "id", "parent id", "text", "author", "time",
        "subreddit", "score", "matched patterns", "source_row"
    ],
    dtype={
        "score": "Int32",
        "source_row": "Int32",
        "id": "string",
        "parent id": "string",
        "text": "string",
        "author": "string",
        "time": "string",
        "subreddit": "string",
        "matched patterns": "string",
    },
    parse_dates=False,
    engine="c",
    memory_map=False,
    low_memory=False,
)

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
        meta = task_meta.setdefault(
            task_id,
            {
                "start_time": now(),
                "total_units": 0,
                "completed_units": 0,
                "ema_units_per_sec": None,
                "last_progress_ts": None,
                "last_progress_units": None,
                "eta_samples": 0,
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
def _update_eta_from_units(task_id: str) -> None:
    with _progress_lock:
        meta = task_meta.get(task_id)
        if not meta:
            return

        total = max(0, int(meta.get("total_units", 0) or 0))
        done = max(0, int(meta.get("completed_units", 0) or 0))
        current_ts = now()

        if total <= 0 or done <= 0:
            task_progress.setdefault(task_id, {})["eta_seconds"] = None
            task_progress.setdefault(task_id, {})["eta_human"] = None
            return

        last_ts = meta.get("last_progress_ts")
        last_units = meta.get("last_progress_units")
        ema_units_per_sec = meta.get("ema_units_per_sec")

        if last_ts is not None and last_units is not None and done > last_units:
            dt = max(1e-6, current_ts - last_ts)
            du = done - last_units
            inst_rate = du / dt

            if ema_units_per_sec is None:
                ema_units_per_sec = inst_rate
            else:
                ema_units_per_sec = (
                    ETA_EMA_ALPHA * inst_rate
                    + (1.0 - ETA_EMA_ALPHA) * ema_units_per_sec
                )

            meta["ema_units_per_sec"] = ema_units_per_sec
            meta["eta_samples"] = int(meta.get("eta_samples", 0)) + 1

        meta["last_progress_ts"] = current_ts
        meta["last_progress_units"] = done

        if not ema_units_per_sec or int(meta.get("eta_samples", 0)) < ETA_MIN_SAMPLES:
            task_progress.setdefault(task_id, {})["eta_seconds"] = None
            task_progress.setdefault(task_id, {})["eta_human"] = None
            return

        remaining = max(0, total - done)
        eta = remaining / max(1e-9, ema_units_per_sec)
        task_progress.setdefault(task_id, {})["eta_seconds"] = eta
        task_progress.setdefault(task_id, {})["eta_human"] = _fmt_eta(eta)

# initial progress metadata
def _init_progress_meta(task_id: str, total_units: int, stage: str) -> None:
    with _progress_lock:
        task_meta[task_id] = {
            "start_time": now(),
            "total_units": max(0, int(total_units)),
            "completed_units": 0,
            "ema_units_per_sec": None,
            "last_progress_ts": None,
            "last_progress_units": None,
            "eta_samples": 0,
        }
        task_progress[task_id] = {"percent": 0.0, "eta_seconds": None, "eta_human": None}
        task_stage[task_id] = stage

# incremental progress evaluation
def _increment_progress(task_id: str, units: int, stage: Optional[str] = None) -> None:
    if units <= 0:
        return

    with _progress_lock:
        meta = task_meta.setdefault(
            task_id,
            {
                "start_time": now(),
                "total_units": 0,
                "completed_units": 0,
                "ema_units_per_sec": None,
                "last_progress_ts": None,
                "last_progress_units": None,
                "eta_samples": 0,
            },
        )
        total_units = int(meta.get("total_units", 0) or 0)
        meta["completed_units"] = min(
            total_units if total_units > 0 else int(meta.get("completed_units", 0) or 0) + int(units),
            int(meta.get("completed_units", 0) or 0) + int(units),
        )

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

# CSV reader stream
def open_csv_stream_local(path_str: str) -> io.TextIOBase:
    path = Path(path_str)
    if path.name.lower().endswith(".csv.gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8", newline="")

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
def _build_sorted_sample_positions(total_rows: int, k: int) -> np.ndarray:
    if total_rows <= 0 or k <= 0:
        return np.empty(0, dtype=np.int64)
    k = min(int(k), int(total_rows))
    if k == total_rows:
        return np.arange(total_rows, dtype=np.int64)
    rng = np.random.default_rng()
    return np.sort(rng.choice(total_rows, size=k, replace=False).astype(np.int64, copy=False))

# write csv chunk to file
def _write_csv_chunk_to_file(df: pd.DataFrame, out_file: io.TextIOBase, *, header: bool) -> int:
    if df.empty:
        return 0
    df.to_csv(out_file, index=False, header=header)
    return len(df)

# copy sampled csv rows to temp csv
def sample_csv_positions_to_temp_csv(
    path_str: str,
    positions: np.ndarray,
    tmp_path: str,
    *,
    chunksize: int = 100_000,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> int:
    if positions.size == 0:
        return 0

    pos_idx = 0
    row_base = 0
    rows_written = 0
    wrote_header = False

    with open_csv_stream_local(path_str) as csv_stream, open(tmp_path, "w", encoding="utf-8", newline="") as out_file:
        for chunk in pd.read_csv(csv_stream, chunksize=chunksize, **READ_CSV_KW):
            chunk_len = len(chunk)
            if chunk_len <= 0:
                continue

            next_base = row_base + chunk_len
            start = pos_idx
            while pos_idx < positions.size and positions[pos_idx] < next_base:
                pos_idx += 1

            if pos_idx > start:
                local_positions = positions[start:pos_idx] - row_base
                sampled = chunk.iloc[local_positions]
                rows_written += _write_csv_chunk_to_file(sampled, out_file, header=not wrote_header)
                wrote_header = True

            row_base = next_base
            if progress_cb is not None:
                progress_cb(chunk_len)

            if pos_idx >= positions.size and row_base >= int(positions[-1]) + 1:
                if positions.size < row_base:
                    break

    return rows_written

# extract row group sizes for parquet sampling
def _parquet_row_group_sizes(pf: pq.ParquetFile) -> List[int]:
    md = pf.metadata
    if md is None:
        return []
    return [md.row_group(i).num_rows for i in range(md.num_row_groups)]

# copy sampled parquet rows to temp csv
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
        with open(tmp_path, "w", encoding="utf-8", newline="") as out_file:
            table = pf.read(columns=columns)
            if table.num_rows <= 0:
                return 0
            sampled = table.take(pa.array(positions.tolist(), type=pa.int64()))
            df = sampled.to_pandas(types_mapper=pd.ArrowDtype)
            return _write_csv_chunk_to_file(df, out_file, header=True)

    cumulative = np.cumsum(np.asarray(row_group_sizes, dtype=np.int64))
    group_ids = np.searchsorted(cumulative, positions, side="right")

    rows_written = 0
    wrote_header = False
    start_idx = 0

    with open(tmp_path, "w", encoding="utf-8", newline="") as out_file:
        while start_idx < positions.size:
            rg_idx = int(group_ids[start_idx])
            end_idx = start_idx
            while end_idx < positions.size and int(group_ids[end_idx]) == rg_idx:
                end_idx += 1

            row_group_start = 0 if rg_idx == 0 else int(cumulative[rg_idx - 1])
            local_positions = (positions[start_idx:end_idx] - row_group_start).astype(np.int64, copy=False)

            table = pf.read_row_group(rg_idx, columns=columns)
            sampled = table.take(pa.array(local_positions.tolist(), type=pa.int64()))
            df = sampled.to_pandas(types_mapper=pd.ArrowDtype)
            rows_written += _write_csv_chunk_to_file(df, out_file, header=not wrote_header)
            wrote_header = True

            if progress_cb is not None:
                progress_cb(int(row_group_sizes[rg_idx]))

            start_idx = end_idx

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
    prefer_parquet = row_groups > 0 and quota <= PARQUET_SAMPLE_MAX_ROWS and ratio <= PARQUET_SAMPLE_RATIO_THRESHOLD and num_rows > SMALL_FILE_ROWS_THRESHOLD
    return primary_path if prefer_parquet else csv_path

# sample any file format to temp csv
def sample_any_to_temp_csv(
    file_info: Dict[str, Any],
    k: int,
    *,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> Tuple[Optional[str], int, str]:
    if k <= 0:
        return None, 0, ""

    num_rows = max(0, int(file_info.get("num_rows") or 0))
    if num_rows <= 0:
        return None, 0, ""

    source_path = choose_sampling_path(file_info, k)
    positions = _build_sorted_sample_positions(num_rows, min(k, num_rows))
    if positions.size == 0:
        return None, 0, source_path

    tmp_name = f"sample_{uuid.uuid4().hex}.csv"
    tmp_path = str(TEMP_DIR / tmp_name)

    try:
        if source_path.lower().endswith((".parquet", ".parq")):
            row_count = sample_parquet_positions_to_temp_csv(
                source_path,
                positions,
                tmp_path,
                columns=list(READ_CSV_KW["usecols"]),
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

# stream CSV to zip
def stream_csv_like_to_zip(zipf: zipfile.ZipFile, path_str: str, arcname: Optional[str] = None, chunk_size: int = ZIP_STREAM_CHUNK_SIZE) -> bool:
    path = Path(path_str)
    if arcname is None:
        arcname = path.name
    lower = path.name.lower()
    if lower.endswith(".csv.gz"):
        arcname = path.name[:-3]
        opener = gzip.open
    else:
        opener = open

    try:
        with opener(path, "rb") as src, zipf.open(arcname, "w") as zf:
            while True:
                chunk = src.read(chunk_size)
                if not chunk:
                    break
                zf.write(chunk)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to stream local file {path}: {e}", file=sys.stderr)
        return False

# stream parquet as csv to zip file
def stream_parquet_as_csv_to_zip(zipf: zipfile.ZipFile, key: str, columns: Optional[List[str]] = None) -> bool:
    path = Path(key)
    csv_name = path.name.replace(".parquet", ".csv").replace(".parq", ".csv")
    try:
        pf = pq.ParquetFile(path)
        with zipf.open(csv_name, "w") as raw_out:
            txt_out = io.TextIOWrapper(raw_out, encoding="utf-8", newline="", write_through=True)
            first_chunk = True
            num_row_groups = pf.metadata.num_row_groups if pf.metadata else 0
            if num_row_groups == 0:
                table = pf.read(columns=columns)
                df = table.to_pandas(types_mapper=pd.ArrowDtype)
                if df.empty:
                    txt_out.flush()
                    return False
                df.to_csv(txt_out, index=False, header=True)
            else:
                for i in range(num_row_groups):
                    table = pf.read_row_group(i, columns=columns)
                    if table.num_rows <= 0:
                        continue
                    df = table.to_pandas(types_mapper=pd.ArrowDtype)
                    if df.empty:
                        continue
                    df.to_csv(txt_out, index=False, header=first_chunk)
                    first_chunk = False
            txt_out.flush()
        return True
    except Exception as e:
        print(f"[ERROR] Failed to convert parquet {key}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return False

# for full month requests, zip the originals from the local dataset
def zip_originals_from_local(task_id: str, files: List[Dict[str, Any]], social_group: str, start_date: str, end_date: str) -> str:
    _set_progress(task_id, stage="Bundling original files")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    zip_filename = f"ISSAC_{social_group}_{start_date}_{end_date}_{task_id}_{timestamp}.zip"
    zip_path = str(TEMP_DIR / zip_filename)

    total_files = len(files)
    _set_progress(task_id, total_units=total_files, completed_units=0)

    zip_lock = threading.Lock()

    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=max(0, min(9, FULL_RANGE_ZIP_COMPRESSLEVEL)),
    ) as zipf:
        def process_one(file_info: Dict[str, Any]) -> bool:
            primary_key = file_info["file_path"]
            csv_key = file_info.get("csv_file_path") or ""
            has_csv = bool(file_info.get("csv_available") and csv_key and os.path.exists(csv_key))
            key = csv_key if has_csv else primary_key
            arcname = os.path.basename(key)
            try:
                with zip_lock:
                    if key.lower().endswith((".parquet", ".parq")):
                        ok = stream_parquet_as_csv_to_zip(zipf, key, columns=list(READ_CSV_KW["usecols"]))
                    else:
                        ok = stream_csv_like_to_zip(zipf, key, arcname)
                return ok
            finally:
                _increment_progress(task_id, 1, stage="Bundling original files")

        with ThreadPoolExecutor(max_workers=DEFAULT_MAX_WORKERS) as executor:
            futures = {executor.submit(process_one, f): f for f in files}
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    print(f"[ERROR] Future failed for {futures[fut].get('file_path')}: {e}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)

    task_result_paths[task_id] = zip_path
    task_results[task_id] = f"/download/{task_id}"
    cleanup_old_tasks()
    return task_results[task_id]

# The main sampling function
def background_sampling(task_id: str, social_group: str, start_date: str, end_date: str, num_docs: Optional[int] = None):
    try:
        overall_timeout = int(os.getenv("DB_OVERALL_TIMEOUT", "300"))
        _set_progress(task_id, stage="Fetching metadata")

        try:
            files = run_coro_in_new_loop(
                fetch_metadata_list(
                    social_group,
                    start_date,
                    end_date,
                    prefer_csv_for_full_range=(num_docs is None),
                ),
                overall_timeout=overall_timeout,
            )
        except Exception as e:
            _set_progress(task_id, stage=f"Error: fetch_metadata failed: {repr(e)}", eta_seconds=0)
            return

        if not files:
            _set_progress(task_id, stage="No files found", eta_seconds=0)
            return

        if num_docs is None:
            zip_originals_from_local(task_id, files, social_group, start_date, end_date)
            _set_progress(task_id, stage="Done", eta_seconds=0)
            return

        total_available = sum((f.get("num_rows") or 0) for f in files)
        num_docs = max(0, min(int(num_docs), total_available))
        quotas = compute_per_file_quotas(files, num_docs)

        total_sampling_units = compute_sampling_total_units(files, quotas)
        total_zip_units = max(1, num_docs // ZIP_PART_SIZE_ROWS + 1)
        total_units = total_sampling_units + total_zip_units
        _init_progress_meta(task_id, total_units=total_units, stage="Sampling documents")

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        zip_filename = f"ISSAC_{social_group}_{start_date}_{end_date}_{task_id}_{timestamp}.zip"
        zip_path = str(TEMP_DIR / zip_filename)

        total_sampled_rows = 0
        chunk_counter = 0
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

        def process_one(file_info: Dict[str, Any], k: int) -> Tuple[Optional[str], int, str]:
            file_key = file_info["file_path"]
            try:
                return sample_any_to_temp_csv(file_info, k, progress_cb=make_progress_cb(file_key))
            finally:
                flush_file_progress(file_key)

        with ThreadPoolExecutor(max_workers=DEFAULT_MAX_WORKERS) as ex:
            futures = [ex.submit(process_one, f, q) for f, q in zip(files, quotas) if q > 0]
            for fut in as_completed(futures):
                tmp_csv_path, row_count, used_path = fut.result()
                if tmp_csv_path and row_count > 0:
                    sampled_temp_files.append((tmp_csv_path, row_count, used_path))
                    total_sampled_rows += row_count
                    _set_progress(task_id, stage=f"Sampled {total_sampled_rows} rows so far")

        if total_sampled_rows == 0:
            with zipfile.ZipFile(
                zip_path,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=max(0, min(9, FULL_RANGE_ZIP_COMPRESSLEVEL)),
            ):
                pass
            task_result_paths[task_id] = zip_path
            task_results[task_id] = f"/download/{task_id}"
            _set_progress(task_id, completed_units=total_units, stage="No matching rows found", eta_seconds=0)
            cleanup_old_tasks()
            return

        with zipfile.ZipFile(
            zip_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=max(0, min(9, FULL_RANGE_ZIP_COMPRESSLEVEL)),
        ) as zipf:
            current_chunk_name: Optional[str] = None
            current_chunk_path: Optional[str] = None
            current_chunk_file: Optional[io.TextIOBase] = None
            current_chunk_writer: Optional[csv.DictWriter] = None
            current_chunk_rows = 0
            canonical_columns: List[str] = list(READ_CSV_KW["usecols"])

            def start_new_chunk() -> None:
                nonlocal chunk_counter, current_chunk_name, current_chunk_path, current_chunk_file, current_chunk_writer, current_chunk_rows
                chunk_counter += 1
                current_chunk_name = f"ISSAC_{social_group}_{start_date}_{end_date}_{chunk_counter}.csv"
                current_chunk_path = str(TEMP_DIR / f"merged_{chunk_counter}_{uuid.uuid4().hex}.csv")
                current_chunk_file = open(current_chunk_path, "w", encoding="utf-8", newline="")
                current_chunk_writer = csv.DictWriter(current_chunk_file, fieldnames=canonical_columns, extrasaction="ignore")
                current_chunk_writer.writeheader()
                current_chunk_rows = 0

            def flush_chunk_to_zip() -> None:
                nonlocal current_chunk_name, current_chunk_path, current_chunk_file, current_chunk_writer, current_chunk_rows
                if current_chunk_file is None or current_chunk_path is None or current_chunk_name is None:
                    return
                current_chunk_file.close()
                zipf.write(current_chunk_path, current_chunk_name)
                os.remove(current_chunk_path)
                current_chunk_name = None
                current_chunk_path = None
                current_chunk_file = None
                current_chunk_writer = None
                current_chunk_rows = 0
                _increment_progress(task_id, 1, stage="Zipping sampled rows")

            for tmp_csv_path, row_count, used_path in sampled_temp_files:
                try:
                    with open(tmp_csv_path, "r", encoding="utf-8", newline="") as src:
                        reader = csv.DictReader(src)
                        if current_chunk_file is None:
                            start_new_chunk()

                        for row in reader:
                            assert current_chunk_writer is not None
                            current_chunk_writer.writerow({col: row.get(col, "") for col in canonical_columns})
                            current_chunk_rows += 1
                            if current_chunk_rows >= ZIP_PART_SIZE_ROWS:
                                flush_chunk_to_zip()
                                start_new_chunk()
                finally:
                    try:
                        os.remove(tmp_csv_path)
                    except FileNotFoundError:
                        pass

            flush_chunk_to_zip()

        if not os.path.exists(zip_path):
            _set_progress(task_id, stage="Error: No output generated", eta_seconds=0)
            return

        task_result_paths[task_id] = zip_path
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

# download button
@app.get("/download/{task_id}")
async def download_result(task_id: str):
    path = task_result_paths.get(task_id)

    if (not path or not os.path.exists(path)) and TEMP_DIR.exists():
        matches = sorted(TEMP_DIR.glob(f"*_{task_id}_*.zip"), key=lambda x: x.stat().st_mtime, reverse=True)
        if matches:
            path = str(matches[0])
            task_result_paths[task_id] = path
            task_results[task_id] = f"/download/{task_id}"

    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found or expired")
    return FileResponse(path, media_type="application/zip", filename=os.path.basename(path))

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

    if not social_group or not start_date or not end_date:
        raise HTTPException(status_code=400, detail="Missing required fields")

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

    cleanup_old_tasks()
    cleanup_old_temp_files()

    task_id = str(uuid.uuid4())
    with _progress_lock:
        task_meta[task_id] = {"start_time": now(), "total_units": 0, "completed_units": 0}
        task_progress[task_id] = {"percent": 0.0, "eta_seconds": None, "eta_human": None}
        task_stage[task_id] = "Queued"

    background_tasks.add_task(background_sampling, task_id, social_group, start_date, end_date, num_docs)
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
