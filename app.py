from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from fastapi.responses import JSONResponse

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
from typing import Optional, List, Dict, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import asyncpg
import boto3
from boto3.s3.transfer import TransferConfig
import aiosmtplib
from email.message import EmailMessage
from dotenv import load_dotenv

import pyarrow as pa
import pyarrow.parquet as pq
import s3fs


load_dotenv()

# CORS configuration - allow specific origins
# When allow_credentials=True, you cannot use allow_origins=["*"]
# Get allowed origins from environment variable or default to localhost:3000
ALLOWED_ORIGINS_ENV = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:3001"
)
# Parse and clean origins, filtering out empty strings
ALLOWED_ORIGINS = [
    origin.strip() 
    for origin in ALLOWED_ORIGINS_ENV.split(",") 
    if origin.strip()
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
)

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT") or "587")
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
ISSUE_RECEIVER_EMAIL = os.getenv("ISSUE_RECEIVER_EMAIL")

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")

SUPABASE_S3_ENDPOINT = os.getenv("SUPABASE_S3_ENDPOINT") 
SUPABASE_S3_REGION = os.getenv("SUPABASE_S3_REGION", "us-east-1")
SUPABASE_S3_ACCESS_KEY_ID = os.getenv("SUPABASE_S3_ACCESS_KEY_ID")
SUPABASE_S3_SECRET_ACCESS_KEY = os.getenv("SUPABASE_S3_SECRET_ACCESS_KEY")
SUPABASE_BUCKET_NAME = os.getenv("SUPABASE_BUCKET_NAME") 

s3_client = boto3.client(
    "s3",
    endpoint_url=SUPABASE_S3_ENDPOINT,
    aws_access_key_id=SUPABASE_S3_ACCESS_KEY_ID,
    aws_secret_access_key=SUPABASE_S3_SECRET_ACCESS_KEY,
    region_name=SUPABASE_S3_REGION,
)

s3_fs = s3fs.S3FileSystem(
    key=SUPABASE_S3_ACCESS_KEY_ID,
    secret=SUPABASE_S3_SECRET_ACCESS_KEY,
    client_kwargs={"endpoint_url": SUPABASE_S3_ENDPOINT, "region_name": SUPABASE_S3_REGION},
)

TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)

task_progress: Dict[str, Dict] = {} 
task_stage: Dict[str, str] = {}
task_results: Dict[str, str] = {}  
task_meta: Dict[str, Dict] = {}     
_progress_lock = threading.Lock()

ASSUMED_UPLOAD_BPS = 10 * 1024 * 1024 
ZIP_PART_SIZE_ROWS = 100_000          
DEFAULT_MAX_WORKERS = min(2, os.cpu_count() or 2)  # Limited to 2 workers for memory efficiency
ETA_EMA_ALPHA = float(os.getenv("ETA_EMA_ALPHA", "0.25"))
FULL_RANGE_ZIP_COMPRESSLEVEL = int(os.getenv("FULL_RANGE_ZIP_COMPRESSLEVEL", "1"))

# Task cleanup configuration
TASK_CLEANUP_MAX_AGE_SECONDS = int(os.getenv("TASK_CLEANUP_MAX_AGE_SECONDS", "86400"))  # 24 hours default
FULL_RANGE_CSV_FIRST = os.getenv("FULL_RANGE_CSV_FIRST", "0").lower() in ("1", "true", "yes")

def now() -> float:
    return time.time()

def _fmt_eta(seconds: Optional[float]) -> Optional[str]:
    if seconds is None:
        return None
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m"

def _set_progress(task_id: str, *, stage: Optional[str] = None,
                  completed_units: Optional[int] = None,
                  total_units: Optional[int] = None,
                  eta_seconds: Optional[float] = None):
    with _progress_lock:
        if stage is not None:
            prev_stage = task_stage.get(task_id)
            task_stage[task_id] = stage
        meta = task_meta.setdefault(task_id, {})
        prog = task_progress.setdefault(task_id, {"percent": 0.0, "eta_seconds": None, "eta_human": None})
        if stage is not None and stage != prev_stage:
            meta.pop("eta_last_completed_units", None)
            meta.pop("eta_last_ts", None)
            meta.pop("eta_per_unit_ema", None)
        if total_units is not None:
            meta["total_units"] = total_units
        if completed_units is not None:
            meta["completed_units"] = completed_units

        tu = meta.get("total_units", 0) or 0
        cu = meta.get("completed_units", 0) or 0
        percent = (cu / tu * 100.0) if tu > 0 else (100.0 if stage == "Done" else 0.0)
        prog["percent"] = max(0.0, min(100.0, percent))

        if eta_seconds is not None:
            prog["eta_seconds"] = max(0, int(eta_seconds))
            prog["eta_human"] = _fmt_eta(prog["eta_seconds"])

def _update_eta_from_units(task_id: str):
    with _progress_lock:
        meta = task_meta.get(task_id, {})
        start_time = meta.get("start_time")
        cu = int(meta.get("completed_units", 0) or 0)
        tu = int(meta.get("total_units", 0) or 0)
        now_ts = now()

        if not start_time or cu <= 0 or tu <= 0 or cu > tu:
            return

        prev_cu = int(meta.get("eta_last_completed_units", 0) or 0)
        prev_ts = float(meta.get("eta_last_ts", start_time) or start_time)
        prev_ema = meta.get("eta_per_unit_ema")
        prev_eta = task_progress.get(task_id, {}).get("eta_seconds")

        completed_delta = cu - prev_cu
        elapsed_delta = max(0.0, now_ts - prev_ts)

        if completed_delta > 0 and elapsed_delta > 0:
            inst_per_unit = elapsed_delta / completed_delta
        else:
            elapsed_total = max(0.0, now_ts - start_time)
            inst_per_unit = (elapsed_total / cu) if cu > 0 else None

        if inst_per_unit is None:
            return

        if prev_ema is None:
            ema_per_unit = inst_per_unit
        else:
            ema_per_unit = (ETA_EMA_ALPHA * inst_per_unit) + ((1.0 - ETA_EMA_ALPHA) * float(prev_ema))

        remaining_units = max(0, tu - cu)
        raw_eta = remaining_units * ema_per_unit

        # Clamp estimate drift to reduce visible oscillation in UI.
        if prev_eta is not None:
            lower = max(0.0, float(prev_eta) * 0.6)
            upper = max(lower + 1.0, float(prev_eta) * 1.4)
            eta = min(upper, max(lower, raw_eta))
        else:
            eta = raw_eta

        meta["eta_last_completed_units"] = cu
        meta["eta_last_ts"] = now_ts
        meta["eta_per_unit_ema"] = ema_per_unit
        prog = task_progress.setdefault(task_id, {"percent": 0.0, "eta_seconds": None, "eta_human": None})
        prog["eta_seconds"] = max(0, int(eta))
        prog["eta_human"] = _fmt_eta(prog["eta_seconds"])

def cleanup_old_tasks(max_age_seconds: int = TASK_CLEANUP_MAX_AGE_SECONDS):
    """Remove tasks older than max_age_seconds to prevent memory leaks."""
    current_time = now()
    with _progress_lock:
        tasks_to_remove = []
        for task_id, meta in task_meta.items():
            start_time = meta.get("start_time", 0)
            if current_time - start_time > max_age_seconds:
                tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            task_progress.pop(task_id, None)
            task_stage.pop(task_id, None)
            task_results.pop(task_id, None)
            task_meta.pop(task_id, None)
        
        if tasks_to_remove:
            print(f"[INFO] Cleaned up {len(tasks_to_remove)} old task(s)", file=sys.stderr)


class IssueReport(BaseModel):
    email: EmailStr
    description: str

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


async def fetch_metadata(
    social_group: str,
    start_month: str,
    end_month: str,
    *,
    prefer_csv_for_full_range: bool = False,
):
    conn = await asyncpg.connect(SUPABASE_DB_URL)
    try:
        base_where = """
        WHERE social_group = $1
          AND date >= (($2 || '-01')::date)
          AND date <  (($3 || '-01')::date + INTERVAL '1 MONTH')
        ORDER BY date ASC
        """
        query_default = f"""
        SELECT file_path, num_rows
        FROM metadata
        {base_where};
        """
        # CSV-first export path for full-range jobs. Keeps parquet fallback.
        query_csv_first = f"""
        SELECT
            CASE
                WHEN csv_available IS TRUE AND csv_file_path IS NOT NULL
                    THEN csv_file_path
                ELSE file_path
            END AS file_path,
            num_rows,
            file_path AS source_file_path,
            CASE
                WHEN csv_available IS TRUE AND csv_file_path IS NOT NULL
                    THEN TRUE
                ELSE FALSE
            END AS using_csv_companion
        FROM metadata
        {base_where};
        """

        query_to_run = query_csv_first if prefer_csv_for_full_range else query_default
        try:
            async with conn.transaction():
                async for record in conn.cursor(query_to_run, social_group, start_month, end_month):
                    out = {
                        "file_path": record["file_path"],
                        "num_rows": record["num_rows"],
                    }
                    if "source_file_path" in record:
                        out["source_file_path"] = record["source_file_path"]
                    if "using_csv_companion" in record:
                        out["using_csv_companion"] = bool(record["using_csv_companion"])
                    yield out
        except asyncpg.UndefinedColumnError:
            # Backward compatible fallback for environments without csv_* columns.
            async with conn.transaction():
                async for record in conn.cursor(query_default, social_group, start_month, end_month):
                    yield {"file_path": record["file_path"], "num_rows": record["num_rows"]}
    finally:
        try:
            await asyncio.sleep(0) 
            await conn.close()
        except Exception as e:
            print(f"[ERROR] Connection close failed: {e}", file=sys.stderr)

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


def s3_open_csv_stream(key: str) -> io.TextIOBase:
    resp = s3_client.get_object(Bucket=SUPABASE_BUCKET_NAME, Key=key)
    body = resp["Body"] 

    is_gzip = key.lower().endswith(".gz")
    enc = resp.get("ContentEncoding")
    ctype = resp.get("ContentType")
    if enc and "gzip" in str(enc).lower():
        is_gzip = True
    if ctype and "gzip" in str(ctype).lower():
        is_gzip = True

    if is_gzip:
        gz = gzip.GzipFile(fileobj=body)
        return io.TextIOWrapper(gz, encoding="utf-8")
    else:
        return io.TextIOWrapper(body, encoding="utf-8")

def stream_object_to_zip(
    zipf: zipfile.ZipFile,
    key: str,
    arcname: Optional[str] = None,
    chunk_size: int = 1024 * 1024,
    stats: Optional[Dict[str, float]] = None,
):
    if arcname is None:
        arcname = os.path.basename(key)
    
    try:
        fetch_started = now()
        obj = s3_client.get_object(Bucket=SUPABASE_BUCKET_NAME, Key=key)
        fetch_elapsed = now() - fetch_started
        stream_started = now()
        zip_write_seconds = 0.0
        source_bytes = 0
        chunk_count = 0
        with zipf.open(arcname, "w") as zf:
            for chunk in obj["Body"].iter_chunks(chunk_size=chunk_size):
                if chunk:
                    chunk_count += 1
                    source_bytes += len(chunk)
                    write_started = now()
                    zf.write(chunk)
                    zip_write_seconds += (now() - write_started)
        stream_elapsed = now() - stream_started
        if stats is not None:
            stats["fetch_seconds"] = float(fetch_elapsed)
            stats["stream_seconds"] = float(stream_elapsed)
            stats["zip_write_seconds"] = float(zip_write_seconds)
            stats["source_read_wait_seconds"] = float(max(0.0, stream_elapsed - zip_write_seconds))
            stats["source_bytes"] = float(source_bytes)
            stats["chunk_count"] = float(chunk_count)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to stream {key}: {e}", file=sys.stderr)
        return False

def _make_upload_progress_callback(task_id: str, file_size: int) -> Callable[[int], None]:
    started = now()
    uploaded = 0
    last_emit = 0.0
    lock = threading.Lock()

    def _callback(bytes_amount: int) -> None:
        nonlocal uploaded, last_emit
        with lock:
            uploaded += int(bytes_amount)
            current = now()
            if (current - last_emit) < 1.0 and uploaded < file_size:
                return
            last_emit = current

            elapsed = max(0.001, current - started)
            bps = uploaded / elapsed
            remaining = max(0, file_size - uploaded)
            eta = (remaining / bps) if bps > 0 else 0
            _set_progress(task_id, eta_seconds=max(0.0, eta))

    return _callback

def upload_file_and_get_presigned(
    local_path: str,
    dest_key: str,
    expires_in: int = 3600,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> str:
    # Use TransferConfig with single-threaded upload to avoid multipart ordering issues
    # This ensures parts are uploaded in order and prevents "parts not in ascending order" errors
    transfer_config = TransferConfig(
        multipart_threshold=1024 * 1024 * 5,  # 5MB threshold for multipart
        max_concurrency=1,  # Single thread to ensure part ordering
        use_threads=True,  # Use threads but with max_concurrency=1 to ensure sequential part uploads
        multipart_chunksize=1024 * 1024 * 8,  # 8MB chunks
    )
    s3_client.upload_file(
        local_path, 
        SUPABASE_BUCKET_NAME, 
        dest_key,
        Config=transfer_config,
        Callback=progress_callback
    )
    
    # Generate presigned URL
    presigned = s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": SUPABASE_BUCKET_NAME, "Key": dest_key},
        ExpiresIn=expires_in
    )
    
    # Clean up local file after successful upload
    try:
        if os.path.exists(local_path):
            os.remove(local_path)
            print(f"[INFO] Cleaned up local file after upload: {local_path}", file=sys.stderr)
    except Exception as e:
        print(f"[WARNING] Failed to cleanup local file {local_path}: {e}", file=sys.stderr)
        # Don't fail the upload if cleanup fails
    
    return presigned


def sample_csv_reservoir_from_s3(key: str, k: int, chunksize: int = 50_000) -> pd.DataFrame:
    if k <= 0:
        return pd.DataFrame()

    rng = np.random.default_rng()
    reservoir = None
    seen = 0

    with s3_open_csv_stream(key) as csv_stream:
        for chunk in pd.read_csv(csv_stream, chunksize=chunksize, **READ_CSV_KW):
            if chunk.empty:
                continue

            start_idx = 0
            if reservoir is None:
                take = min(k, len(chunk))
                reservoir = chunk.iloc[:take].copy()
                seen += take
                start_idx = take
            elif len(reservoir) < k:
                need = min(k - len(reservoir), len(chunk))
                reservoir = pd.concat([reservoir, chunk.iloc[:need].copy()], ignore_index=True)
                seen += need
                start_idx = need

            for idx in range(start_idx, len(chunk)):
                seen += 1
                j = int(rng.integers(0, seen))
                if j < len(reservoir):
                    reservoir.iloc[j] = chunk.iloc[idx]

    if reservoir is None:
        return pd.DataFrame()
    return reservoir

def sample_parquet_reservoir_from_s3(
    key: str,
    k: int,
    *,
    time_col: str = "time",
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    if k <= 0:
        return pd.DataFrame()

    # Use module-level helper functions (no duplicates)
    start_ts = _norm_month(start_ts)
    end_ts   = _norm_month_end(end_ts)

    path = f"{SUPABASE_BUCKET_NAME}/{key}" if not key.startswith(f"{SUPABASE_BUCKET_NAME}/") else key
    
    with s3_fs.open(path, "rb") as parquet_file:
        pf = pq.ParquetFile(parquet_file)

        rg_indices = []
        time_idx = None
        try:
            schema = pf.schema_arrow
            if time_col in schema.names:
                time_idx = schema.get_field_index(time_col)
        except Exception:
            pass

        def _rg_overlaps(rg_meta):
            if time_idx is None:
                return True 
            col = rg_meta.column(time_idx)
            stats = col.statistics
            if stats is None or stats.min is None or stats.max is None:
                return True
            rg_min = stats.min
            rg_max = stats.max
            if isinstance(rg_min, (bytes, bytearray)):
                rg_min = rg_min.decode("utf-8", "ignore")
            if isinstance(rg_max, (bytes, bytearray)):
                rg_max = rg_max.decode("utf-8", "ignore")

            if start_ts and rg_max < start_ts:
                return False
            if end_ts and rg_min > end_ts:
                return False
            return True

        md = pf.metadata
        for i in range(md.num_row_groups):
            rg = md.row_group(i)
            if _rg_overlaps(rg):
                rg_indices.append(i)

        rng = np.random.default_rng()
        reservoir = None
        seen = 0

        read_cols = columns or None
        for i in rg_indices:
            tbl: pa.Table = pf.read_row_group(i, columns=read_cols)
            df = tbl.to_pandas(types_mapper=pd.ArrowDtype).reset_index(drop=True)

            if time_col in df.columns and (start_ts or end_ts):
                if start_ts:
                    df = df[df[time_col] >= start_ts]
                if end_ts:
                    df = df[df[time_col] <= end_ts]

            if df.empty:
                continue

            start_idx = 0
            if reservoir is None:
                take = min(k, len(df))
                reservoir = df.iloc[:take].copy()
                seen += take
                start_idx = take
            elif len(reservoir) < k:
                need = min(k - len(reservoir), len(df))
                reservoir = pd.concat([reservoir, df.iloc[:need].copy()], ignore_index=True)
                seen += need
                start_idx = need

            for idx in range(start_idx, len(df)):
                seen += 1
                j = int(rng.integers(0, seen))
                if j < len(reservoir):
                    reservoir.iloc[j] = df.iloc[idx]

        if reservoir is None:
            return pd.DataFrame()
        return reservoir

def sample_any_from_s3(key: str, k: int, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    if k <= 0:
        return None
    lower = key.lower()
    if lower.endswith(".parquet") or lower.endswith(".parq"):
        return sample_parquet_reservoir_from_s3(
            key, k,
            time_col="time",
            start_ts=start_date,
            end_ts=end_date,
            columns=None
        )
    return sample_csv_reservoir_from_s3(key, k)


def sample_any_to_temp_csv(
    key: str,
    k: int,
    start_date: str,
    end_date: str
) -> Tuple[Optional[str], int]:
    """Sample from a source file and write the sampled rows directly to a temp CSV."""
    if k <= 0:
        return None, 0

    tmp_name = f"sample_{uuid.uuid4().hex}.csv"
    tmp_path = os.path.join(TEMP_DIR, tmp_name)

    try:
        df = sample_any_from_s3(key, k, start_date, end_date)
        if df is None or df.empty:
            return None, 0

        # Keep source provenance stable in output even when source data omits it.
        if "source_row" not in df.columns:
            df["source_row"] = pd.Series([pd.NA] * len(df), dtype="Int64")
        else:
            df["source_row"] = pd.to_numeric(df["source_row"], errors="coerce").astype("Int64")

        row_count = len(df)
        df.to_csv(tmp_path, index=False)

        # Ensure worker memory is released before returning to the caller.
        del df
        gc.collect()
        return tmp_path, row_count
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def compute_per_file_quotas(files: List[Dict], num_docs: int) -> List[int]:
    if not files:
        return []

    total = sum(max(0, f.get("num_rows", 0) or 0) for f in files)
    if total <= 0:
        base = num_docs // len(files)
        quotas = [base] * len(files)
        quotas[-1] += num_docs - base * len(files)
        return quotas

    raw_quotas: List[float] = []
    quotas: List[int] = []
    for f in files:
        share = (f.get("num_rows", 0) or 0) / total
        raw = num_docs * share
        raw_quotas.append(raw)
        quotas.append(int(raw))

    remaining = num_docs - sum(quotas)
    if remaining > 0:
        order = sorted(
            range(len(files)),
            key=lambda i: (raw_quotas[i] - quotas[i], -i),
            reverse=True
        )
        for idx in order[:remaining]:
            quotas[idx] += 1

    return quotas

def _norm_month(s):
    """Normalize month string (YYYY-MM) to datetime format (YYYY-MM-01T00:00:00)."""
    if s and len(s) == 7: 
        return s + "-01T00:00:00"
    return s

def _norm_month_end(s):
    """Normalize month string (YYYY-MM) to end of month datetime format."""
    if s and len(s) == 7:
        try:
            parts = s.split("-")
            if len(parts) != 2:
                return s
            year, month = parts
            import calendar
            last_day = calendar.monthrange(int(year), int(month))[1]
            return f"{s}-{last_day:02d}T23:59:59"
        except (ValueError, IndexError, calendar.IllegalMonthError):
            # If parsing fails, return original string
            return s
    return s

def convert_parquet_to_csv_and_zip(zipf: zipfile.ZipFile, key: str, arcname: str, 
                                   start_date: str, end_date: str) -> bool:
    """
    Convert parquet to CSV using row group streaming for memory efficiency.
    Processes parquet file in chunks (row groups) to avoid loading entire file into memory.
    """
    try:
        path = f"{SUPABASE_BUCKET_NAME}/{key}" if not key.startswith(f"{SUPABASE_BUCKET_NAME}/") else key
        
        csv_name = arcname.replace('.parquet', '.csv').replace('.parq', '.csv')
        
        with s3_fs.open(path, "rb") as parquet_file:
            pf = pq.ParquetFile(parquet_file)
            
            # Check if file has row groups
            num_row_groups = pf.metadata.num_row_groups if pf.metadata else 0
            
            if num_row_groups == 0:
                # Fallback: read entire file if no row groups
                table = pf.read()
                df = table.to_pandas()
                if df.empty:
                    return False
                
                from io import StringIO
                csv_buffer = StringIO()
                df.to_csv(csv_buffer, index=False)
                csv_content = csv_buffer.getvalue()
                with zipf.open(csv_name, "w") as zf:
                    zf.write(csv_content.encode('utf-8'))
                return True
            
            # Stream row groups for memory efficiency
            with zipf.open(csv_name, "w") as zf:
                first_row_group = True
                
                for i in range(num_row_groups):
                    # Read one row group at a time
                    table = pf.read_row_group(i)
                    df = table.to_pandas()
                    
                    if df.empty:
                        continue
                    
                    # Write CSV header only for first row group
                    from io import StringIO
                    csv_buffer = StringIO()
                    df.to_csv(csv_buffer, index=False, header=first_row_group)
                    csv_content = csv_buffer.getvalue()
                    
                    # For subsequent row groups, skip header line
                    if not first_row_group:
                        # Remove header line from subsequent chunks
                        lines = csv_content.split('\n')
                        if len(lines) > 1:
                            csv_content = '\n'.join(lines[1:])
                    
                    zf.write(csv_content.encode('utf-8'))
                    first_row_group = False
            
            return True
        
    except Exception as e:
        print(f"[ERROR] Failed to convert parquet {key}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return False

def process_file_fast(zipf: zipfile.ZipFile, file_info: Dict, start_date: str, end_date: str) -> bool:
    key = file_info["file_path"]
    arcname = os.path.basename(key)
    
    try:
        if key.lower().endswith(('.parquet', '.parq')):
            return convert_parquet_to_csv_and_zip(zipf, key, arcname, start_date, end_date)
        else:
            if not arcname.lower().endswith('.csv'):
                arcname = arcname.rsplit('.', 1)[0] + '.csv'
            return stream_object_to_zip(zipf, key, arcname)
    except Exception as e:
        print(f"[ERROR] Failed to process file {key}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return False

def zip_originals_from_s3(task_id: str, files: List[Dict], social_group: str,
                          start_date: str, end_date: str) -> str:
    """
    Process files in parallel using ThreadPoolExecutor for better performance.
    Parquet files are converted to CSV using streaming row groups for memory efficiency.
    """
    _set_progress(task_id, stage="Bundling original files (parallelized)")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    zip_filename = f"ISAAC_{social_group}_{start_date}_{end_date}_{timestamp}.zip"
    zip_path = os.path.join(TEMP_DIR, zip_filename)

    total_files = len(files)
    _set_progress(task_id, total_units=total_files + 1, completed_units=0)
    csv_stream_candidates = sum(
        1 for f in files if not str(f.get("file_path", "")).lower().endswith((".parquet", ".parq"))
    )
    parquet_fallback_candidates = total_files - csv_stream_candidates
    _set_progress(task_id, stage=f"Preparing files for download ({total_files})")
    print(
        f"[INFO] Full-range CSV-first plan: csv_stream_candidates={csv_stream_candidates}, "
        f"parquet_fallback_candidates={parquet_fallback_candidates}, total_files={total_files}",
        file=sys.stderr,
    )

    pipeline_started = now()
    # Use a lock for thread-safe zip file operations
    zip_lock = threading.Lock()
    completed = 0
    successful = 0
    csv_stream_success = 0
    parquet_convert_success = 0
    csv_fetch_seconds = 0.0
    csv_stream_seconds = 0.0
    csv_stream_wait_seconds = 0.0
    parquet_convert_seconds = 0.0
    zip_write_seconds = 0.0
    csv_source_bytes = 0
    parquet_converted_bytes = 0
    
    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=max(0, min(9, FULL_RANGE_ZIP_COMPRESSLEVEL)),
    ) as zipf:
        bundle_started = now()
        def process_file_with_lock(file_info: Dict) -> bool:
            """Process a single file and update progress in thread-safe manner."""
            nonlocal completed, successful
            nonlocal csv_stream_success, parquet_convert_success
            nonlocal csv_fetch_seconds, csv_stream_seconds, csv_stream_wait_seconds
            nonlocal parquet_convert_seconds, zip_write_seconds
            nonlocal csv_source_bytes, parquet_converted_bytes
            try:
                # Process the file (parquet conversion happens outside lock)
                key = file_info["file_path"]
                arcname = os.path.basename(key)
                file_started = now()
                
                if key.lower().endswith(('.parquet', '.parq')):
                    # Convert parquet to CSV in memory, then add to zip
                    csv_name = arcname.replace('.parquet', '.csv').replace('.parq', '.csv')
                    path = f"{SUPABASE_BUCKET_NAME}/{key}" if not key.startswith(f"{SUPABASE_BUCKET_NAME}/") else key
                    
                    try:
                        with s3_fs.open(path, "rb") as parquet_file:
                            pf = pq.ParquetFile(parquet_file)
                            num_row_groups = pf.metadata.num_row_groups if pf.metadata else 0
                            
                            if num_row_groups == 0:
                                # Fallback: read entire file
                                convert_started = now()
                                table = pf.read()
                                df = table.to_pandas()
                                if df.empty:
                                    return False
                                
                                from io import StringIO
                                csv_buffer = StringIO()
                                df.to_csv(csv_buffer, index=False)
                                csv_content = csv_buffer.getvalue().encode('utf-8')
                                convert_elapsed = now() - convert_started
                                
                                # Add to zip file (thread-safe operation)
                                write_started = now()
                                with zip_lock:
                                    with zipf.open(csv_name, "w") as zf:
                                        zf.write(csv_content)
                                    parquet_convert_success += 1
                                    parquet_convert_seconds += convert_elapsed
                                    write_elapsed = now() - write_started
                                    zip_write_seconds += write_elapsed
                                    parquet_converted_bytes += len(csv_content)
                                return True
                            
                            # Stream row groups for memory efficiency
                            convert_started = now()
                            csv_chunks = []
                            first_row_group = True
                            
                            for i in range(num_row_groups):
                                table = pf.read_row_group(i)
                                df = table.to_pandas()
                                
                                if df.empty:
                                    continue
                                
                                from io import StringIO
                                csv_buffer = StringIO()
                                df.to_csv(csv_buffer, index=False, header=first_row_group)
                                csv_content = csv_buffer.getvalue()
                                
                                # For subsequent row groups, skip header line
                                if not first_row_group:
                                    lines = csv_content.split('\n')
                                    if len(lines) > 1:
                                        csv_content = '\n'.join(lines[1:])
                                
                                csv_chunks.append(csv_content.encode('utf-8'))
                                first_row_group = False
                            convert_elapsed = now() - convert_started
                        
                        # Add to zip file (thread-safe operation)
                        chunk_bytes = sum(len(chunk) for chunk in csv_chunks)
                        write_started = now()
                        with zip_lock:
                            with zipf.open(csv_name, "w") as zf:
                                for chunk in csv_chunks:
                                    zf.write(chunk)
                            parquet_convert_success += 1
                            parquet_convert_seconds += convert_elapsed
                            write_elapsed = now() - write_started
                            zip_write_seconds += write_elapsed
                            parquet_converted_bytes += chunk_bytes
                        
                        return True
                    except Exception as e:
                        print(f"[ERROR] Failed to convert parquet {key}: {e}", file=sys.stderr)
                        traceback.print_exc(file=sys.stderr)
                        return False
                else:
                    # CSV files - stream directly
                    stream_stats: Dict[str, float] = {}
                    with zip_lock:
                        ok = stream_object_to_zip(zipf, key, arcname, stats=stream_stats)
                        if ok:
                            csv_stream_success += 1
                            csv_fetch_seconds += stream_stats.get("fetch_seconds", 0.0)
                            csv_stream_seconds += stream_stats.get("stream_seconds", 0.0)
                            csv_stream_wait_seconds += stream_stats.get("source_read_wait_seconds", 0.0)
                            zip_write_seconds += stream_stats.get("zip_write_seconds", 0.0)
                            csv_source_bytes += int(stream_stats.get("source_bytes", 0.0))
                        return ok
                
            except Exception as e:
                print(f"[ERROR] File processing failed for {file_info.get('file_path', 'unknown')}: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                return False
            finally:
                # Update progress thread-safely
                with zip_lock:
                    completed += 1
                    _set_progress(task_id, completed_units=completed)
                    _update_eta_from_units(task_id)
                    
                    if completed % max(1, total_files // 10) == 0:
                        _set_progress(
                            task_id,
                            stage=f"Processed {completed}/{total_files} files",
                        )
        
        # Process files in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=DEFAULT_MAX_WORKERS) as executor:
            futures = [executor.submit(process_file_with_lock, f) for f in files]
            
            for fut in as_completed(futures):
                try:
                    result = fut.result()
                    if result:
                        with zip_lock:
                            successful += 1
                except Exception as e:
                    print(f"[ERROR] Future failed: {e}", file=sys.stderr)
        bundling_elapsed = now() - bundle_started

    _set_progress(task_id, stage="Uploading to Supabase Storage")
    object_name = f"samples/{zip_filename}"
    file_size = os.path.getsize(zip_path)
    
    estimated_upload_time = max(1, file_size / ASSUMED_UPLOAD_BPS)
    _set_progress(task_id, eta_seconds=estimated_upload_time)
    upload_progress_cb = _make_upload_progress_callback(task_id, file_size)

    upload_started = now()
    presigned = upload_file_and_get_presigned(
        zip_path,
        object_name,
        expires_in=3600,
        progress_callback=upload_progress_cb,
    )
    upload_elapsed = now() - upload_started

    _set_progress(task_id, completed_units=total_files + 1, eta_seconds=0)
    print(
        f"[INFO] Full-range CSV-first completed: successful={successful}/{total_files}, "
        f"csv_stream_candidates={csv_stream_candidates}, "
        f"parquet_fallback_candidates={parquet_fallback_candidates}, "
        f"csv_stream_success={csv_stream_success}, "
        f"parquet_convert_success={parquet_convert_success}, "
        f"csv_fetch_seconds={csv_fetch_seconds:.2f}, "
        f"csv_stream_seconds={csv_stream_seconds:.2f}, "
        f"csv_stream_wait_seconds={csv_stream_wait_seconds:.2f}, "
        f"parquet_convert_seconds={parquet_convert_seconds:.2f}, "
        f"zip_write_seconds={zip_write_seconds:.2f}, "
        f"csv_source_mb={(csv_source_bytes / (1024 * 1024)):.2f}, "
        f"parquet_converted_mb={(parquet_converted_bytes / (1024 * 1024)):.2f}, "
        f"bundling_seconds={bundling_elapsed:.2f}, "
        f"upload_seconds={upload_elapsed:.2f}, "
        f"total_pipeline_seconds={(now() - pipeline_started):.2f}",
        file=sys.stderr,
    )
    
    # Clean up old tasks after successful completion
    cleanup_old_tasks()
    
    return presigned

def background_sampling(task_id: str, social_group: str, start_date: str,
                        end_date: str, num_docs: Optional[int] = None):
    try:
        max_tries = int(os.getenv("DB_RETRIES", "3"))
        overall_timeout = int(os.getenv("DB_OVERALL_TIMEOUT", "300"))

        _set_progress(task_id, stage="Fetching metadata")
        try:
            files = run_coro_in_new_loop(
                fetch_metadata_list(
                    social_group,
                    start_date,
                    end_date,
                    prefer_csv_for_full_range=(num_docs is None and FULL_RANGE_CSV_FIRST),
                ),
                overall_timeout=overall_timeout
            )
        except Exception as e:
            _set_progress(task_id, stage=f"Error: fetch_metadata failed: {repr(e)}", eta_seconds=0)
            return

        print(f"[INFO] Found {len(files)} files for processing", file=sys.stderr)
        if not files:
            _set_progress(task_id, stage="No files found", eta_seconds=0)
            return

        processed_files = files

        if num_docs is None:
            # CSV companion mapping can map many parquet parts to one monthly CSV.
            # Deduplicate by export path to avoid writing the same CSV repeatedly.
            dedup_map: Dict[str, Dict] = {}
            for f in processed_files:
                key = f.get("file_path")
                if key and key not in dedup_map:
                    dedup_map[key] = f
            deduped_files = list(dedup_map.values())
            if len(deduped_files) != len(processed_files):
                print(
                    f"[INFO] Full-range dedupe reduced files: "
                    f"{len(processed_files)} -> {len(deduped_files)}",
                    file=sys.stderr,
                )
            url = zip_originals_from_s3(task_id, deduped_files, social_group, start_date, end_date)
            task_results[task_id] = url
            _set_progress(task_id, stage="Done", eta_seconds=0)
            return

        total_available = sum((f.get("num_rows") or 0) for f in files)
        num_docs = max(0, min(num_docs, total_available))

        quotas = compute_per_file_quotas(files, num_docs)
        files_to_process = files

        approx_zip_units = max(1, num_docs // ZIP_PART_SIZE_ROWS + 1)
        total_units = len(files_to_process) + approx_zip_units + 1 
        _set_progress(task_id, stage="Sampling documents", total_units=total_units)

        # Incremental processing: each worker writes sampled rows to disk, then we zip.
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        zip_filename = f"ISAAC_{social_group}_{start_date}_{end_date}_{timestamp}.zip"
        zip_path = os.path.join(TEMP_DIR, zip_filename)

        completed_units = 0
        total_sampled_rows = 0
        chunk_counter = 0

        def process_one(key: str, k: int) -> Tuple[Optional[str], int]:
            return sample_any_to_temp_csv(key, k, start_date, end_date)

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
            sampled_temp_files: List[str] = []
            current_chunk_name = None
            current_chunk_path = None
            current_chunk_file = None
            current_chunk_writer = None
            current_chunk_rows = 0
            canonical_columns: List[str] = []

            def start_new_chunk() -> None:
                nonlocal chunk_counter, current_chunk_name, current_chunk_path, current_chunk_file
                nonlocal current_chunk_writer, current_chunk_rows
                chunk_counter += 1
                current_chunk_name = f"ISAAC_{social_group}_{start_date}_{end_date}_{chunk_counter}.csv"
                current_chunk_path = os.path.join(TEMP_DIR, f"merged_{chunk_counter}_{uuid.uuid4().hex}.csv")
                current_chunk_file = open(current_chunk_path, "w", encoding="utf-8", newline="")
                current_chunk_writer = csv.DictWriter(
                    current_chunk_file,
                    fieldnames=canonical_columns,
                    extrasaction="ignore"
                )
                current_chunk_writer.writeheader()
                current_chunk_rows = 0

            def flush_chunk_to_zip() -> None:
                nonlocal current_chunk_name, current_chunk_path, current_chunk_file
                nonlocal current_chunk_writer, current_chunk_rows
                if current_chunk_file is None:
                    return
                current_chunk_file.close()
                zipf.write(current_chunk_path, current_chunk_name)
                os.remove(current_chunk_path)
                current_chunk_name = None
                current_chunk_path = None
                current_chunk_file = None
                current_chunk_writer = None
                current_chunk_rows = 0

            file_results = [
                {
                    "key": f["file_path"],
                    "meta_rows": max(0, int(f.get("num_rows") or 0)),
                    "requested": q,
                    "row_count": 0,
                    "tmp_csv_path": None,
                }
                for f, q in zip(files_to_process, quotas)
            ]

            with ThreadPoolExecutor(max_workers=DEFAULT_MAX_WORKERS) as ex:
                futures = {
                    ex.submit(process_one, fr["key"], fr["requested"]): fr
                    for fr in file_results
                }
                for fut in as_completed(futures):
                    fr = futures[fut]
                    tmp_csv_path, row_count = fut.result()
                    if tmp_csv_path and row_count > 0:
                        fr["tmp_csv_path"] = tmp_csv_path
                        fr["row_count"] = row_count
                        total_sampled_rows += row_count
                    else:
                        fr["tmp_csv_path"] = None
                        fr["row_count"] = 0
                        if tmp_csv_path and os.path.exists(tmp_csv_path):
                            os.remove(tmp_csv_path)

                    completed_units += 1
                    _set_progress(
                        task_id,
                        completed_units=completed_units,
                        stage=f"Sampled {total_sampled_rows} rows so far"
                    )
                    _update_eta_from_units(task_id)

            # Top-up pass: if first pass under-returns, retry selected files with larger k.
            # This helps reach exact requested counts when metadata overestimates some files.
            if num_docs and total_sampled_rows < num_docs:
                deficit = num_docs - total_sampled_rows
                _set_progress(task_id, stage=f"Top-up sampling ({total_sampled_rows}/{num_docs})")
                print(
                    f"[INFO] Top-up pass starting: sampled={total_sampled_rows}, "
                    f"requested={num_docs}, deficit={deficit}",
                    file=sys.stderr,
                )

                # Prefer files with larger estimated remaining capacity.
                candidates = sorted(
                    file_results,
                    key=lambda fr: max(0, fr["meta_rows"] - fr["row_count"]),
                    reverse=True
                )

                for fr in candidates:
                    if deficit <= 0:
                        break

                    spare_estimate = max(0, fr["meta_rows"] - fr["row_count"])
                    if spare_estimate <= 0:
                        continue

                    grow_by = min(deficit, spare_estimate)
                    target_k = fr["row_count"] + grow_by
                    new_tmp_csv_path, new_row_count = process_one(fr["key"], target_k)

                    if new_tmp_csv_path and new_row_count > fr["row_count"]:
                        old_tmp_csv_path = fr["tmp_csv_path"]
                        if old_tmp_csv_path and os.path.exists(old_tmp_csv_path):
                            os.remove(old_tmp_csv_path)

                        gained = new_row_count - fr["row_count"]
                        fr["tmp_csv_path"] = new_tmp_csv_path
                        fr["row_count"] = new_row_count
                        total_sampled_rows += gained
                        deficit -= gained
                        print(
                            f"[INFO] Top-up gained={gained} from {fr['key']} "
                            f"(new_row_count={new_row_count}, remaining_deficit={deficit})",
                            file=sys.stderr,
                        )
                    else:
                        if new_tmp_csv_path and os.path.exists(new_tmp_csv_path):
                            os.remove(new_tmp_csv_path)

                    _set_progress(task_id, stage=f"Top-up sampled {total_sampled_rows}/{num_docs} rows")
                print(
                    f"[INFO] Top-up pass finished: final_sampled={total_sampled_rows}, "
                    f"remaining_deficit={max(0, num_docs - total_sampled_rows)}",
                    file=sys.stderr,
                )

            sampled_temp_files = [
                fr["tmp_csv_path"]
                for fr in file_results
                if fr["tmp_csv_path"] and fr["row_count"] > 0
            ]

            # Pass 1: discover canonical schema (union of columns in encounter order).
            for tmp_csv_path in sampled_temp_files:
                with open(tmp_csv_path, "r", encoding="utf-8", newline="") as src:
                    reader = csv.DictReader(src)
                    if not reader.fieldnames:
                        continue
                    for col in reader.fieldnames:
                        if col not in canonical_columns:
                            canonical_columns.append(col)
            if "source_row" not in canonical_columns:
                canonical_columns.append("source_row")

            # Pass 2: stream normalized rows into chunked CSV files.
            if canonical_columns:
                for tmp_csv_path in sampled_temp_files:
                    try:
                        with open(tmp_csv_path, "r", encoding="utf-8", newline="") as src:
                            reader = csv.DictReader(src)
                            if not reader.fieldnames:
                                continue
                            for row in reader:
                                if current_chunk_writer is None:
                                    start_new_chunk()
                                current_chunk_writer.writerow(row)
                                current_chunk_rows += 1
                                if current_chunk_rows >= ZIP_PART_SIZE_ROWS:
                                    flush_chunk_to_zip()
                    finally:
                        if os.path.exists(tmp_csv_path):
                            os.remove(tmp_csv_path)

            # Flush the final partial chunk.
            flush_chunk_to_zip()

        if total_sampled_rows == 0:
            _set_progress(task_id, stage="No data after sampling", eta_seconds=0)
            if os.path.exists(zip_path):
                os.remove(zip_path)
            return

        zip_chunks = chunk_counter
        accurate_total_units = len(files) + zip_chunks + 1
        _set_progress(task_id, stage="Zipping files",
                      total_units=accurate_total_units, completed_units=completed_units)

        _set_progress(task_id, stage="Uploading to Supabase Storage")
        object_name = f"samples/{zip_filename}"
        file_size = os.path.getsize(zip_path)
        _set_progress(task_id, eta_seconds=(file_size / ASSUMED_UPLOAD_BPS))
        upload_progress_cb = _make_upload_progress_callback(task_id, file_size)

        presigned = upload_file_and_get_presigned(
            zip_path,
            object_name,
            expires_in=3600,
            progress_callback=upload_progress_cb,
        )

        completed_units += 1
        _set_progress(task_id, completed_units=completed_units, eta_seconds=0)
        task_results[task_id] = presigned
        _set_progress(task_id, stage="Done", eta_seconds=0)
        
        # Clean up old tasks after successful completion
        cleanup_old_tasks()

    except Exception as e:
        print(f"[ERROR] background_sampling crashed: {repr(e)}")
        traceback.print_exc()
        _set_progress(task_id, stage=f"Error: {str(e)}", eta_seconds=0)


def run_coro_in_new_loop(coro, overall_timeout: int):
    result_box = {}
    error_box = {}
    done_evt = threading.Event()

    def _runner():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(asyncio.wait_for(coro, timeout=overall_timeout))
                result_box["value"] = result
            finally:
                loop.stop()
                loop.close()
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

@app.get("/")
def home():
    return {"message": "FastAPI is running"}

@app.post("/report_issue")
async def report_issue(data: IssueReport):
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
            password=SMTP_PASSWORD
        )
        return {"message": "Issue reported successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send email: {str(e)}")

@app.post("/sample")
async def get_sampled_data(payload: dict, background_tasks: BackgroundTasks):
    social_group = payload.get("social_group")
    start_date = payload.get("start_date") 
    end_date = payload.get("end_date")    
    num_docs = payload.get("num_docs")  

    if not social_group or not start_date or not end_date:
        raise HTTPException(status_code=400, detail="Missing required fields")

    # Validate date format and range
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
    
    # Validate num_docs if provided
    if num_docs is not None:
        try:
            num_docs = int(num_docs)
            if num_docs < 0:
                raise HTTPException(status_code=400, detail="num_docs must be non-negative")
            if num_docs > 100_000_000:  # Reasonable upper limit
                raise HTTPException(status_code=400, detail="num_docs exceeds maximum allowed (100,000,000)")
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="num_docs must be a valid integer")

    # Clean up old tasks periodically (every 10th request)
    cleanup_old_tasks()
    
    task_id = str(uuid.uuid4())
    with _progress_lock:
        task_meta[task_id] = {"start_time": now(), "total_units": 0, "completed_units": 0}
        task_progress[task_id] = {"percent": 0.0, "eta_seconds": None, "eta_human": None}
        task_stage[task_id] = "Queued"

    background_tasks.add_task(background_sampling, task_id, social_group, start_date, end_date, num_docs)
    return {"task_id": task_id}

@app.get("/progress/{task_id}")
async def get_progress(task_id: str):
    prog = task_progress.get(task_id, {})
    return {
        "stage": task_stage.get(task_id, "Initializing..."),
        "percent": prog.get("percent"),
        "eta_seconds": prog.get("eta_seconds"),
        "eta_human": prog.get("eta_human"),
        "download_link": task_results.get(task_id)
    }

@app.get("/dataset-stats")
async def get_dataset_stats():
    """Get comprehensive dataset statistics for 2007-2023."""
    try:
        conn = await asyncpg.connect(SUPABASE_DB_URL)
        
        groups_query = "SELECT DISTINCT social_group FROM metadata ORDER BY social_group"
        groups = await conn.fetch(groups_query)
        social_groups = [row['social_group'] for row in groups]
        
        all_results = {}
        
        for group in social_groups:
            year_query = """
            SELECT 
                EXTRACT(YEAR FROM date) as year,
                COUNT(*) as file_count,
                SUM(num_rows) as total_rows,
                MIN(num_rows) as min_rows,
                MAX(num_rows) as max_rows,
                AVG(num_rows) as avg_rows
            FROM metadata 
            WHERE social_group = $1
            GROUP BY EXTRACT(YEAR FROM date)
            ORDER BY year
            """
            
            year_records = await conn.fetch(year_query, group)
            
            overall_query = """
            SELECT 
                COUNT(*) as total_files,
                SUM(num_rows) as total_rows,
                MIN(date) as earliest_date,
                MAX(date) as latest_date
            FROM metadata 
            WHERE social_group = $1
            """
            
            overall_record = await conn.fetchrow(overall_query, group)
            
            year_data = []
            total_files = 0
            total_rows = 0
            years = []
            
            for record in year_records:
                year = int(record['year'])
                file_count = record['file_count']
                rows = record['total_rows']
                
                years.append(year)
                total_files += file_count
                total_rows += rows
                
                year_data.append({
                    'year': year,
                    'files': file_count,
                    'rows': rows,
                    'min_rows_per_file': record['min_rows'],
                    'max_rows_per_file': record['max_rows'],
                    'avg_rows_per_file': float(record['avg_rows'])
                })
            
            file_size_query = """
            SELECT 
                num_rows,
                COUNT(*) as file_count
            FROM metadata 
            WHERE social_group = $1
            GROUP BY num_rows
            ORDER BY num_rows
            """
            
            size_records = await conn.fetch(file_size_query, group)
            file_sizes = [record['num_rows'] for record in size_records]
            
            result = {
                'social_group': group,
                'total_files': total_files,
                'total_rows': total_rows,
                'year_range': f"{min(years)}-{max(years)}" if years else "No data",
                'earliest_date': str(overall_record['earliest_date']) if overall_record['earliest_date'] else None,
                'latest_date': str(overall_record['latest_date']) if overall_record['latest_date'] else None,
                'file_size_range': f"{min(file_sizes)}-{max(file_sizes)}" if file_sizes else "No data",
                'years_covered': len(years),
                'year_by_year': year_data,
                'file_size_distribution': [
                    {'rows_per_file': record['num_rows'], 'file_count': record['file_count']} 
                    for record in size_records
                ]
            }
            
            all_results[group] = result
        
        await conn.close()
        
        total_files = sum(data['total_files'] for data in all_results.values())
        total_rows = sum(data['total_rows'] for data in all_results.values())
        
        return {
            "summary": {
                "total_files": total_files,
                "total_rows": total_rows,
                "social_groups": len(all_results),
                "year_coverage": "2007-2023",
                "analysis_timestamp": datetime.now().isoformat()
            },
            "by_social_group": all_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dataset stats: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)