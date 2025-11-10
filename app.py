from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from fastapi.responses import JSONResponse

import os
import sys
import uuid
import zipfile
import io
import time
import threading
import asyncio
import gzip  
import traceback 
import gc
from datetime import datetime
from typing import Optional, List, Dict
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

# Log CORS configuration for debugging (only in development)
if os.getenv("ENVIRONMENT", "production") != "production":
    print(f"CORS Configuration - Allowed Origins: {ALLOWED_ORIGINS}")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],
    allow_headers=["Content-Type", "Authorization", "Accept", "X-Requested-With"],
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

# Memory-efficient mode configuration
# Set MEMORY_EFFICIENT_MODE=true to enable memory-efficient processing (slower but uses less memory)
MEMORY_EFFICIENT_MODE = os.getenv("MEMORY_EFFICIENT_MODE", "false").lower() == "true"

# Configure workers based on mode
if MEMORY_EFFICIENT_MODE:
    DEFAULT_MAX_WORKERS = 2  # Reduced workers for memory efficiency
else:
    DEFAULT_MAX_WORKERS = min(8, os.cpu_count() or 4)  # Default: up to 8 workers

# Task cleanup configuration
TASK_CLEANUP_MAX_AGE_SECONDS = int(os.getenv("TASK_CLEANUP_MAX_AGE_SECONDS", "86400"))  # 24 hours default

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
            task_stage[task_id] = stage
        meta = task_meta.setdefault(task_id, {})
        prog = task_progress.setdefault(task_id, {"percent": 0.0, "eta_seconds": None, "eta_human": None})
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
        cu = meta.get("completed_units", 0)
        tu = meta.get("total_units", 0)
    if not start_time or cu <= 0 or tu <= 0 or cu > tu:
        return
    elapsed = now() - start_time
    avg_per_unit = elapsed / cu
    remaining_units = tu - cu
    eta = remaining_units * avg_per_unit
    _set_progress(task_id, eta_seconds=eta)

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

# CSV reading configuration - use low_memory mode in memory-efficient mode
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
    low_memory=MEMORY_EFFICIENT_MODE,  # Use low_memory mode when memory-efficient mode is enabled
)


async def fetch_metadata(social_group: str, start_month: str, end_month: str):
    conn = await asyncpg.connect(SUPABASE_DB_URL)
    try:
        query = """
        SELECT file_path, num_rows
        FROM metadata
        WHERE social_group = $1
          AND date >= (($2 || '-01')::date)
          AND date <  (($3 || '-01')::date + INTERVAL '1 MONTH')
        ORDER BY date ASC;
        """
        async with conn.transaction():
            async for record in conn.cursor(query, social_group, start_month, end_month):
                yield {"file_path": record["file_path"], "num_rows": record["num_rows"]}
    finally:
        try:
            await asyncio.sleep(0) 
            await conn.close()
        except Exception as e:
            print(f"[ERROR] Connection close failed: {e}", file=sys.stderr)

async def fetch_metadata_list(social_group: str, start_month: str, end_month: str):
    items = []
    async for item in fetch_metadata(social_group, start_month, end_month):
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

def stream_object_to_zip(zipf: zipfile.ZipFile, key: str, arcname: Optional[str] = None, chunk_size: int = 1024*1024):
    if arcname is None:
        arcname = os.path.basename(key)
    
    try:
        obj = s3_client.get_object(Bucket=SUPABASE_BUCKET_NAME, Key=key)
        with zipf.open(arcname, "w") as zf:
            for chunk in obj["Body"].iter_chunks(chunk_size=chunk_size):
                if chunk:
                    zf.write(chunk)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to stream {key}: {e}", file=sys.stderr)
        return False

def upload_file_and_get_presigned(local_path: str, dest_key: str, expires_in: int = 3600) -> str:
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
        Config=transfer_config
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


def sample_csv_reservoir_from_s3(key: str, k: int, chunksize: Optional[int] = None) -> pd.DataFrame:
    # Use smaller chunk size in memory-efficient mode
    if chunksize is None:
        chunksize = 50_000 if MEMORY_EFFICIENT_MODE else 200_000
    if k <= 0:
        return pd.DataFrame()

    rng = np.random.default_rng()
    reservoir = None
    seen = 0

    with s3_open_csv_stream(key) as csv_stream:
        for chunk in pd.read_csv(csv_stream, chunksize=chunksize, **READ_CSV_KW):
            if reservoir is None:
                take = min(k, len(chunk))
                if take == 0:
                    continue
                reservoir = chunk.iloc[:take].copy()
                seen += len(chunk)
                continue

            for idx in range(len(chunk)):
                seen += 1
                j = rng.integers(0, seen)
                if j < k:
                    replace_at = rng.integers(0, k)
                    reservoir.iloc[replace_at] = chunk.iloc[idx]

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

            if reservoir is None:
                take = min(k, len(df))
                reservoir = df.iloc[:take].copy()
                seen += len(df)
                continue

            for idx in range(len(df)):
                seen += 1
                j = rng.integers(0, seen)
                if j < k:
                    replace_at = rng.integers(0, k)
                    reservoir.iloc[replace_at] = df.iloc[idx]

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


def compute_per_file_quotas(files: List[Dict], num_docs: int) -> List[int]:
    total = sum(max(0, f.get("num_rows", 0) or 0) for f in files)
    if total <= 0:
        base = num_docs // len(files)
        quotas = [base] * len(files)
        quotas[-1] += num_docs - base * len(files)
        return quotas
    quotas = []
    assigned = 0
    for i, f in enumerate(files):
        if i == len(files) - 1:
            q = num_docs - assigned
        else:
            share = (f.get("num_rows", 0) or 0) / total
            q = int(round(num_docs * share))
            assigned += q
        quotas.append(max(0, q))
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
    zip_filename = f"ISSAC_{social_group}_{start_date}_{end_date}_{timestamp}.zip"
    zip_path = os.path.join(TEMP_DIR, zip_filename)

    total_files = len(files)
    _set_progress(task_id, total_units=total_files + 1, completed_units=0)

    # Use a lock for thread-safe zip file operations
    zip_lock = threading.Lock()
    completed = 0
    successful = 0
    
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        def process_file_with_lock(file_info: Dict) -> bool:
            """Process a single file and update progress in thread-safe manner."""
            nonlocal completed, successful
            try:
                # Process the file (parquet conversion happens outside lock)
                key = file_info["file_path"]
                arcname = os.path.basename(key)
                
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
                                table = pf.read()
                                df = table.to_pandas()
                                if df.empty:
                                    return False
                                
                                from io import StringIO
                                csv_buffer = StringIO()
                                df.to_csv(csv_buffer, index=False)
                                csv_content = csv_buffer.getvalue().encode('utf-8')
                                
                                # Add to zip file (thread-safe operation)
                                with zip_lock:
                                    with zipf.open(csv_name, "w") as zf:
                                        zf.write(csv_content)
                                return True
                            
                            # Stream row groups for memory efficiency
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
                        
                        # Add to zip file (thread-safe operation)
                        with zip_lock:
                            with zipf.open(csv_name, "w") as zf:
                                for chunk in csv_chunks:
                                    zf.write(chunk)
                        
                        return True
                    except Exception as e:
                        print(f"[ERROR] Failed to convert parquet {key}: {e}", file=sys.stderr)
                        traceback.print_exc(file=sys.stderr)
                        return False
                else:
                    # CSV files - stream directly
                    with zip_lock:
                        return stream_object_to_zip(zipf, key, arcname)
                        
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
                        _set_progress(task_id, stage=f"Processed {completed}/{total_files} files ({successful} successful)")
        
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

    _set_progress(task_id, stage="Uploading to Supabase Storage")
    object_name = f"samples/{zip_filename}"
    file_size = os.path.getsize(zip_path)
    
    estimated_upload_time = max(1, file_size / (ASSUMED_UPLOAD_BPS * 2))
    _set_progress(task_id, eta_seconds=estimated_upload_time)

    presigned = upload_file_and_get_presigned(zip_path, object_name, expires_in=3600)

    _set_progress(task_id, completed_units=total_files + 1, eta_seconds=0)
    print(f"[INFO] Fast path completed: {successful}/{total_files} files processed successfully", file=sys.stderr)
    
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
                fetch_metadata_list(social_group, start_date, end_date),
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
            url = zip_originals_from_s3(task_id, processed_files, social_group, start_date, end_date)
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

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        zip_filename = f"ISSAC_{social_group}_{start_date}_{end_date}_{timestamp}.zip"
        zip_path = os.path.join(TEMP_DIR, zip_filename)

        def process_one(key: str, k: int) -> Optional[pd.DataFrame]:
            return sample_any_from_s3(key, k, start_date, end_date)

        if MEMORY_EFFICIENT_MODE:
            # Memory-efficient mode: Process incrementally, write to zip as we go
            completed_units = 0
            total_sampled_rows = 0
            chunk_counter = 0

            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
                with ThreadPoolExecutor(max_workers=DEFAULT_MAX_WORKERS) as ex:
                    futures = [ex.submit(process_one, f["file_path"], q) for f, q in zip(files_to_process, quotas)]
                    for fut in as_completed(futures):
                        df = fut.result()
                        if df is not None and not df.empty:
                            # Process DataFrame in chunks and write incrementally
                            for i in range(0, len(df), ZIP_PART_SIZE_ROWS):
                                part = df.iloc[i:i + ZIP_PART_SIZE_ROWS]
                                if len(part) == 0:
                                    continue
                                
                                # Write to zip file incrementally
                                chunk_counter += 1
                                name = f"ISSAC_{social_group}_{start_date}_{end_date}_{chunk_counter}.csv"
                                tmp = os.path.join(TEMP_DIR, name)
                                part.to_csv(tmp, index=False)
                                zipf.write(tmp, name)
                                os.remove(tmp)
                                
                                total_sampled_rows += len(part)
                                
                                # Update progress
                                completed_units += 1
                                _set_progress(task_id, completed_units=completed_units, 
                                            stage=f"Processed {total_sampled_rows} rows")
                                _update_eta_from_units(task_id)
                            
                            # Explicitly delete DataFrame to free memory
                            del df
                            gc.collect()
                        
                        completed_units += 1
                        _set_progress(task_id, completed_units=completed_units)
                        _update_eta_from_units(task_id)

            if total_sampled_rows == 0:
                _set_progress(task_id, stage="No data after sampling", eta_seconds=0)
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                return

            zip_chunks = chunk_counter
            accurate_total_units = len(files) + zip_chunks + 1
            _set_progress(task_id, stage="Zipping files",
                          total_units=accurate_total_units, completed_units=completed_units)
        else:
            # Default mode: Accumulate all DataFrames, then process
            completed_units = 0
            sampled_parts: List[pd.DataFrame] = []

            with ThreadPoolExecutor(max_workers=DEFAULT_MAX_WORKERS) as ex:
                futures = [ex.submit(process_one, f["file_path"], q) for f, q in zip(files_to_process, quotas)]
                for fut in as_completed(futures):
                    df = fut.result()
                    if df is not None and not df.empty:
                        sampled_parts.append(df)
                    completed_units += 1
                    _set_progress(task_id, completed_units=completed_units)
                    _update_eta_from_units(task_id)

            if not sampled_parts:
                _set_progress(task_id, stage="No data after sampling", eta_seconds=0)
                return

            final_df = pd.concat(sampled_parts, ignore_index=True)

            zip_chunks = max(1, (len(final_df) + ZIP_PART_SIZE_ROWS - 1) // ZIP_PART_SIZE_ROWS)
            accurate_total_units = len(files) + zip_chunks + 1
            _set_progress(task_id, stage="Zipping files",
                          total_units=accurate_total_units, completed_units=completed_units)

            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
                for i in range(0, len(final_df), ZIP_PART_SIZE_ROWS):
                    part = final_df.iloc[i:i + ZIP_PART_SIZE_ROWS]
                    name = f"ISSAC_{social_group}_{start_date}_{end_date}_{(i//ZIP_PART_SIZE_ROWS)+1}.csv"
                    tmp = os.path.join(TEMP_DIR, name)
                    part.to_csv(tmp, index=False)
                    zipf.write(tmp, name)
                    os.remove(tmp)

                    completed_units += 1
                    _set_progress(task_id, completed_units=completed_units)
                    _update_eta_from_units(task_id)

        _set_progress(task_id, stage="Uploading to Supabase Storage")
        object_name = f"samples/{zip_filename}"
        file_size = os.path.getsize(zip_path)
        _set_progress(task_id, eta_seconds=(file_size / ASSUMED_UPLOAD_BPS))

        presigned = upload_file_and_get_presigned(zip_path, object_name, expires_in=3600)

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