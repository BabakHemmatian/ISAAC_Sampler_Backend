from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from fastapi.responses import JSONResponse

import os
import uuid
import zipfile
import io
import time
import threading
import asyncio
import gzip  
import traceback 
from datetime import datetime
from typing import Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import asyncpg
import boto3
import aiosmtplib
from email.message import EmailMessage
from dotenv import load_dotenv

import pyarrow as pa
import pyarrow.parquet as pq
import s3fs

# App & Config

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Email for /report_issue
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT") or "587")
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
ISSUE_RECEIVER_EMAIL = os.getenv("ISSUE_RECEIVER_EMAIL")

# Supabase Postgres (asyncpg)
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")

# Supabase Storage via S3-compatible API
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

# Seekable FS for Parquet range reads
s3_fs = s3fs.S3FileSystem(
    key=SUPABASE_S3_ACCESS_KEY_ID,
    secret=SUPABASE_S3_SECRET_ACCESS_KEY,
    client_kwargs={"endpoint_url": SUPABASE_S3_ENDPOINT, "region_name": SUPABASE_S3_REGION},
)

TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)


# Progress / ETA tracking


task_progress: Dict[str, Dict] = {} 
task_stage: Dict[str, str] = {}
task_results: Dict[str, str] = {}  
task_meta: Dict[str, Dict] = {}     
_progress_lock = threading.Lock()

ASSUMED_UPLOAD_BPS = 10 * 1024 * 1024 
ZIP_PART_SIZE_ROWS = 100_000          
DEFAULT_MAX_WORKERS = min(8, os.cpu_count() or 4)

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

# Models

class IssueReport(BaseModel):
    email: EmailStr
    description: str

# Tailored CSV parsing for your sample schema
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

# Supabase Postgres: fetch metadata rows

async def fetch_metadata(social_group: str, start_month: str, end_month: str):
    """
    start_month, end_month: 'YYYY-MM'
    metadata.date is a DATE (first of month). We compare using >= start and < next-month(end).
    """
    import sys
    print(f"[DEBUG] SUPABASE_DB_URL: {SUPABASE_DB_URL}", file=sys.stderr)
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
        print(f"[DEBUG] Running metadata query for group={social_group}, start={start_month}, end={end_month}", file=sys.stderr)
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

# S3 / Supabase Storage helpers (CSV & CSV.GZ)

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
    obj = s3_client.get_object(Bucket=SUPABASE_BUCKET_NAME, Key=key)
    with zipf.open(arcname, "w") as zf:
        for chunk in obj["Body"].iter_chunks(chunk_size=chunk_size):
            if chunk:
                zf.write(chunk)

def upload_file_and_get_presigned(local_path: str, dest_key: str, expires_in: int = 3600) -> str:
    s3_client.upload_file(local_path, SUPABASE_BUCKET_NAME, dest_key)
    return s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": SUPABASE_BUCKET_NAME, "Key": dest_key},
        ExpiresIn=expires_in
    )

# Sampling helpers (CSV & Parquet)

def sample_csv_reservoir_from_s3(key: str, k: int, chunksize: int = 200_000) -> pd.DataFrame:
    if k <= 0:
        return pd.DataFrame()

    rng = np.random.default_rng()
    reservoir = None
    seen = 0

    with s3_open_csv_stream(key) as stream:
        for chunk in pd.read_csv(stream, chunksize=chunksize, **READ_CSV_KW):
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

# Parquet reservoir sampler (row-group aware)
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

    def _norm_month(s):
        if s and len(s) == 7: 
            return s + "-01T00:00:00"
        return s

    start_ts = _norm_month(start_ts)
    end_ts   = _norm_month(end_ts)

    path = f"{SUPABASE_BUCKET_NAME}/{key}" if not key.startswith(f"{SUPABASE_BUCKET_NAME}/") else key

    with s3_fs.open(path, "rb") as f:
        pf = pq.ParquetFile(f)

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

# Worker: fast path (no num_docs) OR sampling path

def zip_originals_from_s3(task_id: str, files: List[Dict], social_group: str,
                          start_date: str, end_date: str) -> str:
    """
    Bundle original objects from Supabase Storage into a single ZIP and upload.
    """
    _set_progress(task_id, stage="Bundling original files")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    zip_filename = f"ISSAC_{social_group}_{start_date}_{end_date}_{timestamp}.zip"
    zip_path = os.path.join(TEMP_DIR, zip_filename)

    total_units = len(files) + 1
    completed = 0
    _set_progress(task_id, total_units=total_units, completed_units=completed)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        for f in files:
            key = f["file_path"]
            arcname = os.path.basename(key)
            try:
                stream_object_to_zip(zipf, key, arcname)
            except Exception:
                pass
            completed += 1
            _set_progress(task_id, completed_units=completed)
            _update_eta_from_units(task_id)

    # Upload + ETA
    _set_progress(task_id, stage="Uploading to Supabase Storage")
    object_name = f"samples/{zip_filename}"
    file_size = os.path.getsize(zip_path)
    _set_progress(task_id, eta_seconds=(file_size / ASSUMED_UPLOAD_BPS))

    presigned = upload_file_and_get_presigned(zip_path, object_name, expires_in=3600)

    completed += 1
    _set_progress(task_id, completed_units=completed, eta_seconds=0)
    return presigned

def background_sampling(task_id: str, social_group: str, start_date: str,
                        end_date: str, num_docs: Optional[int] = None):
    import sys
    print(f"[DEBUG] Starting background_sampling for group={social_group}, start={start_date}, end={end_date}, num_docs={num_docs}", file=sys.stderr)
    try:
        # Fetch metadata with retries + per-attempt stage updates
        max_tries = int(os.getenv("DB_RETRIES", "3"))
        overall_timeout = int(os.getenv("DB_OVERALL_TIMEOUT", "300"))

        files = None
        last_err = None
        for attempt in range(1, max_tries + 1):
            _set_progress(task_id, stage=f"Fetching metadata (attempt {attempt}/{max_tries})")
            try:
                files = run_coro_in_new_loop(
                    fetch_metadata_list(social_group, start_date, end_date),
                    overall_timeout=overall_timeout
                )
                break
            except Exception as e:
                last_err = e
                print(f"[ERROR] fetch_metadata failed (attempt {attempt}): {repr(e)}", file=sys.stderr)
                traceback.print_exc()
                delay = (1.5 ** attempt)
                _set_progress(
                    task_id,
                    stage=f"Fetching metadata failed (attempt {attempt}/{max_tries}): {repr(e)}. Retrying…"
                )
                time.sleep(delay)

        if files is None:
            _set_progress(task_id, stage=f"Error: fetch_metadata failed: {repr(last_err)}", eta_seconds=0)
            return

        print(f"[DEBUG] Files from metadata: {files}", file=sys.stderr)
        if not files:
            _set_progress(task_id, stage="No files found", eta_seconds=0)
            return

        # FAST PATH: return originals if num_docs is missing
        if num_docs is None:
            url = zip_originals_from_s3(task_id, files, social_group, start_date, end_date)
            task_results[task_id] = url
            _set_progress(task_id, stage="Done", eta_seconds=0)
            return

        # Sampling path
        total_available = sum((f.get("num_rows") or 0) for f in files)
        num_docs = max(0, min(num_docs, total_available))

        quotas = compute_per_file_quotas(files, num_docs)

        approx_zip_units = max(1, num_docs // ZIP_PART_SIZE_ROWS + 1)
        total_units = len(files) + approx_zip_units + 1 
        _set_progress(task_id, stage="Sampling documents", total_units=total_units)

        completed_units = 0
        sampled_parts: List[pd.DataFrame] = []

        def process_one(key: str, k: int) -> Optional[pd.DataFrame]:
            return sample_any_from_s3(key, k, start_date, end_date)

        with ThreadPoolExecutor(max_workers=DEFAULT_MAX_WORKERS) as ex:
            futures = [ex.submit(process_one, f["file_path"], q) for f, q in zip(files, quotas)]
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

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        zip_filename = f"ISSAC_{social_group}_{start_date}_{end_date}_{timestamp}.zip"
        zip_path = os.path.join(TEMP_DIR, zip_filename)

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

# Routes

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

    if int(start_date.split("-")[0]) < 2007 or int(end_date.split("-")[0]) > 2023:
        raise HTTPException(status_code=400, detail="Date range must be between 2007 and 2023")
    if start_date > end_date:
        raise HTTPException(status_code=400, detail="Start date must be before end date")

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
