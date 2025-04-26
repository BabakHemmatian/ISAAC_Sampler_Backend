from fastapi import FastAPI, HTTPException, BackgroundTasks
import pandas as pd
import random
import zipfile
import os
import uuid
from supabase import create_client, Client
import asyncpg
import nest_asyncio
import uvicorn
import boto3
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from email.message import EmailMessage
import aiosmtplib
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
import asyncio
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ssd9.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

nest_asyncio.apply()

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
ISSUE_RECEIVER_EMAIL = os.getenv("ISSUE_RECEIVER_EMAIL")

class IssueReport(BaseModel):
    email: EmailStr
    description: str

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")

sb_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

SUPABASE_S3_ENDPOINT = os.getenv("SUPABASE_S3_ENDPOINT")
SUPABASE_S3_REGION = os.getenv("SUPABASE_S3_REGION")
SUPABASE_S3_ACCESS_KEY_ID = os.getenv("SUPABASE_S3_ACCESS_KEY_ID")
SUPABASE_S3_SECRET_ACCESS_KEY = os.getenv("SUPABASE_S3_SECRET_ACCESS_KEY")
SUPABASE_BUCKET_NAME = os.getenv("SUPABASE_BUCKET_NAME")

s3_client = boto3.client(
    "s3",
    endpoint_url=SUPABASE_S3_ENDPOINT,
    aws_access_key_id=SUPABASE_S3_ACCESS_KEY_ID,
    aws_secret_access_key=SUPABASE_S3_SECRET_ACCESS_KEY,
    region_name=SUPABASE_S3_REGION
)

TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)

task_progress = {}
task_stage = {}
task_results = {}

async def fetch_metadata(social_group: str, start_date: int, end_date: int):
    conn = await asyncpg.connect(SUPABASE_DB_URL)
    query = """
    SELECT file_path, num_rows 
    FROM metadata 
    WHERE social_group = $1 
    AND date BETWEEN 
    TO_DATE($2 || '-01', 'YYYY-MM-DD') 
    AND (DATE_TRUNC('MONTH', TO_DATE($3 || '-01', 'YYYY-MM-DD')) + INTERVAL '1 MONTH' - INTERVAL '1 DAY')
    """
    result = await conn.fetch(query, social_group, start_date, end_date)
    await conn.close()
    return [{"file_path": row["file_path"], "num_rows": row["num_rows"]} for row in result]

def sample_csv_fast(filepath, sample_size):
    sampled_rows = []
    chunk_size = 10000
    reader = pd.read_csv(filepath, chunksize=chunk_size)
    for chunk in reader:
        if len(sampled_rows) < sample_size:
            sampled_chunk = chunk.sample(min(sample_size, len(chunk)), random_state=random.randint(1, 10000))
            sampled_rows.append(sampled_chunk)
            if sum(len(df) for df in sampled_rows) >= sample_size:
                break
    return pd.concat(sampled_rows).head(sample_size)

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

def background_sampling(task_id: str, social_group: str, start_date: str, end_date: str, num_docs: int = None):
    try:
        task_stage[task_id] = "Fetching metadata"

        filtered_files = asyncio.run(fetch_metadata(social_group, start_date, end_date))
        if not filtered_files:
            task_stage[task_id] = "No files found"
            return

        total_available = sum(file["num_rows"] for file in filtered_files)
        num_docs = num_docs if num_docs else total_available
        docs_per_file = max(1, num_docs // len(filtered_files))

        task_stage[task_id] = "Sampling documents"

        def process_file(file):
            return sample_csv_fast(file['file_path'], docs_per_file)

        with ThreadPoolExecutor(max_workers=4) as executor:
            sampled_data = list(executor.map(process_file, filtered_files))

        final_df = pd.concat(sampled_data)

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        zip_filename = f"ISSAC_{social_group}_{start_date}_{end_date}_{timestamp}.zip"
        zip_path = os.path.join(TEMP_DIR, zip_filename)

        task_stage[task_id] = "Zipping files"
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for i in range(0, len(final_df), 10):
                chunk_df = final_df.iloc[i:i + 10]
                file_number = i // 10 + 1
                chunk_filename = f"ISSAC_{social_group}_{start_date}_{end_date}_{file_number}.csv"
                chunk_path = os.path.join(TEMP_DIR, chunk_filename)
                chunk_df.to_csv(chunk_path, index=False)
                zipf.write(chunk_path, chunk_filename)
                os.remove(chunk_path)

        object_name = f"samples/{zip_filename}"
        task_stage[task_id] = "Uploading to Supabase Storage"

        s3_client.upload_file(zip_path, SUPABASE_BUCKET_NAME, object_name)
        presigned_url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": SUPABASE_BUCKET_NAME, "Key": object_name},
            ExpiresIn=3600
        )
        modified_url = presigned_url.replace("/s3/", "/object/public/").split("?")[0]

        task_results[task_id] = modified_url
        task_stage[task_id] = "Done"

    except Exception as e:
        task_stage[task_id] = f"Error: {str(e)}"

@app.post("/sample")
async def get_sampled_data(payload: dict, background_tasks: BackgroundTasks):
    social_group = payload.get("social_group")
    start_date = payload.get("start_date")
    end_date = payload.get("end_date")
    num_docs = payload.get("num_docs")

    if not social_group or not start_date or not end_date:
        raise HTTPException(status_code=400, detail="Missing required fields")

    # Range validation
    if int(start_date.split("-")[0]) < 2007 or int(end_date.split("-")[0]) > 2023:
        raise HTTPException(status_code=400, detail="Date range must be between 2007 and 2023")

    if start_date > end_date:
        raise HTTPException(status_code=400, detail="Start date must be before end date")

    task_id = str(uuid.uuid4())
    background_tasks.add_task(background_sampling, task_id, social_group, start_date, end_date, num_docs)

    return {"task_id": task_id}

@app.get("/progress/{task_id}")
async def get_progress(task_id: str):
    return {
        "stage": task_stage.get(task_id, "Initializing..."),
        "download_link": task_results.get(task_id)
    }

# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000)
