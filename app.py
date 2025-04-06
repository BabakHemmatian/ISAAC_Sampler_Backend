from fastapi import FastAPI, HTTPException
import pandas as pd
import random
import zipfile
import os
import uuid
from supabase import create_client, Client
import asyncpg
from io import BytesIO
import nest_asyncio
import uvicorn
import boto3
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from email.message import EmailMessage
import aiosmtplib
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
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

# Pydantic model for request
class IssueReport(BaseModel):
    email: EmailStr
    description: str


# Supabase credentials
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")

# Initialize Supabase client
sb_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Supabase S3 Credentials
SUPABASE_S3_ENDPOINT = os.getenv("SUPABASE_S3_ENDPOINT")
SUPABASE_S3_REGION = os.getenv("SUPABASE_S3_REGION")
SUPABASE_S3_ACCESS_KEY_ID = os.getenv("SUPABASE_S3_ACCESS_KEY_ID")
SUPABASE_S3_SECRET_ACCESS_KEY = os.getenv("SUPABASE_S3_SECRET_ACCESS_KEY")
SUPABASE_BUCKET_NAME = os.getenv("SUPABASE_BUCKET_NAME")

# Initialize S3 client for Supabase
s3_client = boto3.client(
    "s3",
    endpoint_url=SUPABASE_S3_ENDPOINT, 
    aws_access_key_id=SUPABASE_S3_ACCESS_KEY_ID,
    aws_secret_access_key=SUPABASE_S3_SECRET_ACCESS_KEY,
    region_name=SUPABASE_S3_REGION 
)

# Temporary storage path
TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)

# Establish async connection to Supabase Postgres
async def connect_to_db():
    return await asyncpg.connect(SUPABASE_DB_URL)

async def fetch_metadata(social_group: str, start_date: int, end_date: int):
    """Query the metadata table in Supabase Postgres."""
    conn = await connect_to_db()
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

@app.get("/")
def home():
    return {"message": "FastAPI is running"}

@app.post("/report_issue")
async def report_issue(data: IssueReport):
    # Compose the email
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
async def get_sampled_data(payload: dict):
    """
    Endpoint to fetch a random sample of rows from the CSV files based on user request.
    Returns a ZIP file containing a sampled CSV.
    """
    social_group = payload.get("social_group")
    start_date = payload.get("start_date")
    end_date = payload.get("end_date")
    num_docs = payload.get("num_docs")

    if not social_group or not start_date or not end_date or not num_docs:
        raise HTTPException(status_code=400, detail="Missing required fields")

    # Fetch metadata from Supabase
    filtered_files = await fetch_metadata(social_group, start_date, end_date)

    if not filtered_files:
        raise HTTPException(status_code=404, detail="No matching files for the selected date range.")

    # Determine the number of docs to sample per file
    total_available = sum(file["num_rows"] for file in filtered_files)
    num_docs = min(num_docs, total_available)  # Adjust to available docs
    docs_per_file = max(1, num_docs // len(filtered_files))

    sampled_data = []

    for file in filtered_files:
        file_url = file['file_path'] 
        df = pd.read_csv(file_url)
        sampled_df = df.sample(min(docs_per_file, len(df)), random_state=42)
        sampled_data.append(sampled_df)

    final_df = pd.concat(sampled_data)

        # Create ZIP file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    zip_filename = f"ISSAC_{social_group}_{start_date}_{end_date}_{timestamp}.zip"
    zip_path = os.path.join(TEMP_DIR, zip_filename)

    with zipfile.ZipFile(zip_path, "w") as zipf:
        for i in range(0, len(final_df), 10):
            chunk_df = final_df.iloc[i:i+10]
            file_number = i // 10 + 1
            chunk_filename = f"ISSAC_{social_group}_{start_date}_{end_date}_{file_number}.csv"
            chunk_path = os.path.join(TEMP_DIR, chunk_filename)
            chunk_df.to_csv(chunk_path, index=False)
            zipf.write(chunk_path, chunk_filename)
            os.remove(chunk_path) 

    # Upload ZIP to Supabase Storage
    object_name = f"samples/{zip_filename}"

    # Generate Pre-signed URL
    try:
        s3_client.upload_file(zip_path, SUPABASE_BUCKET_NAME, object_name)
        presigned_url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": SUPABASE_BUCKET_NAME, "Key": object_name},
            ExpiresIn=3600  # 1 hour validity
        )
        modified_url = presigned_url.replace(
            "/s3/", "/object/public/"
        ).split("?")[0]

        print(f"Pre-Signed URL: {modified_url}")
    except Exception as e:
        print("Failed to generate Pre-Signed URL:", e)
        raise HTTPException(status_code=500, detail="Failed to generate pre-signed URL")

    return {
        "message": "File processed successfully",
        "download_link": modified_url
    }

# uvicorn.run(app, host="127.0.0.1", port=8000)

