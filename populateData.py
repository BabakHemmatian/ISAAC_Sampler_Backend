
import os
import pandas as pd
from supabase import create_client, Client
from datetime import datetime
import boto3
import asyncpg
import asyncio
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")

SUPABASE_S3_ENDPOINT = os.getenv("SUPABASE_S3_ENDPOINT")
SUPABASE_S3_REGION = os.getenv("SUPABASE_S3_REGION")
SUPABASE_S3_ACCESS_KEY_ID = os.getenv("SUPABASE_S3_ACCESS_KEY_ID")
SUPABASE_S3_SECRET_ACCESS_KEY = os.getenv("SUPABASE_S3_SECRET_ACCESS_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET_NAME")

# Initialize Supabase client and S3 client
supabase: Client = create_client(SUPABASE_URL, "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJjYWV1Z3hoYW9rcmFua3V3dHNhIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0MjIyMDQ1OSwiZXhwIjoyMDU3Nzk2NDU5fQ.V7lTaijvG7SQR51OoJvMiEzW-AScoYqA-FPrDiERiRk")

s3_client = boto3.client(
    "s3",
    endpoint_url=SUPABASE_S3_ENDPOINT,
    aws_access_key_id=SUPABASE_S3_ACCESS_KEY_ID,
    aws_secret_access_key=SUPABASE_S3_SECRET_ACCESS_KEY,
    region_name=SUPABASE_S3_REGION,
)

# Connect to Supabase Postgres
async def connect_to_db():
    return await asyncpg.connect(SUPABASE_DB_URL)

# Insert metadata into the database
async def insert_metadata(group: str, num_rows: int, file_path: str, file_date: datetime.date):
    conn = await connect_to_db()
    query = """
    INSERT INTO metadata (social_group, num_rows, file_path, date)
    VALUES ($1, $2, $3, $4)
    """
    await conn.execute(query, group, num_rows, file_path, file_date)
    await conn.close()

# Main file processing loop
async def process_files():
    LOCAL_ROOT = "C:/Users/91956/Downloads/race-20250414T114220Z-001/" 
    SOCIAL_GROUPS = ["race"] 

    for group in SOCIAL_GROUPS:
        folder_path = os.path.join(LOCAL_ROOT, group)
        for filename in os.listdir(folder_path):
            if not filename.endswith(".csv"):
                continue

            local_file_path = os.path.join(folder_path, filename)
            storage_path = f"corpus/{group}/{filename}"

            df = pd.read_csv(local_file_path, engine="python")
            df.dropna(how="all", inplace=True)
            num_rows = len(df) - 1

            os.makedirs("temp_cleaned", exist_ok=True)
            cleaned_path = os.path.join("temp_cleaned", filename)
            df.to_csv(cleaned_path, index=False, encoding="utf-8")

            # Upload cleaned file to Supabase
            # s3_client.upload_file(cleaned_path, SUPABASE_BUCKET, storage_path)
            with open(cleaned_path, "rb") as f:
                supabase.storage.from_(SUPABASE_BUCKET).upload(
                    path=storage_path,
                    file=f,
                    file_options={"content-type": "text/csv"}
                )


            file_path = f"https://{SUPABASE_URL.split('//')[1]}/storage/v1/object/public/{SUPABASE_BUCKET}/{storage_path}"

            date_str = filename.replace("RC_", "").replace(".csv", "") + "-01"
            file_date = datetime.strptime(date_str, "%Y-%m-%d").date()

            await insert_metadata(group, num_rows, file_path, file_date)

            print(f"Uploaded and recorded: {filename} → {num_rows} rows")

    print("All files processed successfully!")

if __name__ == "__main__":
    asyncio.run(process_files())
