import os
import time
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


def safe_str(value):
    """Convert None to empty and escape single quotes."""
    if value is None:
        return ""
    return str(value).replace("'", "''")


def load_to_supabase():

    csv_path = "../data/staged/nasa_apod_cleaned.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing file: {csv_path}")

    df = pd.read_csv(csv_path)

    # Standardize extracted_at format 
    df["extracted_at"] = pd.to_datetime(df["extracted_at"]).dt.strftime(
        "%Y-%m-%dT%H:%M:%S"
    )

    # Replace NaN with None for SQL
    batch = df.where(pd.notnull(df), None).to_dict("records")

    values = [
        (
            f"('{r['date']}', "
            f"'{safe_str(r.get('title'))}', "
            f"'{safe_str(r.get('explanation'))}', "
            f"'{safe_str(r.get('media_type'))}', "
            f"'{safe_str(r.get('image_url'))}', "
            f"'{safe_str(r.get('hd_image_url'))}', "
            f"'{safe_str(r.get('copyright'))}', "
            f"'{safe_str(r.get('service_version'))}', "
            f"'{r['extracted_at']}')"
        )
        for r in batch
    ]

    insert_sql = (
        "INSERT INTO nasa_apod "
        "(apod_date, title, explanation, media_type, image_url, hd_image_url, "
        "copyright, service_version, extracted_at) "
        f"VALUES {','.join(values)};"
    )

    try:
        supabase.rpc("execute_sql", {"query": insert_sql}).execute()
        print("✅ NASA APOD inserted successfully!")

    except Exception as e:
        print("❌ Error inserting APOD:")
        print(e)
        print(insert_sql)


if __name__ == "__main__":
    load_to_supabase()