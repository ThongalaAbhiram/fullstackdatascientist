import pandas as pd
import json
import glob
import os
from datetime import datetime

def transform_nasa_apod():
    # Create staged directory if missing
    os.makedirs("../data/staged", exist_ok=True)

    # Get latest APOD JSON file
    latest_file = sorted(glob.glob("../data/raw/apod_*.json"))[-1]

    with open(latest_file, "r") as f:
        data = json.load(f)

    # Convert to DataFrame (APOD has 1 record/day)
    df = pd.DataFrame([{
        "date": data.get("date"),
        "title": data.get("title"),
        "explanation": data.get("explanation"),
        "media_type": data.get("media_type"),
        "image_url": data.get("url"),
        "hd_image_url": data.get("hdurl"),
        "copyright": data.get("copyright"),
        "service_version": data.get("service_version"),
        "extracted_at": datetime.now()
    }])

    # Output path
    output_path = "../data/staged/nasa_apod_cleaned.csv"
    df.to_csv(output_path, index=False)

    print(f"Transformed APOD data saved to: {output_path}")
    return df


if __name__ == "__main__":
    transform_nasa_apod()