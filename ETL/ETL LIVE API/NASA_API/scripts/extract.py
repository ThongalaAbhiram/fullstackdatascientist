import json
from pathlib import Path
from datetime import datetime
import requests
import time

# Directory: project/data/raw/
data_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
data_dir.mkdir(parents=True, exist_ok=True)

def extract_apod_data(api_key="MeQHhynLANvNpFwHqXawS3vkTuYCWR9VDYCSKtaB"):
    url = "https://api.nasa.gov/planetary/apod"
    params = {
        "api_key": api_key,
        "hd": True
    }

    # Retry 429 Too Many Requests
    for attempt in range(3):
        resp = requests.get(url, params=params)

        if resp.status_code == 429:
            print("⚠️ NASA Rate Limit Hit — waiting 2 seconds...")
            time.sleep(2)
            continue

        resp.raise_for_status()
        data = resp.json()

        # Save file
        filename = data_dir / f"apod_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filename.write_text(json.dumps(data, indent=2))

        print(f"✔ NASA APOD data saved to: {filename}")
        return data

    raise Exception("❌ Failed 3 times due to NASA rate limit.")


if __name__ == "__main__":
    extract_apod_data()