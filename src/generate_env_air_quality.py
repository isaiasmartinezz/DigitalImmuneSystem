import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

# Simple helper so you see where you are running from
print("Python executable:", sys.executable)
print("Current working dir:", os.getcwd())

# -------- CONFIG --------
# Choose a city / location that makes sense for your scenario
CITY = "Los Angeles"
COUNTRY = "US"
PARAMETER = "pm25"   # fine particulate matter
DAYS_BACK = 30       # how many days of data to pull

OUTPUT_PATH = "data/env_air_quality_openaq.csv"


def fetch_openaq_pm25(city: str, country: str, days_back: int = 30) -> pd.DataFrame:
    """
    Fetch recent PM2.5 measurements from the OpenAQ API v2 and return as a DataFrame.
    We Aggregate by date to match your outbreak-detection style.
    """
    print(f"\n📡 Fetching OpenAQ data for {city}, {country}, last {days_back} days...")

    # OpenAQ v2 measurements endpoint
    base_url = "https://api.openaq.org/v2/measurements"

    # Time range: from days_back ago to now
    date_to = datetime.utcnow()
    date_from = date_to - timedelta(days=days_back)

    # Build parameters for request
    params = {
        "city": city,
        "country": country,
        "parameter": PARAMETER,
        "date_from": date_from.isoformat(timespec="seconds") + "Z",
        "date_to": date_to.isoformat(timespec="seconds") + "Z",
        "limit": 10000,          # max records
        "page": 1,
        "sort": "desc",
        "order_by": "datetime"
    }

    print("Requesting from:", base_url)
    print("With params:", params)

    resp = requests.get(base_url, params=params, timeout=30)

    if resp.status_code != 200:
        print("❌ Error from OpenAQ:", resp.status_code, resp.text)
        raise RuntimeError(f"OpenAQ request failed with status {resp.status_code}")

    data = resp.json()

    if "results" not in data or not data["results"]:
        print("⚠️ No results returned from OpenAQ.")
        return pd.DataFrame()

    print(f"✅ Retrieved {len(data['results'])} raw measurement records.")

    # Convert to DataFrame
    records = []
    for r in data["results"]:
        records.append({
            "location": r.get("location"),
            "city": r.get("city"),
            "country": r.get("country"),
            "parameter": r.get("parameter"),
            "value": r.get("value"),
            "unit": r.get("unit"),
            "datetime_utc": r.get("date", {}).get("utc"),
        })

    df = pd.DataFrame(records)

    if df.empty:
        print("⚠️ DataFrame is empty after conversion.")
        return df

    # Parse datetime and aggregate by date
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df["date"] = df["datetime_utc"].dt.date

    # Aggregate: mean PM2.5 per day for the chosen city
    daily = (
        df.groupby("date")["value"]
        .mean()
        .reset_index()
        .rename(columns={"value": "pm25_mean"})
    )

    print("\n📌 First few daily aggregated rows:")
    print(daily.head())

    return daily


def main():
    # Ensure data dir exists
    os.makedirs("data", exist_ok=True)

    daily_pm25 = fetch_openaq_pm25(CITY, COUNTRY, DAYS_BACK)

    if daily_pm25.empty:
        print("⚠️ No data to save. Exiting.")
        return

    # To match your pipeline style, rename columns:
    #   - date -> sample_collect_date
    #   - pm25_mean -> env_signal (or something similar)
    daily_pm25 = daily_pm25.rename(
        columns={
            "date": "sample_collect_date",
            "pm25_mean": "env_signal"
        }
    )

    # Add dummy "location_id" so the schema matches your others
    daily_pm25["env_location_id"] = f"{CITY}_pm25"

    # Reorder columns
    daily_pm25 = daily_pm25[["env_location_id", "sample_collect_date", "env_signal"]]

    # Sort by date
    daily_pm25 = daily_pm25.sort_values("sample_collect_date")

    print("\n✅ Final environmental DataFrame (head):")
    print(daily_pm25.head())

    # Save to CSV
    daily_pm25.to_csv(OUTPUT_PATH, index=False)
    print(f"\n💾 Saved OpenAQ-based environmental data to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
