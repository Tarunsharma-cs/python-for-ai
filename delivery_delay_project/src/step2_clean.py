"""Step 2: Data cleaning - duplicates, impossible values, missing values."""
import numpy as np
import pandas as pd

from config import CLEAN_DATA, RAW_DATA


def main():
    df = pd.read_csv(RAW_DATA)
    print("Before cleaning:", df.shape)

    # 1. Remove duplicate rows
    df = df.drop_duplicates()

    # 2. The ID is not useful for prediction
    df = df.drop(columns=["delivery_id"])

    # 3. Unrealistic values -> treat distances as missing, drop impossible times
    bad_distance = (df["distance_km"] <= 0) | (df["distance_km"] > 100)
    df.loc[bad_distance, "distance_km"] = np.nan
    df = df[(df["delivery_time_min"] >= 5) & (df["delivery_time_min"] <= 300)]

    # 4. Fill missing values: median for numbers, most common value for categories
    for col in ["distance_km", "driver_rating"]:
        df[col] = df[col].fillna(df[col].median())
    for col in ["traffic_level", "weather_condition"]:
        df[col] = df[col].fillna(df[col].mode()[0])

    # 5. Make sure data types are correct
    df["delayed"] = df["delayed"].astype(int)
    for col in ["vehicle_type", "traffic_level", "weather_condition", "delivery_priority"]:
        df[col] = df[col].astype("category")

    df = df.reset_index(drop=True)
    df.to_csv(CLEAN_DATA, index=False)

    print("After cleaning: ", df.shape)
    print("Missing values left:", int(df.isna().sum().sum()))
    print("\n", df.head().to_string())
    print(f"\nSaved to {CLEAN_DATA}")


if __name__ == "__main__":
    main()
