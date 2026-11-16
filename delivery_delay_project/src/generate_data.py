import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 2000

distance = np.round(rng.gamma(shape=3.0, scale=6.0, size=n) + 1, 1)          # km
vehicle = rng.choice(["Bike", "Scooter", "Car", "Van"], n, p=[0.25, 0.35, 0.25, 0.15])
traffic = rng.choice(["Low", "Medium", "High"], n, p=[0.3, 0.45, 0.25])
weather = rng.choice(["Clear", "Cloudy", "Rainy", "Foggy", "Stormy"], n, p=[0.40, 0.25, 0.20, 0.10, 0.05])
rating = np.round(np.clip(rng.normal(4.3, 0.5, n), 2.5, 5.0), 1)
priority = rng.choice(["Low", "Standard", "Express"], n, p=[0.2, 0.6, 0.2])

speed = {"Bike": 18, "Scooter": 25, "Car": 30, "Van": 28}
traffic_mult = {"Low": 1.0, "Medium": 1.25, "High": 1.6}
weather_mult = {"Clear": 1.0, "Cloudy": 1.05, "Rainy": 1.25, "Foggy": 1.3, "Stormy": 1.6}

base = np.array([distance[i] / speed[vehicle[i]] * 60 for i in range(n)])
t_mult = np.array([traffic_mult[t] for t in traffic])
w_mult = np.array([weather_mult[w] for w in weather])
rating_effect = 1 + (4.3 - rating) * 0.08
delivery_time = np.round(
    (base * t_mult * w_mult * rating_effect + 10 + rng.normal(0, 5, n)).clip(8), 1
)

# On-time promise depends on priority (minutes allowed beyond the ideal time)
promised = base + 10 + np.array([{"Low": 25, "Standard": 18, "Express": 10}[p] for p in priority])
delayed = (delivery_time > promised).astype(int)

df = pd.DataFrame({
    "delivery_id": np.arange(1, n + 1),
    "distance_km": distance,
    "vehicle_type": vehicle,
    "traffic_level": traffic,
    "weather_condition": weather,
    "driver_rating": rating,
    "delivery_priority": priority,
    "delivery_time_min": delivery_time,
    "delayed": delayed,
})

# --- add realistic mess for Step 2 (cleaning) ---
for col, frac in [("distance_km", 0.04), ("driver_rating", 0.03), ("traffic_level", 0.02), ("weather_condition", 0.02)]:
    idx = rng.choice(n, int(n * frac), replace=False)
    df.loc[idx, col] = np.nan

bad = rng.choice(n, 12, replace=False)
df.loc[bad[:6], "delivery_time_min"] = rng.choice([-5, 0, 999, 1500], 6)   # unrealistic times
df.loc[bad[6:], "distance_km"] = rng.choice([-3.0, 450.0], 6)              # unrealistic distances

dupes = df.sample(25, random_state=1)
df = pd.concat([df, dupes], ignore_index=True).sample(frac=1, random_state=7).reset_index(drop=True)

from pathlib import Path
out = Path(__file__).resolve().parent.parent / "data" / "raw" / "logistics_delivery_data.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(df.shape, "| delayed rate:", round(df["delayed"].mean(), 3))
