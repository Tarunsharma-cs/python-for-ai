"""Step 3: Data visualization - histogram, boxplot, bar charts, heatmap."""
import matplotlib

matplotlib.use("Agg")  # save figures without opening a window
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import FIG_DIR
from utils import load_clean

sns.set_theme(style="whitegrid")
ORDER_TRAFFIC = ["Low", "Medium", "High"]
ORDER_WEATHER = ["Clear", "Cloudy", "Rainy", "Foggy", "Stormy"]


def save(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=150)
    plt.close()


def main():
    df = load_clean()

    # Histogram of delivery times
    plt.figure(figsize=(7, 4))
    sns.histplot(df["delivery_time_min"], bins=40, kde=True)
    plt.title("Distribution of delivery time")
    plt.xlabel("Delivery time (minutes)")
    save("01_hist_delivery_time.png")

    # Boxplot of distance
    plt.figure(figsize=(7, 3.5))
    sns.boxplot(x=df["distance_km"])
    plt.title("Delivery distance")
    plt.xlabel("Distance (km)")
    save("02_box_distance.png")

    # Delayed rate by traffic level
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x="traffic_level", y="delayed", order=ORDER_TRAFFIC, errorbar=None)
    plt.title("Share of delayed deliveries by traffic level")
    plt.ylabel("Delay rate")
    save("03_bar_delay_by_traffic.png")

    # Delayed rate by weather
    plt.figure(figsize=(7, 4))
    sns.barplot(data=df, x="weather_condition", y="delayed", order=ORDER_WEATHER, errorbar=None)
    plt.title("Share of delayed deliveries by weather")
    plt.ylabel("Delay rate")
    save("04_bar_delay_by_weather.png")

    # Delayed rate by vehicle and by priority
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.barplot(data=df, x="vehicle_type", y="delayed", errorbar=None, ax=axes[0])
    axes[0].set_title("Delay rate by vehicle type")
    sns.barplot(data=df, x="delivery_priority", y="delayed",
                order=["Low", "Standard", "Express"], errorbar=None, ax=axes[1])
    axes[1].set_title("Delay rate by delivery priority")
    save("05_bar_delay_by_vehicle_priority.png")

    # Correlation heatmap (encode categories so they can be correlated)
    enc = df.drop(columns=["delivery_time_min"]).copy()
    enc["traffic_level"] = enc["traffic_level"].map({"Low": 0, "Medium": 1, "High": 2})
    enc["delivery_priority"] = enc["delivery_priority"].map({"Low": 0, "Standard": 1, "Express": 2})
    enc = pd.get_dummies(enc, columns=["vehicle_type", "weather_condition"], dtype=int)
    plt.figure(figsize=(10, 8))
    sns.heatmap(enc.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, annot_kws={"size": 7})
    plt.title("Correlation heatmap")
    save("06_correlation_heatmap.png")

    # Written insights computed from the data
    by_traffic = df.groupby("traffic_level", observed=True)["delayed"].mean().reindex(ORDER_TRAFFIC)
    by_weather = df.groupby("weather_condition", observed=True)["delayed"].mean().reindex(ORDER_WEATHER)
    corr = enc.corr()["delayed"].drop("delayed").sort_values(key=abs, ascending=False)
    print("Delay rate by traffic:\n", by_traffic.round(3).to_string())
    print("\nDelay rate by weather:\n", by_weather.round(3).to_string())
    print("\nStrongest correlations with 'delayed':\n", corr.head(5).round(3).to_string())
    print(f"\nFigures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
