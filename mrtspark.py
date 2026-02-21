#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, sum as spark_sum, to_date
from sklearn.cluster import KMeans
from prophet import Prophet

# ---------------------------
# SETTINGS
# ---------------------------
INPUT_FILE = "trainvolume.csv"
OUTPUT_DIR = "output"
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# INITIALIZE SPARK
# ---------------------------
spark = SparkSession.builder.appName("Singapore MRT Analysis").getOrCreate()

# ---------------------------
# LOAD DATA
# ---------------------------
df = spark.read.csv(INPUT_FILE, header=True, inferSchema=True)
print("Data loaded:")
df.show(5)

# ---------------------------
# PREPROCESS
# ---------------------------
# Convert train_volume_year_month to date
df = df.withColumn("date", to_date("train_volume_year_month", "yyyy-MM-dd"))
# Total volume per row
df = df.withColumn("total_volume", col("train_volume_tap_in") + col("train_volume_tap_out"))

# ---------------------------
# EDA: Hourly Stats
# ---------------------------
hourly_stats = df.groupBy("train_volume_hour") \
    .agg(avg("total_volume").alias("avg_volume")) \
    .orderBy("train_volume_hour")
hourly_stats_pd = hourly_stats.toPandas()
hourly_stats_pd.to_csv(f"{OUTPUT_DIR}/hourly_stats.csv", index=False)

plt.figure()
plt.bar(hourly_stats_pd["train_volume_hour"], hourly_stats_pd["avg_volume"], color="skyblue")
plt.xlabel("Hour of Day")
plt.ylabel("Average Passenger Volume")
plt.title("Hourly Passenger Volume")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/hourly_stats.png")
plt.close()

# ---------------------------
# Peak Hour Detection
# ---------------------------
peak_hour_pd = hourly_stats.orderBy(col("avg_volume").desc()).limit(1).toPandas()
peak_hour_pd.to_csv(f"{OUTPUT_DIR}/peak_hour.csv", index=False)

plt.figure()
plt.bar(hourly_stats_pd["train_volume_hour"], hourly_stats_pd["avg_volume"], color='orange')
plt.axvline(peak_hour_pd['train_volume_hour'][0], color='red', linestyle='--', label='Peak Hour')
plt.xlabel("Hour of Day")
plt.ylabel("Average Passenger Volume")
plt.title("Peak Hour Detection")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/peak_hour.png")
plt.close()

# ---------------------------
# Daily Time-Series Forecasting
# ---------------------------
daily_volume = df.groupBy("date") \
    .agg(spark_sum("total_volume").alias("total_volume")) \
    .orderBy("date") \
    .toPandas()
daily_volume.rename(columns={"date": "ds", "total_volume": "y"}, inplace=True)

m = Prophet(daily_seasonality=True)
m.fit(daily_volume)
future = m.make_future_dataframe(periods=30)
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(f"{OUTPUT_DIR}/forecast.csv", index=False)

fig = m.plot(forecast)
fig.savefig(f"{OUTPUT_DIR}/forecast_plot.png")
plt.close()

# ---------------------------
# Station Clustering
# ---------------------------
station_df = df.groupBy("train_code") \
    .agg(avg("total_volume").alias("avg_volume")) \
    .toPandas()
kmeans_station = KMeans(n_clusters=3, random_state=RANDOM_SEED)
station_df['cluster'] = kmeans_station.fit_predict(station_df[['avg_volume']])
station_df.to_csv(f"{OUTPUT_DIR}/station_clusters.csv", index=False)

# ---------------------------
# Improved Station Clustering Plot
# ---------------------------
plt.figure(figsize=(20,6))  # wider figure for many stations
for c in station_df['cluster'].unique():
    subset = station_df[station_df['cluster'] == c]
    plt.scatter(subset['train_code'], subset['avg_volume'], label=f'Cluster {c}', s=50)

plt.xticks(rotation=90)  # rotate x labels
plt.xlabel("Train Code")
plt.ylabel("Average Volume")
plt.title("Station Clusters")
plt.legend()

# Optional: reduce label density if too crowded
# plt.xticks(ticks=range(0, len(station_df['train_code']), 2),
#            labels=station_df['train_code'][::2], rotation=90)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/station_clusters.png", dpi=300)
plt.close()

# ---------------------------
# Spark SQL Dashboard
# ---------------------------
df.createOrReplaceTempView("mrt_volume")

busiest_station = spark.sql("""
SELECT train_code, AVG(total_volume) AS avg_volume
FROM mrt_volume
GROUP BY train_code
ORDER BY avg_volume DESC
LIMIT 5
""")
busiest_station.toPandas().to_csv(f"{OUTPUT_DIR}/busiest_stations.csv", index=False)

top_hours = spark.sql("""
SELECT train_code, train_volume_hour AS hour, AVG(total_volume) AS avg_volume
FROM mrt_volume
GROUP BY train_code, train_volume_hour
ORDER BY avg_volume DESC
""")
top_hours.toPandas().to_csv(f"{OUTPUT_DIR}/top_hours_per_station.csv", index=False)

# ---------------------------
# Markdown Report
# ---------------------------
report_md = f"""
# Singapore MRT Analysis Report

## 1. Overview
- Total rows: {df.count()}
- Columns: {df.columns}

## 2. EDA
- Hourly passenger volume saved as `hourly_stats.csv` and `hourly_stats.png`

## 3. Peak Hour Detection
- Peak hour: {peak_hour_pd['train_volume_hour'][0]}
- CSV: `peak_hour.csv`, Plot: `peak_hour.png`

## 4. Forecasting
- Forecast for next 30 days saved as `forecast.csv` and `forecast_plot.png`

## 5. Station/Line Segmentation
- Station clusters saved in `station_clusters.csv` and `station_clusters.png`

## 6. Spark SQL Dashboard
- Top stations: `busiest_stations.csv`
- Top hours per station: `top_hours_per_station.csv`
"""

with open(f"{OUTPUT_DIR}/report.md", "w") as f:
    f.write(report_md)

print("✅ All outputs saved in folder:", OUTPUT_DIR)
spark.stop()