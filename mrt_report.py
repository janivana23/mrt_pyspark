# mrtspark_report.py
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import pandas as pd

# -------------------------
# Setup Spark
# -------------------------
spark = SparkSession.builder \
    .appName("Singapore MRT Analysis") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# -------------------------
# Paths
# -------------------------
DATA_PATH = "trainvolume.csv"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Load Data
# -------------------------
df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)
print("Data loaded:")
df.show(5)

# -------------------------
# Data Preprocessing
# -------------------------
# For time-based analysis, use the existing 'train_volume_hour' column
# Convert Spark DataFrame to Pandas for plotting
pdf = df.toPandas()

# -------------------------
# Plot 1: Peak Hours
# -------------------------
plt.figure(figsize=(12,6))
sns.barplot(data=pdf.groupby('train_volume_hour')['train_volume_tap_in'].sum().reset_index(),
            x='train_volume_hour', y='train_volume_tap_in', palette="viridis")
plt.title("Peak Hours Tap-ins")
plt.xlabel("Hour of Day")
plt.ylabel("Total Tap-ins")
plt.tight_layout()
peak_hours_path = os.path.join(OUTPUT_DIR, "peak_hours.png")
plt.savefig(peak_hours_path)
plt.close()

# -------------------------
# Plot 2: Station Clustering (Cluster by total tap-ins)
# -------------------------
station_totals = pdf.groupby('train_code')['train_volume_tap_in'].sum().sort_values(ascending=False)
plt.figure(figsize=(14,6))
sns.barplot(x=station_totals.index, y=station_totals.values, palette="magma")
plt.title("Total Tap-ins per Station")
plt.xlabel("Train Code")
plt.ylabel("Total Tap-ins")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
station_cluster_path = os.path.join(OUTPUT_DIR, "station_clusters.png")
plt.savefig(station_cluster_path)
plt.close()

# -------------------------
# Plot 3: Forecast Placeholder
# -------------------------
# Here you can replace with your actual forecast data
forecast = pdf.groupby('train_volume_hour')['train_volume_tap_in'].mean().reset_index()
plt.figure(figsize=(12,6))
sns.lineplot(data=forecast, x='train_volume_hour', y='train_volume_tap_in', marker='o')
plt.title("Average Hourly Tap-ins Forecast")
plt.xlabel("Hour")
plt.ylabel("Avg Tap-ins")
plt.tight_layout()
forecast_path = os.path.join(OUTPUT_DIR, "forecast.png")
plt.savefig(forecast_path)
plt.close()

# -------------------------
# Generate PDF Report
# -------------------------
pdf_report_path = os.path.join(OUTPUT_DIR, "MRT_Report.pdf")
with PdfPages(pdf_report_path) as pdf_file:

    # Title Page
    plt.figure(figsize=(11,8.5))
    plt.axis('off')
    plt.text(0.5, 0.6, "Singapore MRT Data Analysis", fontsize=28, ha='center')
    plt.text(0.5, 0.5, "Portfolio Report", fontsize=22, ha='center')
    plt.text(0.5, 0.4, "Author: Janice Ivana", fontsize=18, ha='center')
    pdf_file.savefig()
    plt.close()

    # Peak Hours Page
    peak_fig = plt.figure(figsize=(11,8.5))
    peak_img = plt.imread(peak_hours_path)
    plt.imshow(peak_img)
    plt.axis('off')
    pdf_file.savefig()
    plt.close()

    # Station Clustering Page
    cluster_fig = plt.figure(figsize=(11,8.5))
    cluster_img = plt.imread(station_cluster_path)
    plt.imshow(cluster_img)
    plt.axis('off')
    pdf_file.savefig()
    plt.close()

    # Forecast Page
    forecast_fig = plt.figure(figsize=(11,8.5))
    forecast_img = plt.imread(forecast_path)
    plt.imshow(forecast_img)
    plt.axis('off')
    pdf_file.savefig()
    plt.close()

    # Summary Table Page
    plt.figure(figsize=(11,8.5))
    plt.axis('off')
    df_summary = pd.DataFrame({
        "Metric": ["Total Tap-ins", "Total Tap-outs", "Avg Hourly Tap-ins"],
        "Value": [pdf['train_volume_tap_in'].sum(),
                  pdf['train_volume_tap_out'].sum(),
                  pdf['train_volume_tap_in'].mean()]
    })
    plt.table(cellText=df_summary.values,
              colLabels=df_summary.columns,
              cellLoc='center',
              loc='center')
    pdf_file.savefig()
    plt.close()

print(f"✅ Portfolio PDF saved: {pdf_report_path}")