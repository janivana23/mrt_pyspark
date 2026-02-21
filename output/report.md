
# Singapore MRT Analysis Report

## 1. Overview
- Total rows: 11702
- Columns: ['train_volume_id', 'train_volume_year_month', 'train_volume_day', 'train_volume_hour', 'train_code', 'train_volume_tap_in', 'train_volume_tap_out', 'date', 'total_volume']

## 2. EDA
- Hourly passenger volume saved as `hourly_stats.csv` and `hourly_stats.png`

## 3. Peak Hour Detection
- Peak hour: 18
- CSV: `peak_hour.csv`, Plot: `peak_hour.png`

## 4. Forecasting
- Forecast for next 30 days saved as `forecast.csv` and `forecast_plot.png`

## 5. Station/Line Segmentation
- Station clusters saved in `station_clusters.csv` and `station_clusters.png`

## 6. Spark SQL Dashboard
- Top stations: `busiest_stations.csv`
- Top hours per station: `top_hours_per_station.csv`
