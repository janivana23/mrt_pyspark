# mrtspark_full.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
import matplotlib.pyplot as plt

# -------------------------
# 1. Initialize Spark
# -------------------------
spark = SparkSession.builder.appName("MRT_DataScience").getOrCreate()

# -------------------------
# 2. Load CSV data
# -------------------------
df = spark.read.csv("trainvolume.csv", header=True, inferSchema=True)

# Convert date column
df = df.withColumn("train_volume_date", to_date(col("train_volume_year_month"), "yyyy-MM-dd"))

print("Columns in DF:", df.columns)
df.show(5)

# -------------------------
# 3. Exploratory Data Analysis
# -------------------------

# a) Check for missing values
from pyspark.sql.functions import isnan, when, count
df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]).show()

# b) Basic statistics
df.describe(["train_volume_tap_in", "train_volume_tap_out"]).show()

# c) Passenger counts by train line
df.groupBy("train_code").sum("train_volume_tap_in", "train_volume_tap_out").show()

# d) Passenger distribution by day type
df.groupBy("train_volume_day").avg("train_volume_tap_in", "train_volume_tap_out").show()

# e) Hourly passenger patterns
df.groupBy("train_volume_hour").avg("train_volume_tap_in", "train_volume_tap_out").orderBy("train_volume_hour").show()

# -------------------------
# 4. Feature Engineering
# -------------------------

# Total passengers
df = df.withColumn("total_passengers", col("train_volume_tap_in") + col("train_volume_tap_out"))

# Encode day type
indexer = StringIndexer(inputCol="train_volume_day", outputCol="day_type_index")
df = indexer.fit(df).transform(df)

# Use hour from train_volume_hour
df = df.withColumn("hour_extracted", col("train_volume_hour"))

df.show(5)

# -------------------------
# 5. Aggregations
# -------------------------

# a) Daily passenger volumes per train line
daily_df = df.groupBy("train_volume_date", "train_code") \
             .sum("total_passengers") \
             .withColumnRenamed("sum(total_passengers)", "daily_total_passengers")
daily_df.show(5)

# b) Average passengers per hour
hourly_df = df.groupBy("train_volume_hour") \
              .avg("total_passengers") \
              .orderBy("train_volume_hour")
hourly_df.show()

# -------------------------
# 6. Visualization
# -------------------------

# Convert to Pandas for plotting
hourly_pd = hourly_df.toPandas()

plt.figure(figsize=(10,5))
plt.plot(hourly_pd['train_volume_hour'], hourly_pd['avg(total_passengers)'], marker='o')
plt.xlabel("Hour of Day")
plt.ylabel("Average Passengers")
plt.title("Average Passengers by Hour")
plt.grid(True)
plt.show()

# -------------------------
# 7. Predictive Modeling
# -------------------------

# Features: hour and day type index
assembler = VectorAssembler(inputCols=["hour_extracted", "day_type_index"], outputCol="features")
train_data = assembler.transform(df).select("features", "total_passengers")

# Linear regression
lr = LinearRegression(featuresCol="features", labelCol="total_passengers")
model = lr.fit(train_data)

# Predictions
predictions = model.transform(train_data)
predictions.select("features", "total_passengers", "prediction").show(5)

# -------------------------
# 8. Stop Spark
# -------------------------
spark.stop()