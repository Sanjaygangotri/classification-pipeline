import polars as pl
import time

DATASET_PATH = "dataset 10M.parquet"

print("Starting Dataset Analysis...\n")
start_time = time.time()

# Connect lazily to avoid loading full file into memory
lazy_df = pl.scan_parquet(DATASET_PATH)

# Dataset Metadata
total_rows = lazy_df.select(pl.len()).collect().item()
schema = lazy_df.schema

print(f"=== 1. DATASET OVERVIEW ===")
print(f"Total Rows: {total_rows:,}")
print(f"Schema: {schema}\n")

# Class Distribution
print("=== 2. TOPIC DISTRIBUTION ===")
topic_counts = (
    lazy_df.group_by("TOPIC")
    .agg(pl.len().alias("count"))
    .with_columns((pl.col("count") / total_rows * 100).round(2).alias("percentage"))
    .sort("count", descending=True)
    .collect(engine="streaming")
)
print(topic_counts)
print()

# Missing Values Check
print("=== 3. MISSING VALUE CHECK ===")
null_counts = lazy_df.select(
    [pl.col("DATA").null_count().alias("null_data"), pl.col("TOPIC").null_count().alias("null_topic")]
).collect()
print(null_counts)
print()

# Text Statistics (Sampled on 100,000 rows for high speed)
print("=== 4. TEXT LENGTH STATISTICS (100k Sample) ===")
sample_df = (
    lazy_df.select(["DATA", "TOPIC"])
    .collect(engine="streaming")
    .sample(n=100_000, seed=42)
)

stats_df = sample_df.with_columns(
    [
        pl.col("DATA").str.len_bytes().alias("char_len"),
        pl.col("DATA").str.split(" ").list.len().alias("word_count"),
    ]
)

print(stats_df.select(["char_len", "word_count"]).describe(percentiles=[0.5, 0.90, 0.95, 0.99]))

print(f"\nAnalysis completed in {round(time.time() - start_time, 2)} seconds.")