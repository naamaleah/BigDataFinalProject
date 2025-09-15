# ================== ALS Recommendations - fixed full version ==================
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, desc, broadcast
from pyspark.ml.recommendation import ALSModel
import pandas as pd
from random import sample
import os
import math
import time

# ================== CONFIG ==================
BASE_PATH = r"C:\Users\shira\OneDrive\שולחן העבודה\Big Data\final"
DB_PATH = r"C:\Users\shira\OneDrive\שולחן העבודה\Big Data\final\books\books_no_outliers.db"
NUM_USERS = 500          # number of sampled users
NUM_RECS = 10            # number of final recommendations per user
NUM_LARGER = 20          # number of raw ALS recs before filtering down
TOP_POPULAR = 100        # pool size for popular books to fill gaps

# ================== Spark setup ==================
spark = SparkSession.builder \
    .appName("ALS_Recommendations_Fixed") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.jars", r"C:\spark\jars\sqlite-jdbc-3.45.2.0.jar") \
    .config("spark.driver.extraClassPath", r"C:\spark\jars\sqlite-jdbc-3.45.2.0.jar") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

start_time = time.time()
print(f"START: {time.asctime()}")

# ================== Helpers ==================
def load_table(table_name):
    """Load a SQLite table into a Spark DataFrame."""
    return spark.read.format("jdbc") \
        .option("url", f"jdbc:sqlite:{DB_PATH}") \
        .option("dbtable", table_name) \
        .option("driver", "org.sqlite.JDBC") \
        .load()

# ================== Load tables ==================
print("Loading tables (this may take a moment)...")
ratings = load_table("Book_Ratings")   # used for popularity stats
user_mapping = spark.read.parquet(f"{BASE_PATH}\\user_mapping")
item_mapping = spark.read.parquet(f"{BASE_PATH}\\item_mapping")
books = load_table("Books").select(col("ISBN").alias("book_id"), col("Book-Title").alias("book_title"))
model = ALSModel.load(f"{BASE_PATH}\\als_model")
print("Loaded model and metadata.")

# ================== Choose sample users ==================
print(f"Sampling first {NUM_USERS} users from user_mapping...")
sample_users = user_mapping.limit(NUM_USERS).select("userId", "original_userId")
sample_users_pd = sample_users.toPandas()
print(f"Sampled {len(sample_users_pd)} users.")

# ================== ALS recommendations ==================
print(f"Requesting up to {NUM_LARGER} ALS recommendations per sampled user...")
user_recs = model.recommendForUserSubset(sample_users.select("userId"), NUM_LARGER)
print("ALS recommendForUserSubset returned.")

# Explode recommendations and join metadata
print("Exploding ALS results and joining to book titles...")
recs_spark = (
    user_recs
    .withColumn("rec", explode("recommendations"))
    .select("userId",
            col("rec.itemId").alias("itemId"),
            col("rec.rating").alias("predicted_rating"))
    .join(sample_users, "userId")
    .join(item_mapping, "itemId")
    .join(broadcast(books), item_mapping["original_itemId"] == books["book_id"])
    .select("userId", "original_userId", "book_title", "predicted_rating")
)

top_recs_pd = recs_spark.toPandas()
print(f"ALS provided predictions for {top_recs_pd['original_userId'].nunique()} of {len(sample_users_pd)} sampled users.")

# ================== Popular books pool ==================
print("Computing popular books pool...")
popular_books_spark = (
    ratings.groupBy("ISBN")
    .count()
    .orderBy(desc("count"))
    .limit(TOP_POPULAR * 3)
    .join(books, books.book_id == col("ISBN"))
    .select("book_title")
)
popular_books_list = [r.book_title for r in popular_books_spark.collect()]
popular_books_list = list(dict.fromkeys(popular_books_list))  # deduplicate
if len(popular_books_list) == 0:
    raise RuntimeError("Popular books list is empty — check Book_Ratings and Books tables.")
print(f"Built popular pool with {len(popular_books_list)} titles.")

popular_pool = popular_books_list[:TOP_POPULAR]

# ================== Build final per-user recommendations ==================
print("Assembling final recommendations for each sampled user...")
rows_out = []

# Pre-group ALS predictions
if not top_recs_pd.empty:
    top_recs_pd = top_recs_pd.sort_values(["original_userId", "predicted_rating"], ascending=[True, False])
    grouped = {}
    for _, r in top_recs_pd.iterrows():
        uid = r["original_userId"]
        if uid not in grouped:
            grouped[uid] = []
        title = r["book_title"]
        if title not in grouped[uid]:
            grouped[uid].append((title, r["predicted_rating"]))
else:
    grouped = {}

import random
random.seed(42)

# Guarantee output for every sampled user
for _, urow in sample_users_pd.iterrows():
    uid = urow["original_userId"]
    user_list = grouped.get(uid, []).copy()
    seen = set([t for t, _ in user_list])

    if len(user_list) > NUM_RECS:
        user_list = user_list[:NUM_RECS]

    if len(user_list) < NUM_RECS:
        needed = NUM_RECS - len(user_list)
        choices = [b for b in popular_pool if b not in seen]
        if len(choices) < needed:
            extra_choices = [b for b in popular_books_list if b not in seen and b not in choices]
            choices.extend(extra_choices)
        if len(choices) > 0:
            fill = random.sample(choices, min(needed, len(choices)))
        else:
            fill = []
        for title in fill:
            user_list.append((title, None))
            seen.add(title)

    while len(user_list) < NUM_RECS:
        user_list.append((None, None))

    for i, (title, score) in enumerate(user_list[:NUM_RECS], start=1):
        rows_out.append({
            "original_userId": uid,
            "rank": i,
            "book_title": title,
            "predicted_rating": score
        })

final_df = pd.DataFrame(rows_out)
print(f"Final assembled entries: {len(final_df)} rows.")

# ================== Convert to wide format ==================
print("Converting to wide format...")
grouped_list = final_df.sort_values(["original_userId", "rank"]).groupby("original_userId")["book_title"].apply(list).reset_index()

max_len = max(grouped_list["book_title"].apply(len).tolist()) if not grouped_list.empty else NUM_RECS
for i in range(max_len):
    grouped_list[f"book_{i+1}"] = grouped_list["book_title"].apply(lambda lst: lst[i] if i < len(lst) else None)
grouped_list = grouped_list.drop(columns=["book_title"])

order = sample_users_pd["original_userId"].tolist()
grouped_list["__order_idx"] = grouped_list["original_userId"].apply(lambda x: order.index(x) if x in order else math.inf)
grouped_list = grouped_list.sort_values("__order_idx").drop(columns="__order_idx").reset_index(drop=True)

sample_map_df = sample_users_pd[["userId", "original_userId"]]
grouped_list = grouped_list.merge(sample_map_df, on="original_userId", how="left")

cols = ["original_userId", "userId"] + [c for c in grouped_list.columns if c.startswith("book_")]
grouped_list = grouped_list[cols]

# ================== Export CSV ==================
out_path = os.path.join(BASE_PATH, "user_recommendations_full.csv")
grouped_list.to_csv(out_path, index=False, encoding="utf-8")
print(f"✅ Exported full recommendations to: {out_path}")

print("Preview (first 10 rows):")
print(grouped_list.head(10))

# ================== Book quality check ==================
print("📊 Checking average predicted ratings for recommended books...")
valid_recs = top_recs_pd.dropna(subset=["predicted_rating"])
good_recs = valid_recs[valid_recs["predicted_rating"] > 5]

book_quality = (
    good_recs.groupby("book_title")["predicted_rating"]
    .agg(["mean", "count"])
    .reset_index()
    .sort_values("mean", ascending=False)
)

book_quality.to_csv(f"{BASE_PATH}\\book_quality.csv", index=False)
print(f"✅ Saved book quality report to {BASE_PATH}\\book_quality.csv")

# ================== Wrap up ==================
spark.stop()
print(f"FINISHED in {time.time()-start_time:.1f}s")
