# ALS_model3_fixed.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand, lit
from pyspark.sql.window import Window
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, col, rand
import time, os

# ================== CONFIG ==================
os.environ['PYSPARK_PYTHON'] = r"C:\Python\python.exe"
os.environ['PYSPARK_DRIVER_PYTHON'] = r"C:\Python\python.exe"
os.environ['PATH'] = r"C:\Python;" + os.environ.get('PATH','')

os.environ['HADOOP_HOME'] = r"C:\hadoop"
os.environ['PATH'] += r";C:\hadoop\bin"

DB_PATH = r"C:\Users\shira\OneDrive\שולחן העבודה\Big Data\final\books\books_no_outliers.db"
BASE_PATH = r"C:\Users\shira\OneDrive\שולחן העבודה\Big Data\final"

import sys

# Path to log file
LOG_FILE = os.path.join(BASE_PATH, "run_log.txt")

# Open log file for writing
log_file = open(LOG_FILE, "w", encoding="utf-8")

class Logger(object):
    """Custom logger to write output to multiple files/streams."""
    def __init__(self, *files):
        self.files = files
    def write(self, message):
        for f in self.files:
            f.write(message)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

# Redirect stdout/stderr to both console and log file
sys.stdout = Logger(sys.stdout, log_file)
sys.stderr = Logger(sys.stderr, log_file)

print(f"🔔 Logging all output also to {LOG_FILE}")


# Spark session
spark = SparkSession.builder \
    .appName("BookRecommenderALS_fixed") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "700") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.jars", r"C:\spark\jars\sqlite-jdbc-3.45.2.0.jar") \
    .config("spark.driver.extraClassPath", r"C:\spark\jars\sqlite-jdbc-3.45.2.0.jar") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.kryoserializer.buffer", "64m") \
    .config("spark.kryoserializer.buffer.max", "256m")\
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
# ============================================

def load_table(table_name):
    """Load a table from SQLite DB using Spark JDBC."""
    return spark.read.format("jdbc") \
        .option("url", f"jdbc:sqlite:{DB_PATH}") \
        .option("dbtable", table_name) \
        .option("driver", "org.sqlite.JDBC") \
        .load()

def create_user_item_mappings(ratings_df):
    """Create numeric user/item IDs (mapping to original IDs)."""
    user_window = Window.orderBy("User-ID")
    user_mapping = ratings_df.select("User-ID").distinct() \
        .withColumn("userId", row_number().over(user_window) - 1) \
        .select(col("User-ID").alias("original_userId"), col("userId"))

    item_window = Window.orderBy("ISBN")
    item_mapping = ratings_df.select("ISBN").distinct() \
        .withColumn("itemId", row_number().over(item_window) - 1) \
        .select(col("ISBN").alias("original_itemId"), col("itemId"))

    return user_mapping, item_mapping

def create_user_folds(ratings_df, k=3, seed=42, min_ratings=None):
    """
    Create per-user folds:
    - Users with >= min_ratings are split into k folds.
    - Users with < min_ratings are assigned fold = -1 (always in train).
    """
    user_counts = ratings_df.groupBy("userId").count()
    eligible_users = user_counts.filter(col("count") >= min_ratings).select("userId")

    eligible_ratings = ratings_df.join(eligible_users, on="userId", how="inner")

    ranked = eligible_ratings.withColumn(
        "rn",
        row_number().over(
            Window.partitionBy("userId").orderBy(rand(seed))
        )
    )

    return ranked.withColumn("fold", (col("rn") % k))

def custom_cv_train(ratings_df, k=3, min_ratings=None):
    """Perform custom cross-validation using per-user folds."""
    print(f"🔀 Creating per-user folds with k={k}, min_ratings={min_ratings or k} ...")
    ratings_with_folds = create_user_folds(ratings_df, k=k, seed=42, min_ratings=min_ratings)
    ratings_with_folds.cache()
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

    param_grid = [
        {"rank": 20, "regParam": 0.7},
    ]

    best_rmse = float("inf")
    best_params = {}
    summary = []

    for params in param_grid:
        rank = params["rank"]
        reg = params["regParam"]
        fold_rmses = []
        print(f"\n>> Evaluating params rank={rank}, reg={reg}")

        for fold in range(k):
            train = ratings_with_folds.filter(col("fold") != fold).drop("fold")
            test = ratings_with_folds.filter(col("fold") == fold).drop("fold")

            tbefore = test.count()
            if tbefore == 0:
                print(f"  ⚠️ Fold {fold} has 0 rows — skipping")
                continue

            # Cold-start filtering
            train_users = train.select("userId").distinct()
            train_items = train.select("itemId").distinct()
            test_filtered = test.join(train_users, "userId", "inner").join(train_items, "itemId", "inner")
            tafter = test_filtered.count()
            print(f"  Fold {fold}: test_before={tbefore}, test_after_filter={tafter}, train_rows={train.count()}")

            if tafter == 0:
                print(f"   ⚠️ Fold {fold} empty after filtering -> skip")
                continue

            als = ALS(
                userCol="userId", itemCol="itemId", ratingCol="rating",
                coldStartStrategy="drop", nonnegative=True,
                rank=10, regParam=0.7, maxIter=15, seed=42
            )
            model = als.fit(train)

            preds = model.transform(test_filtered)
            pcount = preds.count()
            if pcount == 0:
                print(f"   ⚠️ No predictions for fold {fold} -> skip")
                continue

            rmse = evaluator.evaluate(preds)
            fold_rmses.append(rmse)
            print(f"   ✅ Fold {fold} RMSE={rmse:.4f}")

        if len(fold_rmses) == 0:
            print(f"  ⚠️ No valid folds for params rank={rank}, reg={reg} -> skipped")
            continue

        avg_rmse = sum(fold_rmses) / len(fold_rmses)
        print(f"  >>> Params rank={rank}, reg={reg} -> Avg RMSE = {avg_rmse:.4f}")
        summary.append({"params": params, "avg_rmse": avg_rmse, "fold_rmses": fold_rmses})

        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_params = params.copy()

    ratings_with_folds.unpersist()
    return best_params, best_rmse, summary

def get_popular_books(ratings_df, item_map, top_n=100):
    """Return top-N popular books by average rating and count."""
    from pyspark.sql.functions import avg, count
    popular = ratings_df.groupBy("itemId").agg(
        avg("rating").alias("avg_rating"),
        count("rating").alias("num_ratings")
    ).filter(col("num_ratings") >= 20) \
     .orderBy(col("avg_rating").desc(), col("num_ratings").desc()) \
     .limit(top_n)
    return [row["itemId"] for row in popular.collect()]

# ================== MAIN ==================
try:
    print("📊 Loading Book_Ratings...")
    ratings = load_table("Book_Ratings")
    clean_ratings = ratings.select(
        col("User-ID"),
        col("ISBN"),
        col("Book-Rating").cast("int").alias("rating")
    ).filter((col("rating") > 0) & (col("rating") <= 10)).na.drop()

    print("🔄 Creating mappings (user/item)...")
    user_mapping, item_mapping = create_user_item_mappings(clean_ratings)
    ratings_df = clean_ratings \
                .join(user_mapping, clean_ratings["User-ID"] == user_mapping["original_userId"]) \
                .join(item_mapping, clean_ratings["ISBN"] == item_mapping["original_itemId"]) \
                .select("userId", "itemId", "rating")

    ratings_df.cache()
    print(f"✅ Ratings rows: {ratings_df.count()}, Unique users: {user_mapping.count()}, Unique items: {item_mapping.count()}")

    k = 5
    best_params, best_rmse, summary = custom_cv_train(ratings_df, k=k, min_ratings=7)
    if not best_params:
        print("⚠️ WARNING: No valid parameters found in CV. Falling back to defaults (rank=50, regParam=0.1).")

    best_params = {"rank": 30, "regParam": 0.2}
    print("\n🏆 Best params:", best_params, " Best CV RMSE:", best_rmse)

    # Final leave-one-out split
    window = Window.partitionBy("userId").orderBy(rand(42))
    ranked = ratings_df.withColumn("rn", row_number().over(window))
    test = ranked.filter(col("rn") == 1).drop("rn")
    train = ranked.filter(col("rn") > 1).drop("rn")

    print(f"Final split: train={train.count()}, test={test.count()}")

    # Train final model
    final_als = ALS(
        userCol="userId",
        itemCol="itemId",
        ratingCol="rating",
        coldStartStrategy="drop",
        nonnegative=True,
        rank=best_params["rank"],
        regParam=best_params["regParam"],
        maxIter=15, seed=42
    )

    final_model = final_als.fit(train)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse_test = evaluator.evaluate(final_model.transform(test))
    rmse_train = evaluator.evaluate(final_model.transform(train))

    print("\n📈 Final Model Performance:")
    print(f" Train RMSE: {rmse_train:.4f}")
    print(f" Test RMSE: {rmse_test:.4f}")

    # Save mappings & model
    user_mapping.write.mode("overwrite").parquet(f"{BASE_PATH}\\user_mapping")
    item_mapping.write.mode("overwrite").parquet(f"{BASE_PATH}\\item_mapping")
    final_model.write().overwrite().save(f"{BASE_PATH}\\als_model")
    print("✅ Mapped & model saved")

    # Build lookup dicts for printing
    user_map = {row['userId']: row['original_userId'] for row in user_mapping.collect()}
    item_map = {row['itemId']: row['original_itemId'] for row in item_mapping.collect()}

    sample = final_model.recommendForAllUsers(5).limit(5).collect()
    print("\n🔮 Sample recommendations:")

    for r in sample:
        uid = r['userId']
        books = [item_map[i[0]] for i in r['recommendations']]
        print(f"User {user_map[uid]} -> {books}")

except Exception as e:
    import traceback
    print("❌ Error:", e)

finally:
    spark.stop()
    print("✅ Spark stopped")
