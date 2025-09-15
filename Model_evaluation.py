# evaluate_model_custom.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, rand, row_number
from pyspark.sql.types import DoubleType
from pyspark.ml.recommendation import ALSModel
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator
from pyspark.sql.window import Window

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sk_auc
import os

# ================== CONFIG ==================
BASE_DIR = r"C:\Users\shira\OneDrive\שולחן העבודה\Big Data\final"
DB_FILE = r"C:\Users\shira\OneDrive\שולחן העבודה\Big Data\final\books\books_no_outliers.db"

# Spark session
spark = SparkSession.builder \
    .appName("Custom_ALS_Evaluation") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.jars", r"C:\spark\jars\sqlite-jdbc-3.45.2.0.jar") \
    .config("spark.driver.extraClassPath", r"C:\spark\jars\sqlite-jdbc-3.45.2.0.jar") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# ========== Load tables ==========
def load_table(table_name):
    return spark.read.format("jdbc") \
        .option("url", f"jdbc:sqlite:{DB_FILE}") \
        .option("dbtable", table_name) \
        .option("driver", "org.sqlite.JDBC") \
        .load()

try:
    print("📊 Loading ratings table...")
    ratings_raw = load_table("Book_Ratings")
    ratings_clean = ratings_raw.select(
        col("User-ID"),
        col("ISBN"),
        col("Book-Rating").cast("int").alias("rating")
    ).filter((col("rating") > 0) & (col("rating") <= 10)).na.drop()

    print("🔄 Loading user/item mappings...")
    users_map = spark.read.parquet(f"{BASE_DIR}\\user_mapping")
    items_map = spark.read.parquet(f"{BASE_DIR}\\item_mapping")

    ratings_mapped = ratings_clean \
        .join(users_map, ratings_clean["User-ID"] == users_map["original_userId"]) \
        .join(items_map, ratings_clean["ISBN"] == items_map["original_itemId"]) \
        .select("userId", "itemId", "rating")

    print(f"✅ Ratings rows after mapping: {ratings_mapped.count()}")

    # ========== Complex Train/Test Split ==========
    N_test_per_user = 2
    window_spec = Window.partitionBy("userId").orderBy(rand(42))
    ranked_df = ratings_mapped.withColumn("rn", row_number().over(window_spec))

    test_set = ranked_df.filter(col("rn") <= N_test_per_user).drop("rn")
    train_set = ranked_df.filter(col("rn") > N_test_per_user).drop("rn")

    print(f"🔀 Train/Test split → train: {train_set.count()}, test: {test_set.count()}")

    # ========== Load trained ALS model ==========
    print("🔄 Loading ALS model...")
    als_model = ALSModel.load(f"{BASE_DIR}\\als_model")

    # ========== Predictions ==========
    predictions_df = als_model.transform(test_set)

    # ========== Regression metric: RMSE ==========
    rmse_evaluator = RegressionEvaluator(
        metricName="rmse", labelCol="rating", predictionCol="prediction"
    )
    rmse_value = rmse_evaluator.evaluate(predictions_df)
    print(f"📊 RMSE: {rmse_value:.4f}")

    # ========== Binary metric: AUC-ROC ==========
    binary_df = predictions_df.withColumn(
        "label", when(col("rating") > 5, 1.0).otherwise(0.0)
    ).withColumnRenamed("prediction", "score") \
     .withColumn("score", col("score").cast(DoubleType())) \
     .dropna(subset=["score"])

    auc_evaluator = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="score", metricName="areaUnderROC"
    )
    auc_value = auc_evaluator.evaluate(binary_df)
    print(f"🟢 Binary AUC (rating > 5): {auc_value:.4f}")

    # ========== ROC curve ==========
    roc_pd = binary_df.select("label", "score").toPandas()
    fpr, tpr, _ = roc_curve(roc_pd["label"], roc_pd["score"])
    roc_auc = sk_auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Binary: rating > 5)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(f"{BASE_DIR}\\roc_curve.png", dpi=300)  
    plt.show()


except Exception as e:
    import traceback
    print("❌ Error:", e)
    traceback.print_exc()

finally:
    spark.stop()
    print("✅ Spark stopped")
