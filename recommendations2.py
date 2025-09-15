# evaluate_model.py
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
BASE_PATH = r"C:\Users\shira\OneDrive\שולחן העבודה\Big Data\final"
DB_PATH = r"C:\Users\shira\OneDrive\שולחן העבודה\Big Data\final\books\books_no_outliers.db"

# Spark session
spark = SparkSession.builder \
    .appName("ALS_Model_Evaluation") \
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
        .option("url", f"jdbc:sqlite:{DB_PATH}") \
        .option("dbtable", table_name) \
        .option("driver", "org.sqlite.JDBC") \
        .load()

# ========== MAIN ==========
try:
    print("📊 Loading Book_Ratings...")
    ratings = load_table("Book_Ratings")
    clean_ratings = ratings.select(
        col("User-ID"),
        col("ISBN"),
        col("Book-Rating").cast("int").alias("rating")
    ).filter((col("rating") > 0) & (col("rating") <= 10)).na.drop()

    print("🔄 Loading mappings...")
    user_mapping = spark.read.parquet(f"{BASE_PATH}\\user_mapping")
    item_mapping = spark.read.parquet(f"{BASE_PATH}\\item_mapping")

    ratings_df = clean_ratings \
        .join(user_mapping, clean_ratings["User-ID"] == user_mapping["original_userId"]) \
        .join(item_mapping, clean_ratings["ISBN"] == item_mapping["original_itemId"]) \
        .select("userId", "itemId", "rating")

    print(f"✅ Ratings rows: {ratings_df.count()}")

    # Split train/test (leave-1-out style)
    window = Window.partitionBy("userId").orderBy(rand(42))
    ranked = ratings_df.withColumn("rn", row_number().over(window))
    test = ranked.filter(col("rn") == 1).drop("rn")
    train = ranked.filter(col("rn") > 1).drop("rn")

    print(f"Split: train={train.count()}, test={test.count()}")

    # ========== Load trained ALS model ==========
    print("🔄 Loading saved ALS model...")
    final_model = ALSModel.load(f"{BASE_PATH}\\als_model")

    # ========== Predictions ==========
    predictions = final_model.transform(test)

    # ========== Regression metrics ==========
    evaluator_rmse = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    evaluator_mse = RegressionEvaluator(metricName="mse", labelCol="rating", predictionCol="prediction")
    evaluator_mae = RegressionEvaluator(metricName="mae", labelCol="rating", predictionCol="prediction")

    rmse = evaluator_rmse.evaluate(predictions)
    mse = evaluator_mse.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)

    print(f"📊 Regression → RMSE: {rmse:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")

    # ========== Binary AUC ==========
    test_binary = predictions.withColumn("label", when(col("rating") > 6, 1.0).otherwise(0.0)) \
                             .withColumnRenamed("prediction", "score") \
                             .withColumn("score", col("score").cast(DoubleType())) \
                             .dropna(subset=["score"])

    evaluator_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="score", metricName="areaUnderROC")
    auc = evaluator_auc.evaluate(test_binary)

    print(f"🟢 Binary classification AUC (rating > 6): {auc:.4f}")

    # ========== ROC curve ==========
    test_pd = test_binary.select("label", "score").toPandas()
    fpr, tpr, _ = roc_curve(test_pd["label"], test_pd["score"])
    roc_auc = sk_auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Binary: rating > 6)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

except Exception as e:
    import traceback
    print("❌ Error:", e)
    traceback.print_exc()

finally:
    spark.stop()
    print("✅ Spark stopped")
