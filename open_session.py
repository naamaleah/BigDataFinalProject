from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# === שלב 1: פתיחת Spark Session ===
spark = SparkSession.builder \
    .appName("BookRecommenderALS") \
    .config("spark.driver.extraClassPath", "sqlite-jdbc-3.41.2.1.jar") \
    .getOrCreate()

print("✅ Spark Session started")

# === שלב 2: טעינת הטבלאות מה-DB ===
db_path = "books.db"

ratings = spark.read.format("jdbc") \
    .option("url", f"jdbc:sqlite:{db_path}") \
    .option("dbtable", "books_ratings") \
    .load()

books = spark.read.format("jdbc") \
    .option("url", f"jdbc:sqlite:{db_path}") \
    .option("dbtable", "books") \
    .load()

# === שלב 3: עיבוד הדאטה ל-ALS ===
ratings_df = ratings.select(
    col("user_id").cast("int"),
    col("book_id").cast("int"),
    col("rating").cast("float")
).na.drop()

# === שלב 4: בניית ALS ===
als = ALS(
    userCol="user_id",
    itemCol="book_id",
    ratingCol="rating",
    coldStartStrategy="drop"
)

# === שלב 5: בניית Grid של פרמטרים ל-CV ===
paramGrid = ParamGridBuilder() \
    .addGrid(als.rank, [5, 10, 15]) \
    .addGrid(als.maxIter, [5, 10]) \
    .addGrid(als.regParam, [0.05, 0.1]) \
    .build()

evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)

# === שלב 6: Cross Validation עם 10 folds ===
cv = CrossValidator(
    estimator=als,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=10,
    parallelism=2  # להריץ במקביל אם יש ליבה פנויה
)

# === שלב 7: אימון ===
print("🚀 Training ALS model with 10-fold CV...")
cvModel = cv.fit(ratings_df)

# === שלב 8: בדיקה על כל הדאטה (רק כדי למדוד RMSE) ===
predictions = cvModel.transform(ratings_df)
rmse = evaluator.evaluate(predictions)
print(f"✅ Best Model RMSE = {rmse}")

# === שלב 9: הפקת המלצות למשתמש מסוים ===
bestModel = cvModel.bestModel
user_id = 60244  # אפשר לשנות למשתמש שאת רוצה
userRecs = bestModel.recommendForAllUsers(5)
userRecs.filter(col("user_id") == user_id).show(truncate=False)

# === שלב 10: הצגת שמות הספרים המומלצים ===
from pyspark.sql.functions import explode

recs = userRecs.filter(col("user_id") == user_id) \
    .select("user_id", explode("recommendations").alias("rec")) \
    .select("user_id", col("rec.book_id").alias("book_id"), col("rec.rating").alias("predicted_rating"))

recs = recs.join(books, on="book_id", how="left")
recs.show(truncate=False)

input("Press Enter to stop Spark...")
spark.stop()

