from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import Normalizer
from pyspark.ml.linalg import DenseVector
import pandas as pd
import numpy as np

# ================== CONFIG ==================
BASE_PATH = r"C:\Users\shira\OneDrive\שולחן העבודה\Big Data\final"
DB_PATH = r"C:\Users\shira\OneDrive\שולחן העבודה\Big Data\final\books\books_no_outliers.db"

spark = SparkSession.builder \
    .appName("Similarity_Matrix") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.jars", r"C:\spark\jars\sqlite-jdbc-3.45.2.0.jar") \
    .config("spark.driver.extraClassPath", r"C:\spark\jars\sqlite-jdbc-3.45.2.0.jar") \
    .getOrCreate()

# ================== Load ratings ==================
ratings = spark.read.format("jdbc") \
    .option("url", f"jdbc:sqlite:{DB_PATH}") \
    .option("dbtable", "Book_Ratings") \
    .option("driver", "org.sqlite.JDBC") \
    .load()

ratings = ratings.select(
    col("User-ID").alias("userId"),
    col("ISBN").alias("bookId"),
    col("Book-Rating").cast("float").alias("rating")
).filter(col("rating") > 0)

# ================== Create user-item rating matrix ==================
# pivot table: rows = books, cols = users
spark.conf.set("spark.sql.pivotMaxValues", 5000000)
ratings_pivot = ratings.groupBy("bookId").pivot("userId").avg("rating").fillna(0)

# ================== Convert to pandas (dense matrix) ==================
ratings_pd = ratings_pivot.toPandas().set_index("bookId")
rating_matrix = ratings_pd.values  # books x users

# ================== Compute cosine similarity between books ==================
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(rating_matrix)
similarity_df = pd.DataFrame(similarity_matrix, index=ratings_pd.index, columns=ratings_pd.index)

# ================== Save similarity matrix ==================
similarity_df.to_csv(f"{BASE_PATH}\\book_similarity_matrix.csv", encoding="utf-8")
print(f"✅ Similarity matrix saved to {BASE_PATH}\\book_similarity_matrix.csv")
print(similarity_df.head())

spark.stop()
