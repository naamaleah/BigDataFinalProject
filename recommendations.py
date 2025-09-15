# recommendations_fixed.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand
from pyspark.ml.recommendation import ALSModel
import os

BASE_PATH = r"C:\Users\shira\OneDrive\שולחן העבודה\Big Data\final"
DB_PATH = r"C:\Users\shira\OneDrive\שולחן העבודה\Big Data\final\books\books_no_outliers.db"

os.environ['PYSPARK_PYTHON'] = r"C:\Python\python.exe"
os.environ['PYSPARK_DRIVER_PYTHON'] = r"C:\Python\python.exe"
os.environ['HADOOP_HOME'] = r"C:\hadoop"
os.environ['PATH'] += r";C:\hadoop\bin"

spark = SparkSession.builder \
    .appName("BookRecommenderALS_Inference") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "700") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.jars", r"C:\spark\jars\sqlite-jdbc-3.45.2.0.jar") \
    .config("spark.driver.extraClassPath", r"C:\spark\jars\sqlite-jdbc-3.45.2.0.jar") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

def load_table(table_name, db_path=DB_PATH):
    return spark.read.format("jdbc") \
        .option("url", f"jdbc:sqlite:{db_path}") \
        .option("dbtable", table_name) \
        .option("driver", "org.sqlite.JDBC") \
        .load()

# Load data
ratings = load_table("Book_Ratings")
books = load_table("Books")
user_mapping = spark.read.parquet(f"{BASE_PATH}\\user_mapping")
item_mapping = spark.read.parquet(f"{BASE_PATH}\\item_mapping")
als_model = ALSModel.load(f"{BASE_PATH}\\als_model")

# Mappings
user_map = {int(row['userId']): row['original_userId'] for row in user_mapping.collect()}
item_map = {int(row['itemId']): row['original_itemId'] for row in item_mapping.collect()}
book_titles = {row['ISBN']: row.asDict().get('Book-Title', "") for row in books.collect()}

# Prepare ratings_df joined with mappings
clean_ratings = ratings.select(col("User-ID"), col("ISBN"), col("Book-Rating").cast("int").alias("rating")) \
                       .filter((col("Book-Rating") > 0) & (col("Book-Rating") <= 10)).na.drop()

ratings_df = clean_ratings.join(user_mapping, clean_ratings["User-ID"] == user_mapping["original_userId"]) \
                          .join(item_mapping, clean_ratings["ISBN"] == item_mapping["original_itemId"]) \
                          .select("userId", "itemId", "rating") \
                          .withColumn("userId", col("userId").cast("int")) \
                          .withColumn("itemId", col("itemId").cast("int"))

# Popularity fallback
from pyspark.sql.functions import avg, count
item_stats_df = ratings_df.groupBy("itemId").agg(
    avg("rating").alias("avg_rating"),
    count("rating").alias("num_ratings")
)
POP_MIN_RATINGS = 20
popular_items = [int(r["itemId"]) for r in item_stats_df.filter(col("num_ratings") >= POP_MIN_RATINGS) \
                 .orderBy(col("avg_rating").desc(), col("num_ratings").desc()).limit(500).collect()]

# Item metadata
books_by_isbn = {row['ISBN']: row.asDict() for row in books.collect()}
item_meta = {}
genre_column = None
for cand in ["genre", "category", "book_genre", "subject", "book-category"]:
    for c in books.columns:
        if cand in c.lower():
            genre_column = c
            break
    if genre_column:
        break

for row in item_mapping.collect():
    isbn = row['original_itemId']
    iid = int(row['itemId'])
    meta = books_by_isbn.get(isbn)
    if meta:
        if genre_column and meta.get(genre_column):
            genres = [g.strip().lower() for g in str(meta.get(genre_column)).replace("/", ",").split(",") if g.strip()]
            item_meta[iid] = {"genres": genres, "title": str(meta.get("Book-Title","")).lower()}
        else:
            title = str(meta.get("Book-Title","")).lower()
            tokens = [t for t in "".join(ch if ch.isalnum() else " " for ch in title).split() if len(t)>2]
            item_meta[iid] = {"genres": [], "title_tokens": tokens, "title": title}
    else:
        item_meta[iid] = {"genres": [], "title": ""}

# Helper scoring
item_stats = {int(r["itemId"]): (float(r["avg_rating"]), int(r["num_ratings"])) for r in item_stats_df.collect()}
avg_ratings = [v[0] for v in item_stats.values()] if item_stats else [0.0]
min_avg = float(min(avg_ratings))
max_avg = float(max(avg_ratings)) if max(avg_ratings) != min_avg else (min_avg + 1.0)
def norm_avg(itemId):
    st = item_stats.get(int(itemId))
    if not st:
        return 0.0
    a = float(st[0])
    return (a - min_avg) / (max_avg - min_avg)

HYBRID_ALPHA = 0.75
GENRE_BOOST = 0.12
def hybrid_score(iid, pred, user_top_genres=set(), user_title_tokens=set()):
    base = HYBRID_ALPHA * float(pred) + (1.0 - HYBRID_ALPHA) * norm_avg(iid)
    meta = item_meta.get(int(iid), {})
    genres = set(meta.get("genres", []))
    title_tokens = set(meta.get("title_tokens", [])) if meta.get("title_tokens") else set(meta.get("title","").split())
    boost = 0.0
    if genres and user_top_genres and len(genres & user_top_genres) > 0:
        boost += GENRE_BOOST
    if user_title_tokens and len(title_tokens & user_title_tokens) > 0:
        boost += GENRE_BOOST * 0.6
    return base + boost

def compute_user_profile(uid):
    rows = ratings_df.filter(col("userId") == uid).collect()
    genre_counts = {}
    title_tokens = {}
    for r in rows:
        iid = int(r['itemId'])
        score = float(r['rating'])
        meta = item_meta.get(iid, {})
        for g in meta.get("genres", []):
            genre_counts[g] = genre_counts.get(g, []) + [score] if isinstance(genre_counts.get(g, []), list) else [score]
            if not isinstance(genre_counts[g], list):
                genre_counts[g] = [genre_counts[g]]
        for t in meta.get("title_tokens", []):
            title_tokens[t] = title_tokens.get(t, 0) + 1
    user_top_genres = set()
    for g, arr in genre_counts.items():
        if isinstance(arr, list) and len(arr) > 0:
            if sum(arr)/len(arr) >= 7.0:
                user_top_genres.add(g)
    user_top_title_tokens = set([t for t,cnt in title_tokens.items() if cnt >= 1])
    return user_top_genres, user_top_title_tokens

# Generate recommendations with no repeats
import random

print("🔮 Generating diversified recommendations for 10 users:")
random_users = [int(r['userId']) for r in ratings_df.select("userId").distinct().orderBy(rand()).limit(10).collect()]

TOP_N = 5  # מספר המלצות לכל משתמש
FILLER_POOL = 100  # כמה ספרים פופולריים נשתמש כ-fallback

for uid in random_users:
    user_top_genres, user_top_title_tokens = compute_user_profile(uid)
    already_seen_iids = set([int(r['itemId']) for r in ratings_df.filter(col("userId") == uid).collect()])

    # ALS recommendations
    user_df = spark.createDataFrame([(uid,)], ["userId"])
    recs_rows = als_model.recommendForUserSubset(user_df, 20).collect()
    rec_itemids = []
    if recs_rows:
        raw_recs = [(int(t[0]), float(t[1])) for t in recs_rows[0]['recommendations']]
        scored = [(iid, hybrid_score(iid, pred, user_top_genres, user_top_title_tokens)) for iid, pred in raw_recs]
        scored = [x for x in scored if x[0] not in already_seen_iids]  # filter seen
        scored.sort(key=lambda x: x[1], reverse=True)
        rec_itemids = [iid for iid, sc in scored][:TOP_N]

    # Fallback with diversification if needed
    if len(rec_itemids) < TOP_N:
        filler_candidates = [iid for iid in popular_items[:FILLER_POOL] if iid not in rec_itemids and iid not in already_seen_iids]
        random.shuffle(filler_candidates)
        rec_itemids += filler_candidates[:TOP_N - len(rec_itemids)]

    # Map to book titles
    rec_books = [book_titles.get(item_map.get(iid), "<Unknown Title>") for iid in rec_itemids]

    # Actual rated books (for display)
    actual_rows = ratings_df.filter(col("userId") == uid).limit(2).collect()
    actual_titles = [book_titles.get(item_map.get(int(ar['itemId'])), "<Unknown Title>") for ar in actual_rows]

    print(f"User {user_map.get(uid, uid)} -> Recommended: {rec_books}, Actual (2 samples): {actual_titles}")

spark.stop()
