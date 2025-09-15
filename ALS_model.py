from pyspark.sql import SparkSession
from pyspark.sql.functions import col, desc, monotonically_increasing_id, row_number
from pyspark.sql.window import Window
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import time
import os

# Environment setup
os.environ['HADOOP_HOME'] = r"C:\hadoop"
os.environ['PATH'] += r";C:\hadoop\bin"

# Create Spark Session with optimized configuration
spark = SparkSession.builder \
    .appName("BookRecommenderALS") \
    .master("local[2]") \
    .config("spark.driver.memory", "2g")\
    .config("spark.executor.memory", "2g") \
    .config("spark.executor.extraJavaOptions", "-Xss32m") \
    .config("spark.driver.extraJavaOptions", "-Xss32m") \
    .config("spark.sql.shuffle.partitions", "50")\
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")\
    .config("spark.jars", r"C:\spark\jars\sqlite-jdbc-3.45.2.0.jar") \
    .config("spark.driver.extraClassPath", r"C:\spark\jars\sqlite-jdbc-3.45.2.0.jar") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
    .config("spark.sql.adaptive.skewJoin.enabled", "true") \
    .getOrCreate()

# Set log level to show INFO for progress tracking
spark.sparkContext.setLogLevel("WARN")

print("✅ Spark Session started successfully")

# Database path
db_path = r"C:\Users\shira\OneDrive\שולחן העבודה\Big Data\final\books\books_no_outliers.db"


def load_table(table_name):
    """Load table from SQLite database"""
    return spark.read \
        .format("jdbc") \
        .option("url", f"jdbc:sqlite:{db_path}") \
        .option("dbtable", table_name) \
        .option("driver", "org.sqlite.JDBC") \
        .load()


def create_user_item_mappings(ratings_df):
    """Create efficient user and item mappings using window functions"""
    print("🔄 Creating user mapping...")
    # Create user mapping
    user_window = Window.orderBy("User-ID")
    user_mapping = ratings_df.select("User-ID").distinct() \
        .withColumn("userId", row_number().over(user_window) - 1) \
        .select(col("User-ID").alias("original_userId"), col("userId"))

    print("🔄 Creating item mapping...")
    # Create item mapping
    item_window = Window.orderBy("ISBN")
    item_mapping = ratings_df.select("ISBN").distinct() \
        .withColumn("itemId", row_number().over(item_window) - 1) \
        .select(col("ISBN").alias("original_itemId"), col("itemId"))

    return user_mapping, item_mapping


def train_als_model(ratings_df):
    """Train ALS model with cross validation"""
    print("\n🤖 Setting up ALS model...")

    # Configure ALS
    als = ALS(
        userCol="userId",
        itemCol="itemId",
        ratingCol="rating",
        coldStartStrategy="drop",
        nonnegative=True,
        implicitPrefs=False
    )

    # Parameter grid for cross validation
    param_grid = ParamGridBuilder() \
        .addGrid(als.rank, [10, 50, 100]) \
        .addGrid(als.regParam, [0.01, 0.1, 1.0]) \
        .addGrid(als.alpha, [1.0, 10.0, 40.0]) \
        .build()

    print(f"Parameter combinations: {len(param_grid)}")

    # Regression evaluator
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )

    # Cross validator
    cross_validator = CrossValidator(
        estimator=als,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        # numFolds=5,
        seed=42,
        parallelism=5  # Reduce parallelism to avoid Python worker issues
    )

    return cross_validator, evaluator


try:
    # 1. Load Book_Ratings table
    print("📊 Loading Book_Ratings table...")
    ratings = load_table("Book_Ratings")
    print(f"Total ratings: {ratings.count()}")

    # 2. Convert to ratings_df with data cleaning
    print("\n🔧 Preparing data for ALS...")

    # Clean and prepare ratings data
    clean_ratings = ratings.select(
        col("User-ID"),
        col("ISBN"),
        col("Book-Rating").cast("int").alias("rating")
    ).filter(
        (col("rating") > 0) & (col("rating") <= 10)  # Only positive ratings
    ).na.drop()

    print(f"Clean ratings count: {clean_ratings.count()}")

    # Cache the clean ratings for better performance
    clean_ratings.cache()

    # Create mappings efficiently
    user_mapping, item_mapping = create_user_item_mappings(clean_ratings)

    # Cache mappings
    user_mapping.cache()
    item_mapping.cache()

    print(f"Unique users: {user_mapping.count()}")
    print(f"Unique items: {item_mapping.count()}")

    # Join with mappings to create final ratings_df
    print("🔗 Creating final ratings DataFrame...")
    ratings_df = clean_ratings \
        .join(user_mapping, clean_ratings["User-ID"] == user_mapping["original_userId"]) \
        .join(item_mapping, clean_ratings["ISBN"] == item_mapping["original_itemId"]) \
        .select("userId", "itemId", "rating")

    # Cache final ratings dataframe with better storage level
    ratings_df.persist()

    print("✅ Data prepared successfully:")
    ratings_df.show(10)
    print(f"Final ratings count: {ratings_df.count()}")

    # Basic statistics
    print("\n📈 Rating statistics:")
    ratings_df.describe("rating").show()

    print("\n📊 Rating distribution:")
    ratings_df.groupBy("rating").count().orderBy("rating").show()

    # 3. Split data for training and testing
    print("\n🔀 Splitting data into train/test...")
    train_data, test_data = ratings_df.randomSplit([0.8, 0.2], seed=42)

    # Use lighter caching for train/test data
    train_data.persist()
    test_data.persist()

    print(f"Train: {train_data.count()}, Test: {test_data.count()}")

    # Clear cache for intermediate data to free memory
    clean_ratings.unpersist()
    user_mapping.unpersist()
    item_mapping.unpersist()

    # 4. Train ALS with 10-fold cross validation
    cross_validator, evaluator = train_als_model(ratings_df)

    print("\n🎯 Starting ALS training with Cross Validation...")
    start_time = time.time()

    # Fit the cross validator
    cv_model = cross_validator.fit(train_data)

    training_time = time.time() - start_time
    print(f"⏰ Training completed in: {training_time:.2f} seconds")

    # 5. Select best model and evaluate
    print("\n📊 Evaluating best model...")

    # Get best model
    best_model = cv_model.bestModel

    # Get best parameters - use the correct attribute access
    best_params = {
        'rank': best_model.rank,
        'regParam': best_model.regParam,
        'alpha': best_model.alpha
    }

    print("🏆 Best parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    # Make predictions on test data
    test_predictions = cv_model.transform(test_data)
    train_predictions = cv_model.transform(train_data)

    # Calculate RMSE
    test_rmse = evaluator.evaluate(test_predictions)
    train_rmse = evaluator.evaluate(train_predictions)

    print(f"\n📈 Model Performance:")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  Train RMSE: {train_rmse:.4f}")

    # Show sample predictions
    print("\n🔮 Sample predictions:")
    test_predictions.select("userId", "itemId", "rating", "prediction") \
        .orderBy("userId") \
        .limit(10) \
        .show()

    # Generate recommendations for sample users
    print("\n💡 Sample user recommendations...")
    sample_users = ratings_df.select("userId").distinct().limit(3).collect()
    sample_user_ids = [row["userId"] for row in sample_users]

    if sample_user_ids:
        user_subset = spark.createDataFrame([(uid,) for uid in sample_user_ids], ["userId"])
        user_recommendations = best_model.recommendForUserSubset(user_subset, 5)

        print("🎯 Top 5 recommendations per user:")
        user_recommendations.show(truncate=False)

        # Show detailed recommendations for first user
        if len(sample_user_ids) > 0:
            first_user = sample_user_ids[0]
            print(f"\n📚 Detailed recommendations for User {first_user}:")

            user_recs = user_recommendations.filter(col("userId") == first_user).collect()
            if user_recs:
                recommendations = user_recs[0]['recommendations']

                # Get original user ID
                original_user = user_mapping.filter(col("userId") == first_user) \
                    .select("original_userId").collect()[0]["original_userId"]
                print(f"Original User ID: {original_user}")

                for i, rec in enumerate(recommendations, 1):
                    mapped_item_id = rec['itemId']
                    rating_pred = rec['rating']

                    # Get original ISBN
                    original_isbn = item_mapping.filter(col("itemId") == mapped_item_id) \
                        .select("original_itemId").collect()

                    if original_isbn:
                        isbn = original_isbn[0]["original_itemId"]
                        print(f"  {i}. ISBN: {isbn}, Predicted Rating: {rating_pred:.2f}")
                    else:
                        print(f"  {i}. Item ID: {mapped_item_id}, Predicted Rating: {rating_pred:.2f}")

    # Model coverage statistics
    total_items = item_mapping.count()
    predicted_items = test_predictions.select("itemId").distinct().count()
    coverage = (predicted_items / total_items) * 100
    print(f"\n📊 Model Coverage: {coverage:.2f}% ({predicted_items}/{total_items} items)")

    # Save mappings and model
    print("\n💾 Saving mappings and model...")
    base_path = r"C:\Users\shira\OneDrive\שולחן העבודה\Big Data\final"

    user_mapping_path = f"{base_path}\\user_mapping"
    item_mapping_path = f"{base_path}\\item_mapping"
    model_path = f"{base_path}\\als_model"

    # Save with overwrite mode
    user_mapping.write.mode("overwrite").parquet(user_mapping_path)
    item_mapping.write.mode("overwrite").parquet(item_mapping_path)
    best_model.write().overwrite().save(model_path)

    print("✅ Mappings and model saved successfully!")

    # Summary
    print(f"\n🎊 Training Complete!")
    print(f"📋 Summary:")
    print(f"  • Total ratings processed: {ratings_df.count()}")
    print(f"  • Best RMSE: {test_rmse:.4f}")
    print(f"  • Training time: {training_time:.2f} seconds")
    print(f"  • Model coverage: {coverage:.1f}%")

except Exception as e:
    print(f"❌ Error occurred: {str(e)}")
    import traceback

    traceback.print_exc()

finally:
    print("\n🛑 Stopping Spark Session...")
    spark.stop()
    print("✅ Spark Session stopped successfully")