from pyspark.sql import SparkSession
from pyspark.sql.functions import col, desc, monotonically_increasing_id, row_number, rand, broadcast
from pyspark.sql.window import Window
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import time
import os
import concurrent.futures
import threading

# Environment setup
os.environ['HADOOP_HOME'] = r"C:\hadoop"
os.environ['PATH'] += r";C:\hadoop\bin"

# Create Spark Session with optimized configuration for higher parallelism
spark = SparkSession.builder \
    .appName("BookRecommenderALS") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.extraJavaOptions", "-Xss32m") \
    .config("spark.driver.extraJavaOptions", "-Xss32m") \
    .config("spark.sql.shuffle.partitions", "100") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.jars", r"C:\spark\jars\sqlite-jdbc-3.45.2.0.jar") \
    .config("spark.driver.extraClassPath", r"C:\spark\jars\sqlite-jdbc-3.45.2.0.jar") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
    .config("spark.sql.adaptive.skewJoin.enabled", "true") \
    .config("spark.broadcast.compress", "true") \
    .config("spark.rdd.compress", "true") \
    .config("spark.shuffle.compress", "true") \
    .config("spark.shuffle.spill.compress", "true") \
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


def create_cold_start_safe_folds(ratings_df, num_folds=5, seed=42):
    """
    Create cross-validation folds that avoid cold-start problems.
    Uses improved distribution to minimize cold users/items.
    """
    print(f"🔀 Creating {num_folds} cold-start safe folds...")

    # Get user and item statistics
    user_counts = ratings_df.groupBy("userId").count().withColumnRenamed("count", "user_rating_count")
    item_counts = ratings_df.groupBy("itemId").count().withColumnRenamed("count", "item_rating_count")

    # Join with counts
    ratings_with_counts = ratings_df \
        .join(broadcast(user_counts), "userId") \
        .join(broadcast(item_counts), "itemId")

    # Add random column but use user/item info to reduce cold-start
    ratings_with_rand = ratings_with_counts.withColumn("rand", rand(seed))

    # IMPROVED COLD-START PREVENTION: Mix user and item IDs into fold assignment
    ratings_with_fold_assignment = ratings_with_rand \
        .withColumn("fold",
                    ((col("rand") + col("userId") * 0.1 + col("itemId") * 0.1) * num_folds).cast("int") % num_folds
                    ) \
        .drop("rand", "user_rating_count", "item_rating_count")

    print("✅ Cold-start safe folds created")
    return ratings_with_fold_assignment


def train_als_with_custom_cv(ratings_df_with_folds, num_folds=5, parallelism=7):
    """Train ALS model with custom cross validation and parallelism"""
    print(f"\n🤖 Setting up ALS model with custom CV (parallelism={parallelism})...")

    # Smaller parameter grid as requested (2x2x2 = 8 combinations)
    param_grid = [
        {'rank': 50, 'regParam': 0.01, 'alpha': 1.0},
        {'rank': 50, 'regParam': 0.1, 'alpha': 1.0},
        {'rank': 100, 'regParam': 0.01, 'alpha': 1.0},
        {'rank': 100, 'regParam': 0.1, 'alpha': 1.0},
        {'rank': 50, 'regParam': 0.01, 'alpha': 10.0},
        {'rank': 50, 'regParam': 0.1, 'alpha': 10.0},
        {'rank': 100, 'regParam': 0.01, 'alpha': 10.0},
        {'rank': 100, 'regParam': 0.1, 'alpha': 10.0}
    ]

    print(f"Parameter combinations: {len(param_grid)}")
    print(f"Total models to train: {len(param_grid) * num_folds}")

    # Create all parameter-fold combinations for parallel processing
    tasks = []
    for param_idx, params in enumerate(param_grid):
        for fold in range(num_folds):
            tasks.append((param_idx, params, fold))

    print(f"🚀 Using parallelism={parallelism} for {len(tasks)} total tasks")

    # Evaluator
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )

    def train_single_model(task_info):
        """Train a single model for one parameter combination and fold"""
        param_idx, params, fold = task_info

        try:
            # Create train and validation sets for this fold
            train_fold = ratings_df_with_folds.filter(col("fold") != fold).drop("fold")
            val_fold = ratings_df_with_folds.filter(col("fold") == fold).drop("fold")

            # COLD-START PROTECTION: Filter validation to only include seen users/items
            train_users = train_fold.select("userId").distinct()
            train_items = train_fold.select("itemId").distinct()

            # Only evaluate on users/items that were in training (prevents cold-start)
            val_fold_filtered = val_fold \
                .join(train_users, "userId", "inner") \
                .join(train_items, "itemId", "inner")

            val_count = val_fold_filtered.count()
            if val_count == 0:
                return (param_idx, fold, None, f"No valid validation samples")

            # Configure ALS for this fold
            als = ALS(
                userCol="userId",
                itemCol="itemId",
                ratingCol="rating",
                coldStartStrategy="drop",
                nonnegative=True,
                implicitPrefs=False,
                rank=params['rank'],
                regParam=params['regParam'],
                alpha=params['alpha'],
                maxIter=10,
                seed=42 + fold  # Different seed per fold
            )

            # Train model
            model = als.fit(train_fold)

            # Make predictions on filtered validation set
            predictions = model.transform(val_fold_filtered)

            # Evaluate RMSE
            rmse = evaluator.evaluate(predictions)

            return (param_idx, fold, rmse, "Success")

        except Exception as e:
            return (param_idx, fold, None, str(e))

    # IMPLEMENT PARALLELISM using ThreadPool
    print(f"\n🎯 Starting parallel training with {parallelism} workers...")
    start_time = time.time()

    results = {}
    completed_tasks = 0
    lock = threading.Lock()

    def update_progress():
        nonlocal completed_tasks
        with lock:
            completed_tasks += 1
            progress = (completed_tasks / len(tasks)) * 100
            print(f"   Progress: {completed_tasks}/{len(tasks)} ({progress:.1f}%)")

    # Use ThreadPoolExecutor for parallelism
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(train_single_model, task): task for task in tasks}

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                param_idx, fold, rmse, status = future.result()

                if param_idx not in results:
                    results[param_idx] = {'params': param_grid[param_idx], 'fold_rmses': [], 'errors': []}

                if rmse is not None:
                    results[param_idx]['fold_rmses'].append(rmse)
                else:
                    results[param_idx]['errors'].append(f"Fold {fold}: {status}")

                update_progress()

            except Exception as e:
                param_idx, params, fold = task
                print(f"   ❌ Task failed: param_idx={param_idx}, fold={fold}, error={e}")
                update_progress()

    training_time = time.time() - start_time
    print(f"\n⏰ Parallel CV completed in: {training_time:.2f} seconds")

    # Process results to find best parameters
    best_rmse = float('inf')
    best_params = None
    all_results = []

    print(f"\n📊 CV Results Summary:")
    for param_idx, result in results.items():
        if result['fold_rmses']:
            avg_rmse = sum(result['fold_rmses']) / len(result['fold_rmses'])
            params = result['params']

            print(f"   Params {param_idx}: rank={params['rank']}, reg={params['regParam']}, alpha={params['alpha']}")
            print(f"   RMSE: {avg_rmse:.4f} (±{max(result['fold_rmses']) - min(result['fold_rmses']):.4f})")

            all_results.append({
                'params': params,
                'avg_rmse': avg_rmse,
                'fold_rmses': result['fold_rmses']
            })

            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                best_params = params
                print(f"   🏆 New best!")

        if result['errors']:
            print(f"   ⚠️ Errors: {len(result['errors'])}")

    # Train final model with best parameters on full dataset
    best_model = None
    if best_params:
        print(f"\n🎯 Training final model with best parameters...")
        print(f"Best parameters: {best_params}")

        final_als = ALS(
            userCol="userId",
            itemCol="itemId",
            ratingCol="rating",
            coldStartStrategy="drop",
            nonnegative=True,
            implicitPrefs=False,
            rank=best_params['rank'],
            regParam=best_params['regParam'],
            alpha=best_params['alpha'],
            maxIter=15,  # More iterations for final model
            seed=42
        )

        # Train on full dataset (without fold column)
        full_training_data = ratings_df_with_folds.drop("fold")
        best_model = final_als.fit(full_training_data)

    return best_model, best_params, best_rmse, all_results


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

    # Cache final ratings dataframe
    ratings_df.persist()

    print("✅ Data prepared successfully:")
    ratings_df.show(10)
    print(f"Final ratings count: {ratings_df.count()}")

    # Basic statistics
    print("\n📈 Rating statistics:")
    ratings_df.describe("rating").show()

    print("\n📊 Rating distribution:")
    ratings_df.groupBy("rating").count().orderBy("rating").show()

    # Clear cache for intermediate data to free memory
    clean_ratings.unpersist()

    # 3. Create cold-start safe folds
    ratings_with_folds = create_cold_start_safe_folds(ratings_df, num_folds=5, seed=42)
    ratings_with_folds.cache()

    # 4. Train ALS with custom cross validation - NOW WITH CORRECT PARAMETERS
    best_model, best_params, best_rmse, all_results = train_als_with_custom_cv(
        ratings_with_folds, num_folds=5, parallelism=7
    )

    # 5. Final evaluation on held-out test set
    print("\n📊 Final model evaluation...")

    # Create a separate test set for final evaluation
    train_final, test_final = ratings_df.randomSplit([0.8, 0.2], seed=42)
    train_final.cache()
    test_final.cache()

    if best_model:
        print("🏆 Best parameters found:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")

        # Make predictions on test set
        test_predictions = best_model.transform(test_final)
        train_predictions = best_model.transform(train_final)

        # Calculate RMSE
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
        test_rmse = evaluator.evaluate(test_predictions)
        train_rmse = evaluator.evaluate(train_predictions)

        print(f"\n📈 Final Model Performance:")
        print(f"  CV Best RMSE: {best_rmse:.4f}")
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
                    original_user_result = user_mapping.filter(col("userId") == first_user) \
                        .select("original_userId").collect()

                    if original_user_result:
                        original_user = original_user_result[0]["original_userId"]
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
        print(f"  • Best CV RMSE: {best_rmse:.4f}")
        print(f"  • Final test RMSE: {test_rmse:.4f}")
        print(f"  • Model coverage: {coverage:.1f}%")
        print(f"  • Parameters tested: {len(all_results)}")

    else:
        print("❌ No valid model found during cross-validation")

except Exception as e:
    print(f"❌ Error occurred: {str(e)}")
    import traceback

    traceback.print_exc()

finally:
    print("\n🛑 Stopping Spark Session...")
    spark.stop()
    print("✅ Spark Session stopped successfully")