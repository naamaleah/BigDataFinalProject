from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import os

# הגדרת משתני סביבה
os.environ['HADOOP_HOME'] = r"C:\hadoop"
os.environ['PATH'] += r";C:\hadoop\bin"

# === יצירת Spark Session ===
spark = SparkSession.builder \
    .appName("BookRecommenderALS") \
    .master("local[2]") \
    .config("spark.jars", r"C:\spark\jars\sqlite-jdbc-3.45.2.0.jar") \
    .config("spark.driver.extraClassPath", r"C:\spark\jars\sqlite-jdbc-3.45.2.0.jar") \
    .getOrCreate()

print("✅ Spark Session started successfully")

# === נתיב לקובץ DB ===
db_path = r"C:\Users\shira\OneDrive\שולחן העבודה\Big Data\final\books\books_no_outliers.db"


# === פונקציה לטעינת טבלה ===
def load_table(table_name):
    return spark.read \
        .format("jdbc") \
        .option("url", f"jdbc:sqlite:{db_path}") \
        .option("dbtable", table_name) \
        .option("driver", "org.sqlite.JDBC") \
        .load()


try:
    # === טעינת טבלאות מה-DB ===
    print("📚 טוען טבלת ספרים...")
    books = load_table("Books")

    print("👥 טוען טבלת משתמשים...")
    users = load_table("Users")

    print("⭐ טוען טבלת דירוגים...")
    ratings = load_table("Book_Ratings")

    # === הצגת מידע על הטבלאות ===
    print(f"\n📊 סטטיסטיקות:")
    print(f"מספר ספרים: {books.count()}")
    print(f"מספר משתמשים: {users.count()}")
    print(f"מספר דירוגים: {ratings.count()}")

    # === הצגת דוגמאות מהדאטה ===
    print("\n📚 דוגמאות מטבלת ספרים:")
    books.show(5, truncate=False)

    print("\n👥 דוגמאות מטבלת משתמשים:")
    users.show(5, truncate=False)

    print("\n⭐ דוגמאות מטבלת דירוגים:")
    ratings.show(5, truncate=False)

    # === הצגת schema של הטבלאות ===
    print("\n📋 Schema של טבלת דירוגים:")
    ratings.printSchema()

    # === עיבוד הדאטה ל-ALS ===
    print("\n🔧 מכין דאטה עבור ALS...")
    ratings_df = ratings.select(
        col("User-ID").cast("long"),
        col("ISBN").cast("string"),
        col("Book-Rating").cast("long")
    ).na.drop()

    print("✅ דאטה מוכן עבור ALS:")
    ratings_df.show(10)

    print(f"מספר דירוגים לאחר ניקוי: {ratings_df.count()}")

    # === בדיקת טווח הערכים ===
    print("\n📈 סטטיסטיקות דירוגים:")
    ratings_df.describe().show()

except Exception as e:
    print(f"❌ שגיאה: {str(e)}")
    print("\n🔍 בדיקות אפשריות:")
    print("1. ודא שקובץ ה-DB קיים בנתיב שצוין")
    print("2. ודא שקובץ sqlite-jdbc-3.45.2.0.jar קיים ב-C:\\spark\\jars\\")
    print("3. בדוק שהטבלאות books, users, books_ratings קיימות ב-DB")

finally:
    # ניקוי
    spark.stop()
    print("\n🛑 Spark Session נסגר")