import pandas as pd
import sqlite3
import os

# ================== CONFIG ==================
BASE_PATH = r"C:\Users\shira\OneDrive\שולחן העבודה\Big Data\final"
DB_PATH = os.path.join(BASE_PATH, "books", "books_no_outliers.db")
RECS_CSV = os.path.join(BASE_PATH, "user_recommendations_full.csv")

# ================== Load recommended users ==================
recs_df = pd.read_csv(RECS_CSV, encoding="utf-8")
# נשלוף את ה-original_userId בלבד
recommended_user_ids = recs_df["original_userId"].unique().tolist()

print(f"Found {len(recommended_user_ids)} unique users with recommendations.")

# ================== Connect to SQLite ==================
conn = sqlite3.connect(DB_PATH)

# ================== Query BX-Users ==================
placeholders = ",".join(["?"] * len(recommended_user_ids))  # ? placeholders for SQL IN
query = f"""
SELECT [User-ID], Location, Age
FROM [Users]
WHERE [User-ID] IN ({placeholders})
ORDER BY [User-ID]
"""

users_df = pd.read_sql_query(query, conn, params=recommended_user_ids)
conn.close()

# ================== Print results ==================
print("\n📋 Details of users with recommendations:")
print(users_df.to_string(index=False))
