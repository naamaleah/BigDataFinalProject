import sqlite3

# התחברות למסד הנתונים
conn = sqlite3.connect(r"C:\Users\shira\OneDrive\שולחן העבודה\Big Data\final\books\books_no_outliers.db")
cursor = conn.cursor()

# דוגמה להצגת הטבלאות
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())
# # מיפוי של שמות ישנים לשמות חדשים
# rename_map = {
#     "BX-Users": "Users",
#     "BX-Book-Ratings": "Book_Ratings",
#     "BX-Books": "Books"
# }

# # להריץ את פקודת שינוי השם לכל טבלה
# for old_name, new_name in rename_map.items():
#     cursor.execute(f'ALTER TABLE "{old_name}" RENAME TO "{new_name}";')

# # לשמור את השינויים
# conn.commit()

# # לוודא שהתבצע
# cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
# print(cursor.fetchall())

cursor.execute("SELECT * FROM Book_Ratings LIMIT 30;")
print(cursor.fetchall())

conn.close()
