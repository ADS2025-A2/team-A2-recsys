import sqlite3
import hashlib

DB_NAME = "user_data.db"

def get_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

# --- Create the table if it doesn't exist ---
def init_db():
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS preferences (
            username TEXT PRIMARY KEY,
            genres TEXT
        )
    """)
    conn.commit()
    conn.close()

# --- User registration ---
def register_user(username, password):
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", 
                  (username, password_hash))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # Username already exists
    finally:
        conn.close()

# --- User login ---
def verify_login(username, password):
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()
    if row and row[0] == password_hash:
        return True
    return False

def get_preferences(username):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT genres FROM preferences WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()
    if row:
        return row[0].split(",")
    return []

def save_preferences(username, genres):
    conn = get_connection()
    c = conn.cursor()
    genres_str = ",".join(genres)
    c.execute("""
        INSERT INTO preferences (username, genres)
        VALUES (?, ?)
        ON CONFLICT(username)
        DO UPDATE SET genres=excluded.genres
    """, (username, genres_str))
    conn.commit()
    conn.close()