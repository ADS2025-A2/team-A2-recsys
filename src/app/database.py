import sqlite3
import hashlib

DB_NAME = "user_data.db"

def get_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

# --- Create the table if it doesn't exist ---
def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS initial_ratings (
            username TEXT PRIMARY KEY,
            done BOOLEAN DEFAULT 0,
            FOREIGN KEY(username) REFERENCES users(username)
        )
    """)
    cursor.execute("""
        INSERT OR IGNORE INTO initial_ratings (username, done)
        SELECT username, 0 FROM users
        """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS preferences (
            username TEXT PRIMARY KEY,
            genres TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ratings (
            username TEXT,
            movie TEXT,
            rating INTEGER,
            PRIMARY KEY (username, movie)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            username KEY,
            movie TEXT,
            year INTEGER,
            PRIMARY KEY (username, movie)
        )
    """)

    conn.commit()
    conn.close()

# --- User registration ---
def register_user(username, password):
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", 
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
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    if row and row[0] == password_hash:
        return True
    return False

def get_preferences(username):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT genres FROM preferences WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return row[0].split(",")
    return []

def save_preferences(username, genres):
    conn = get_connection()
    cursor = conn.cursor()
    genres_str = ",".join(genres)
    cursor.execute("""
        INSERT INTO preferences (username, genres)
        VALUES (?, ?)
        ON CONFLICT(username)
        DO UPDATE SET genres=excluded.genres
    """, (username, genres_str))
    conn.commit()
    conn.close()


def save_rating(username, title, rating):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO ratings (username, movie, rating) VALUES (?, ?, ?)
        ON CONFLICT(username, movie)
        DO UPDATE SET rating = excluded.rating
        """, (username, title, rating)
    )
    conn.commit()
    conn.close()



def get_rating(username, movie):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT rating FROM ratings WHERE username = ? AND movie = ?",
        (username, movie)
    )
    result = cursor.fetchone()
    return result[0] if result else 0


def set_initial_true(username):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE initial_ratings
        SET done = 1
        WHERE username = ?
    """, (username,))
    conn.commit()
    conn.close()


def get_initial(username):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT done FROM initial_ratings WHERE username = ?",
        (username,)
    )
    result = cursor.fetchone()
    return result[0]


def add_to_watchlist(username, movie, year):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO watchlist (username, movie, year)
        VALUES (?, ?, ?)
        ON CONFLICT(username, movie) DO NOTHING
    """, (username, movie, year))

    conn.commit()
    conn.close()


def get_watchlist(username):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT movie, year FROM watchlist WHERE username = ?
    """, (username,))

    movies = cursor.fetchall()
    conn.close()
    return movies


def remove_from_watchlist(username, movie):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        DELETE FROM watchlist
        WHERE username = ? AND movie = ?
    """, (username, movie))

    conn.commit()
    conn.close()
