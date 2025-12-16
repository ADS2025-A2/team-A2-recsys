import sqlite3
import hashlib
from pathlib import Path

# Point to the same DB file your Streamlit app uses.
# If you prefer absolute: Path(__file__).resolve().parent / "user_data.db"
DB_NAME = "user_data.db"


def get_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False)


def _table_columns(cursor, table_name: str) -> set[str]:
    cursor.execute(f"PRAGMA table_info({table_name})")
    # rows: cid, name, type, notnull, dflt_value, pk
    return {row[1] for row in cursor.fetchall()}


def init_db():
    """
    Initialize DB schema safely.
    If a table already exists but is missing expected columns, add them.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # --- USERS ---
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password_hash TEXT
        )
    """)
    # If an old users table exists without user_id, add it (SQLite supports ADD COLUMN).
    cols = _table_columns(cursor, "users")
    if "user_id" not in cols:
        cursor.execute("ALTER TABLE users ADD COLUMN user_id INTEGER")
        # NOTE: If you truly had no user_id before, old rows won't auto-populate.
        # We'll tolerate that; get_user_id will then return None for those users.
    if "username" not in cols:
        cursor.execute("ALTER TABLE users ADD COLUMN username TEXT")
    if "password_hash" not in cols:
        cursor.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")

    # --- INITIAL RATINGS ---
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS initial_ratings (
            username TEXT PRIMARY KEY,
            done BOOLEAN DEFAULT 0,
            FOREIGN KEY(username) REFERENCES users(username)
        )
    """)
    # Ensure every user has an initial_ratings row
    cursor.execute("""
        INSERT OR IGNORE INTO initial_ratings (username, done)
        SELECT username, 0 FROM users
    """)

    # --- PREFERENCES ---
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS preferences (
            username TEXT PRIMARY KEY,
            genres TEXT,
            FOREIGN KEY(username) REFERENCES users(username)
        )
    """)

    # --- RATINGS ---
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ratings (
            username TEXT,
            user_id INTEGER,
            movie TEXT,
            rating INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (username, movie),
            FOREIGN KEY(username) REFERENCES users(username)
        )
    """)
    # If an old ratings table exists missing user_id/timestamp, add them
    cols = _table_columns(cursor, "ratings")
    if "user_id" not in cols:
        cursor.execute("ALTER TABLE ratings ADD COLUMN user_id INTEGER")
    if "timestamp" not in cols:
        cursor.execute("ALTER TABLE ratings ADD COLUMN timestamp DATETIME")

    # --- WATCHLIST ---
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            username TEXT,
            movie TEXT,
            year INTEGER,
            PRIMARY KEY (username, movie),
            FOREIGN KEY(username) REFERENCES users(username)
        )
    """)

    conn.commit()
    conn.close()


# --- User registration ---
def register_user(username, password):
    """
    Register a user.
    Returns:
      - user_id (int) on success
      - None if username already exists
    """
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    conn = get_connection()
    cursor = conn.cursor()

    try:
        # Always ensure schema is ready (protects against stale DBs)
        init_db()

        cursor.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, password_hash)
        )
        conn.commit()

        # Ensure initial_ratings row exists for this user
        cursor.execute(
            "INSERT OR IGNORE INTO initial_ratings (username, done) VALUES (?, 0)",
            (username,)
        )
        conn.commit()

        # Return user_id robustly
        
        cursor.execute("SELECT 1 FROM users WHERE username=? LIMIT 1", (username,))
        
        row = cursor.fetchone()
        return row[0] if row else None

    except sqlite3.IntegrityError:
        return None  # Username already exists
    finally:
        conn.close()


# --- User login ---
def verify_login(username, password):
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash FROM users WHERE username = ? LIMIT 1", (username,))
    row = cursor.fetchone()
    conn.close()
    return bool(row and row[0] == password_hash)


def get_user_id(username: str):
    """
    Return numeric user_id for a username, or None.
    Works even if DB is old (missing user_id) without crashing.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Ensure schema exists
    init_db()

    cols = _table_columns(cursor, "users")
    if "user_id" not in cols:
        conn.close()
        return None

    cursor.execute("SELECT user_id FROM users WHERE username = ? LIMIT 1", (username,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None


def get_preferences(username):
    conn = get_connection()
    cursor = conn.cursor()
    init_db()
    cursor.execute("SELECT genres FROM preferences WHERE username = ? LIMIT 1", (username,))
    row = cursor.fetchone()
    conn.close()
    if row and row[0]:
        return [g for g in row[0].split(",") if g]
    return []


def get_rating(username, movie):
    conn = get_connection()
    cursor = conn.cursor()
    init_db()
    cursor.execute(
        "SELECT rating FROM ratings WHERE username = ? AND movie = ? LIMIT 1",
        (username, movie)
    )
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else 0


def set_initial_true(username):
    conn = get_connection()
    cursor = conn.cursor()
    init_db()
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
    init_db()
    cursor.execute(
        "SELECT done FROM initial_ratings WHERE username = ? LIMIT 1",
        (username,)
    )
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else 0


def add_to_watchlist(username, movie, year):
    conn = get_connection()
    cursor = conn.cursor()
    init_db()
    cursor.execute("""
        INSERT INTO watchlist (username, movie, year)
        VALUES (?, ?, ?)
        ON CONFLICT(username, movie) DO NOTHING
    """, (username, movie, year))
    conn.commit()
    conn.close()


def save_rating(username, movie, rating):
    """
    Store a rating. Also stores user_id if available.
    """
    init_db()
    user_id = get_user_id(username)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO ratings (username, user_id, movie, rating, timestamp)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(username, movie)
        DO UPDATE SET rating=excluded.rating, timestamp=CURRENT_TIMESTAMP, user_id=excluded.user_id
    """, (username, user_id, movie, rating))

    conn.commit()
    conn.close()


def save_preferences(username: str, genres: list[str]):
    """
    Save or update the user's genre preferences.
    `genres` should be a list like ["Action", "Comedy"].
    """
    init_db()
    genres_str = ",".join(genres)

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO preferences (username, genres)
        VALUES (?, ?)
        ON CONFLICT(username) DO UPDATE SET genres=excluded.genres
    """, (username, genres_str))
    conn.commit()
    conn.close()


def get_watchlist(username):
    conn = get_connection()
    cursor = conn.cursor()
    init_db()
    cursor.execute("""
        SELECT movie, year FROM watchlist WHERE username = ?
    """, (username,))
    movies = cursor.fetchall()
    conn.close()
    return movies


def remove_from_watchlist(username, movie):
    conn = get_connection()
    cursor = conn.cursor()
    init_db()
    cursor.execute("""
        DELETE FROM watchlist
        WHERE username = ? AND movie = ?
    """, (username, movie))
    conn.commit()
    conn.close()
