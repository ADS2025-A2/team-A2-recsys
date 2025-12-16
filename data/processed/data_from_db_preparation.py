import sqlite3
import pandas as pd
from pathlib import Path
import re



PROJECT_ROOT = Path(__file__).resolve().parents[2]

DB_PATH = PROJECT_ROOT / "src" / "app" / "user_data.db"
MOVIES_PATH = PROJECT_ROOT / "data" / "processed" / "ml-10M100K" / "movies.dat"
OUT_DIR = PROJECT_ROOT / "data" / "processed"



def clean_title(s: str):
    """Make titles comparable: lowercase, strip spaces, remove '(1995)' at the end."""
    if pd.isna(s):
        return s
    s = s.strip().lower()
    # remove ' (YYYY)' at the end
    s = re.sub(r"\s*\(\d{4}\)\s*$", "", s)
    return s


def extract_tables_from_db():
    """Read ratings and preferences tables from the SQLite DB."""
    print(f"Reading DB from: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)

    ratings = pd.read_sql("SELECT * FROM ratings", conn)
    preferences = pd.read_sql("SELECT * FROM preferences", conn)

    conn.close()
    return ratings, preferences


def add_movie_ids_to_ratings(ratings: pd.DataFrame):
    """Merge ratings with movies.dat to get movie_id based on the title."""

    print(f"Reading movies from: {MOVIES_PATH}")
    # Adjust sep if needed (e.g. sep="::" for original MovieLens)
    movies = pd.read_csv(MOVIES_PATH)

    # Clean titles in movies.dat
    movies["title_clean"] = movies["title"].apply(clean_title)
    movies = movies.drop_duplicates(subset="title_clean")

    # Clean movie names from DB
    ratings["title_clean"] = ratings["movie"].apply(clean_title)

    merged = ratings.merge(
        movies[["movie_id", "title_clean"]],
        on="title_clean",
        how="left",
    )

    # Check which didn't match
    missing = merged[merged["movie_id"].isna()]
    if not missing.empty:
        missing_path = OUT_DIR / "ratings_from_db_unmatched_titles.csv"
        missing.to_csv(missing_path, index=False)
        print(
            f"⚠️  {len(missing)} rows could not be matched to a movie_id.\n"
            f"    They were saved to: {missing_path}"
        )

    # Drop helper column and reorder columns nicely
    merged = merged.drop(columns=["title_clean"])

    desired_order = ["username", "movie_id", "rating", "movie"]
    merged = merged[[col for col in desired_order if col in merged.columns]]

    return merged


def main():
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    # 1) Extract tables
    ratings, preferences = extract_tables_from_db()

    # 2) Add movie_id to ratings
    ratings_with_ids = add_movie_ids_to_ratings(ratings)

    # 3) Save final CSVs
    ratings_path = OUT_DIR / "ratings_from_db.csv"
    prefs_path = OUT_DIR / "preferences_from_db.csv"

    ratings_with_ids.to_csv(ratings_path, index=False)
    preferences.to_csv(prefs_path, index=False)

    print("✅ Finished exporting:")
    print(f"   - {ratings_path}")
    print(f"   - {prefs_path}")


if __name__ == "__main__":
    main()
