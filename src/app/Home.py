import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager
from datetime import datetime, timedelta
import pandas as pd
import requests
from database import init_db, verify_login, register_user, save_rating,save_preferences, get_rating, get_initial, set_initial_true, add_to_watchlist
from model.recommendations import DUMMY_RECOMMENDATIONS
import os
from api import get_movie_poster
from streamlit_star_rating import st_star_rating
from pathlib import Path
import random  # ✅ ADDED


st.markdown("""
<style>

[data-testid="stSidebar"] {
    background-color: #111111 !important;
}


[data-testid="stSidebarNav"] span {
    color: #FFFFFF !important;         /* texto blanco */
    font-weight: none;
    font-size: 16px !important;
}


[data-testid="stSidebarNav"] .css-1fv8s86 {
    color: #d72a18 !important;      
}


[data-testid="stSidebarNav"] span:hover {
    color: #fff !important;
    cursor: pointer;
}

[data-testid="stSidebarNav"] {
    padding-top: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

div[data-baseweb="select"] > div {
    background-color: #111 !important;      
    border-radius: 8px !important;
    border: 1px solid #fff !important;    
}


div[data-baseweb="select"] svg {
    color: #fff !important;                 
}

div[data-baseweb="select"] input {
    color: #fff !important;                  
}

div[data-baseweb="select"] span {
    color: #fff !important;                 
    font-size: 16px !important;

}


ul {
    background-color: #111 !important;
    border: 1px solid #111 !important;
}

ul li {
    background-color: #111 !important;
    color: #fff !important;
}

ul li:hover {
    background-color: #d72a18 !important;
    color: #fff !important;
}


div[data-baseweb="tag"] {
    background-color: #d72a18 !important;
    color: #fff !important;
    border-radius: 5px !important;
}


</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>

.login-box {
    background-color: #111111;
    max-width: 600px;
    width: 90%;
    padding: 50px;
    border-radius: 15px;
    margin: 50px auto;
    box-shadow: 0 0 20px #000;
    text-align: center;
    border: 10px
}


.login-box h2 {
    color: #d72a18;
    font-size: 80px;
    font-family: 'Bebas Neue', sans-serif;
    margin-bottom: 20px;
}


.login-box p {
    color: #fff;
    font-size: 20px;
    margin-bottom: 30px;
}


div[data-testid="stTextInput"] input {
    background-color: #fff !important;  
    color: #000 !important;             
    border: 1px solid #333 !important;
    border-radius: 6px !important;
    padding: 8px 10px !important;
    font-size: 16px !important;
}


div[data-testid="stTextInput"] label {
    font-size: 16px !important;
    color: #000 !important;
    font-weight: bold !important;
    margin-bottom: 5px;
}


div[data-testid="stButton"] {
    display: flex !important;
    justify-content: center !important;
    margin-top: 15px;
}

div[data-testid="stButton"] button {
    background-color: #d72a18 !important;
    color: white !important;
    border: none !important;
    padding: 12px 40px !important;
    border-radius: 8px !important;
    font-size: 18px !important;
    font-weight: bold !important;
    cursor: pointer;
    transition: 0.25s ease-in-out;
}

div[data-testid="stButton"] button:hover {
    background-color: #b02114 !important;
    transform: scale(1.03);
}


div[data-baseweb="radio"] label span {
    color: #fff !important;
    font-size: 16px !important;
    font-weight: bold;
}

div[data-baseweb="radio"] > div {
    margin-top: 5px !important;
}


</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

div[data-testid="stTextInput"] label {
    font-size: 50px !important; 
    color: #000 !important;   
    font-weight: bold !important;
}
</style>
""", unsafe_allow_html=True)




# ========================
# INIT DATABASE
# ========================
init_db()

# ========================
# COOKIE SETUP
# ========================
cookies = EncryptedCookieManager(
    prefix="movie_app/",
    password="super_secret_key_123"
)

if not cookies.ready():
    st.stop()

# ========================
# SESSION STATE
# ========================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "mode" not in st.session_state:
    st.session_state.mode = "login"

# Read cookies
cookie_username = cookies.get("username")
cookie_expiry = cookies.get("expiry")

if cookie_username and cookie_expiry:
    expiry_time = datetime.fromisoformat(cookie_expiry)
    if datetime.now() < expiry_time:
        st.session_state.authenticated = True
        st.session_state.username = cookie_username

# --- Hide sidebar on login screen ---
if not st.session_state.authenticated:
    hide_sidebar_style = """
        <style>
            [data-testid="stSidebar"] {display: none;}
            [data-testid="stSidebarNav"] {display: none;}
        </style>
    """
    st.markdown(hide_sidebar_style, unsafe_allow_html=True)
    

# ========================
# LOGIN SCREEN
# ========================

if not st.session_state.authenticated:
    st.markdown("""
        <div class="login-box">
            <h2>Welcome</h2>
            <p>Please log in or register to continue</p>
        </div>
    """, unsafe_allow_html=True)


    st.session_state.mode = st.radio("", ["Login", "Register"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")


    # LOGIN
    if st.session_state.mode == "Login":
        if st.button("Log in"):
            if verify_login(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username

                cookies["username"] = username
                cookies["expiry"] = (datetime.now() + timedelta(hours=1)).isoformat()
                cookies.save()

                st.success("Login successful!")
                st.rerun()
            else:

                st.error("Invalid username or password.")


    # REGISTER
    else:
        confirm = st.text_input("Confirm Password", type="password")
        if st.button("Register"):
            if password != confirm:
               st.error("Passwords do not match.")
            else:
                success = register_user(username, password)
                if success:
                    st.success("Account created. Please log in.")
                    st.session_state.mode = "Login"
                else:
                    st.error("Username already exists.")

    st.stop()

# ========================
# INITIAL RATINGS NEW USER
# ========================

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #111111 !important; 
}

[data-testid="stMainContent"] {
    background-color: #111111 !important;
    color: #ffffff !important;
}


.recommendation-section h1 {
    color: #d72a18;         
    font-size: 48px;        
    font-family: 'Bebas Neue', sans-serif; 
    font-weight: normal;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

def askInitialRating(movie, year):
    poster_url = get_movie_poster(movie, year)
    st.image(poster_url, width=150)
    previous_rating = get_rating(st.session_state.username, movie)


    rating = st_star_rating(
        label="", 
        maxValue=5, 
        defaultValue=previous_rating, 
        key=f"rating_{movie}",
        size=16,
        dark_theme=True
    )
    return rating 

if get_initial(st.session_state.username) == 0:
    st.markdown(
        "<p style='font-size:28px; color:#d72a18; margin:0;'>Please rate these 5 movies before continuing.</p>",
        unsafe_allow_html=True
    )

    initial_movies = ["Toy Story", "Star Wars: Episode I - The Phantom Menace", "Groundhog Day", "Pulp Fiction", "Grease"]
    initial_years = [1995, 1999, 1993, 1994, 1978]

    ratings = {}
    for movie in initial_movies:
        ratings[movie] = 0


    columns = st.columns(5)
    for i, (col, movie, year) in enumerate(zip(columns, initial_movies, initial_years)):
        with col:
            rating = askInitialRating(movie, year)
            ratings[movie] = rating if rating is not None else 0


    
    if st.button("Save Ratings"):
        if all(r >= 1 for r in ratings.values()):
            for movie in initial_movies:
                save_rating(st.session_state.username, movie, ratings[movie])
            set_initial_true(st.session_state.username)
            st.rerun()
        else:
            st.markdown("""
            <style>
            div[data-testid="stAlert"] {
                background-color: #2b2b2b !important;  
                color: #ffffff !important;             
                border-left: 5px solid #d72a18 !important; 
                padding: 15px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 16px;
                margin: 10px 0;
            }
            </style>
            """, unsafe_allow_html=True)

            # Mostrar el warning
            st.warning("Please rate all 5 movies before continuing")
                    

    st.stop()

# ========================
# MAIN PAGE — USER LOGGED IN
# ========================

# Fondo oscuro para toda la sección de recomendaciones

st.markdown("""
<style>
.recommendation-section {
    text-align: center;
}


.recommendation-section h1 {
    color: #d72a18;         
    font-size: 70px !important;        
    font-family: 'Bebas Neue', sans-serif; 
    font-weight: normal;
    margin-bottom: 15px;
}


.recommendation-section p {
    color: #ffffff;            
    font-size: 22px;
    margin-top: 0;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background-color: #111111 !important; 
}


[data-testid="stMainContent"] {
    background-color: #111111 !important;
    color: #ffffff !important;
}

[data-testid="stSidebar"] {
    background-color: #111111 !important;
    color: #ffffff !important;
}

section {
    background-color: #111111 !important;
}

.recommendation-section h1 {
    color: #d72a18;         
    font-size: 48px;        
    font-family: 'Bebas Neue', sans-serif; 
    font-weight: normal;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)



#nuevo

st.markdown("""
<style>
@keyframes fadeIn {
    0% { opacity: 0; transform: translateY(10px); }
    100% { opacity: 1; transform: translateY(0); }
}


.red-title {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-weight: 700;
    font-size: 4.5rem !important;
    text-align: center;
    color: #d72a18 !important;
    letter-spacing: 1px;
    margin-top: 20px;
    text-transform: none;
    width: 100%;
    animation: fadeIn 1s ease-out forwards;

    text-shadow:
        0 0 10px rgba(215, 42, 24, 0.6),
        0 0 20px rgba(215, 42, 24, 0.4),
        0 0 30px rgba(215, 42, 24, 0.2);
}

.red-subtitle {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-weight: 400;
    font-size: 1.3rem !important;
    text-align: center;
    color: #FFFFFF !important;
    margin-top: -10px;
    margin-bottom: 20px;
    text-transform: none;
    width: 100%;
    animation: fadeIn 1.3s ease-out forwards;

    text-shadow:
        0 0 8px rgba(255, 255, 255, 0.4),
        0 0 14px rgba(255, 255, 255, 0.2);
}
</style>
""", unsafe_allow_html=True)



st.markdown("""
<style>
@keyframes fadeIn {
    0% { opacity: 0; transform: translateY(10px); }
    100% { opacity: 1; transform: translateY(0); }
}

.red-title2 {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-weight: 700;
    font-size: 2.5rem !important;
    text-align: center;
    color: #d72a18 !important;
    letter-spacing: 0.5px;
    margin-top: 15px;
    margin-bottom: 20px;
    text-transform: none;
    width: 100%;
    animation: fadeIn 1.2s ease-out forwards;

    /* Sombra suave */
    text-shadow: 0 0 4px rgba(0,0,0,0.5);
}
</style>
""", unsafe_allow_html=True)

# Bloque de recomendaciones
st.markdown('<h1 class="red-title">Movie Recommendations</h1>', unsafe_allow_html=True)

st.markdown('<h3 class="red-subtitle" style="color:#FFFFFF !important;">Checkout Your Personalized Movie Picks!</h3>', unsafe_allow_html=True)


st.title("🎬 Movie Recommendations")
# ========================
# LOAD FRONTEND USER -> TRAINING USER_ID MAPPING
# ========================
@st.cache_data
def load_frontend_user_map(mapping_path_str: str, mtime: float):
    df_map = pd.read_csv(mapping_path_str)
    # tolerate old column name if you ever had one
    if "internal_user_id" not in df_map.columns and "user_id" in df_map.columns:
        df_map = df_map.rename(columns={"user_id": "internal_user_id"})
    return dict(zip(df_map["username"].astype(str), df_map["internal_user_id"].astype(int)))

base_dir = Path(__file__).resolve().parents[2]
mapping_path = base_dir / "models" / "frontend_user_map.csv"
st.session_state.user_map = load_frontend_user_map(str(mapping_path), mapping_path.stat().st_mtime)

if "user_map" not in st.session_state:
    try:
        st.session_state.user_map = load_frontend_user_map()
    except Exception as e:
        st.session_state.user_map = {}
        st.warning(f"Could not load user mapping: {e}")

def fix_title(title):
    articles = ["The", "A", "An"]
    for article in articles:
        suffix = f", {article}"
        if title.endswith(suffix):
            title = f"{article} {title[:-len(suffix)]}"
            break
    return title

# ========================
# RANDOM MOVIE FALLBACK ✅ ADDED
# ========================
def get_random_movies(df: pd.DataFrame, k: int = 12, seed_key: str = "random_seed"):
    """
    Returns k random movies from df.
    Uses a session seed so the list doesn't change on every rerun,
    unless you explicitly change the seed.
    """
    if df is None or df.empty:
        return []

    if seed_key not in st.session_state:
        st.session_state[seed_key] = random.randint(1, 10_000_000)

    rnd = random.Random(st.session_state[seed_key])
    sample_idx = rnd.sample(list(df.index), k=min(k, len(df)))

    out = []
    for i in sample_idx:
        row = df.loc[i]
        out.append({
            "title": row["title"],     # includes (YEAR) in your dataset
            "genre": row.get("genres", "Unknown")
        })
    return out

# ========================
# MOVIE CATALOG
# ========================

try:
    response = requests.get("http://127.0.0.1:8000/movies")
    response.raise_for_status()
    movies = response.json()

    st.session_state.df = pd.DataFrame(movies)
    st.session_state.df["genres"] = st.session_state.df["genres"].str.replace("|", ", ")

    if "avg_ratings" not in st.session_state:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        path= os.path.join(BASE_DIR,"..", "..", "data", "processed", "avg_rating_per_movie.csv")
        st.session_state.avg_ratings = pd.read_csv(path)
        

except Exception as e:
    st.error(f"The information could not be loaded: {e}")

# ========================
# RECOMMENDATIONS FOR LOGGED-IN USER
# ========================

st.markdown("")
st.markdown("")
st.markdown("")

st.markdown('<h3 class="red-title2" style="font-size:1rem !important;">Recommended Movies For You</h3>', unsafe_allow_html=True)




username = st.session_state.username

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR,"..", "..", "models", "top12_recommendations_with_titles.csv")
try:
    rec_df = pd.read_csv(csv_path)

    mapped_internal_id = st.session_state.user_map.get(username)

    # ✅ CHANGED: if user is not in mapping, fall back later to random movies
    if mapped_internal_id is None:
        user_recs = pd.DataFrame()
    else:
        # Match on internal_user_id (Option A)
        rec_df["internal_user_id"] = rec_df["internal_user_id"].astype(int)
        user_recs = rec_df[rec_df["internal_user_id"] == int(mapped_internal_id)]

    recommended_movies = []
    if not user_recs.empty:
        # CSV columns are: internal_user_id, user_id, item_1..item_12
        titles = user_recs.iloc[0, 2:].dropna().tolist()
        for title in titles:
            movie_row = st.session_state.df[st.session_state.df['title'] == title]
            if not movie_row.empty:
                recommended_movies.append({
                    "title": title,
                    "genre": movie_row.iloc[0]["genres"]
                })
            else:
                recommended_movies.append({
                    "title": title,
                    "genre": "Unknown"
                })

except Exception as e:
    st.error(f"The recommendations could not be loaded: {e}")
    recommended_movies = []

if recommended_movies:
    cols = st.columns(3)
    for idx, movie in enumerate(recommended_movies):
        title, year = movie["title"].rsplit(" (", 1)
        year = year.replace(")", "")
        with cols[idx % 3]:
            with st.form(key=f"movie_form_{idx}"):
                st.markdown(
                    f"""
                    <div style='padding: 10px; border: 1px solid #ddd; 
                                border-radius: 10px; margin-bottom: 15px;'>
                        <h4 style='margin-bottom: 5px;'>{fix_title(title)}</h4>
                        <p style='color: gray;'>🎭 {movie['genre']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                col1, col2 = st.columns(2)
                with col1:
                    submitted = st.form_submit_button("Info", key=f"info_{movie}")
                    if submitted:
                        st.session_state.selected_movie = movie["title"]
                        st.switch_page("pages/1_Details.py")
                with col2:
                    if st.form_submit_button("Add to Watchlist", key=f"watchlist_{movie}"):
                        add_to_watchlist(st.session_state.username, title, year)
                        st.success(f"{title} added to watchlist!")
else:
    # ✅ NEW: Random fallback UI (no CSS changes)
    if st.button("🎲 Refresh random picks"):
        st.session_state["random_seed"] = random.randint(1, 10_000_000)
        st.rerun()

    st.markdown("""
    <div style="
        background-color: #111111;
        color: #ffffff;
        border: 1px solid #2b2b2b;
        padding: 14px 18px;
        border-radius: 12px;
        text-align: center;
        font-size: 16px;
        font-family: 'Montserrat', sans-serif;
    ">
    No personalized recommendations yet, but here are some random picks to explore 👇
    </div>
    """, unsafe_allow_html=True)

    random_movies = get_random_movies(st.session_state.df, k=12)

    cols = st.columns(3)
    for idx, movie in enumerate(random_movies):
        title, year = movie["title"].rsplit(" (", 1)
        year = year.replace(")", "")

        with cols[idx % 3]:
            with st.form(key=f"random_movie_form_{idx}"):
                st.markdown(
                    f"""
                    <div style='padding: 10px; border: 1px solid #ddd; 
                                border-radius: 10px; margin-bottom: 15px;'>
                        <h4 style='margin-bottom: 5px;'>{fix_title(title)}</h4>
                        <p style='color: gray;'>🎭 {movie['genre']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                col1, col2 = st.columns(2)
                with col1:
                    if st.form_submit_button("Info", key=f"rand_info_{idx}"):
                        st.session_state.selected_movie = movie["title"]
                        st.switch_page("pages/1_Details.py")
                with col2:
                    if st.form_submit_button("Add to Watchlist", key=f"rand_watch_{idx}"):
                        add_to_watchlist(st.session_state.username, title, year)
                        st.success(f"{title} added to watchlist!")


options = st.session_state.df["title"]

st.session_state.selected_movie = None
st.markdown("")
st.markdown("")

# ==== CUSTOM CSS para el MULTISELECT =====
st.markdown("""
<style>

div[data-baseweb="select"] > div {
    background-color: #111 !important;       
    border-radius: 8px !important;
    border: 1px solid #fff !important;    
}


div[data-baseweb="select"] svg {
    color: #fff !important;                  
}

div[data-baseweb="select"] input {
    color: #fff !important;                  
}

div[data-baseweb="select"] span {
    color: #fff !important;              
    font-size: 16px !important;

}

ul {
    background-color: #111 !important;
    border: 1px solid #111 !important;
    color: #111 !important;
}

ul li {
    background-color: #111 !important;
    color: #fff !important;
}

ul li:hover {
    background-color: #d72a18 !important;
    color: #fff !important;
}

div[data-baseweb="tag"] {
    background-color: #d72a18 !important;
    color: #fff !important;
    border-radius: 5px !important;
}


</style>
""", unsafe_allow_html=True)



# ==== MULTISELECT ====
st.markdown("""
<h2 style="
    color: #d72a18;           
    font-family: 'Montserrat', sans-serif;
    font-size: 22px;
    margin-bottom: 6px;
    text-transform: none;
">
  Search for a movie
</h2>
""", unsafe_allow_html=True)

st.session_state.selected_movie = st.multiselect(
    "",
    options
)

if st.session_state.selected_movie:
    st.switch_page("pages/1_Details.py")
