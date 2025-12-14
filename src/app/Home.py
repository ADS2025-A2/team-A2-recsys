import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager
from datetime import datetime, timedelta
import pandas as pd
import requests
from database import init_db, verify_login, register_user, save_rating, get_rating, get_initial, set_initial_true, add_to_watchlist
from model.recommendations import DUMMY_RECOMMENDATIONS
import os
from api import get_movie_poster
from streamlit_star_rating import st_star_rating



# ========================
# CUSTOM CSS PARA SIDEBAR
# ========================
st.markdown("""
<style>
/* Fondo negro del sidebar */
[data-testid="stSidebar"] {
    background-color: #111111 !important;
}

/* Items del menú (Home, Details, Watchlist, Profile) */
[data-testid="stSidebarNav"] span {
    color: #FFFFFF !important;         /* texto blanco */
    font-weight: none;
    font-size: 16px !important;
}

/* Item seleccionado */
[data-testid="stSidebarNav"] .css-1fv8s86 {
    color: #d72a18 !important;         /* item activo rojo estilo Netflix */
}

/* Hover sobre los items */
[data-testid="stSidebarNav"] span:hover {
    color: #fff !important;
    cursor: pointer;
}

/* Ajuste de padding interno para que se vea ordenado */
[data-testid="stSidebarNav"] {
    padding-top: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

div[data-baseweb="select"] > div {
    background-color: #111 !important;       /* Caja negra */
    border-radius: 8px !important;
    border: 1px solid #fff !important;    /* Borde rojo */
}


div[data-baseweb="select"] svg {
    color: #fff !important;                  /* Flechas blancas */
}

div[data-baseweb="select"] input {
    color: #fff !important;                  /* Texto blanco */
}

div[data-baseweb="select"] span {
    color: #fff !important;                  /* Texto de opciones */
    font-size: 16px !important;

}

/* Opciones desplegadas */
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

/* Chips seleccionados */
div[data-baseweb="tag"] {
    background-color: #d72a18 !important;
    color: #fff !important;
    border-radius: 5px !important;
}


</style>
""", unsafe_allow_html=True)



# ========================
# CUSTOM CSS FOR FULL BLACK SCREEN
# ========================

st.markdown("""
<style>
/* ==== LOGIN BOX ===== */
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

/* ==== LOGIN TITLE ===== */
.login-box h2 {
    color: #d72a18;
    font-size: 80px;
    font-family: 'Bebas Neue', sans-serif;
    margin-bottom: 20px;
}

/* ==== LOGIN SUBTITLE ===== */
.login-box p {
    color: #fff;
    font-size: 20px;
    margin-bottom: 30px;
}

/* ==== INPUTS BLANCOS ===== */
div[data-testid="stTextInput"] input {
    background-color: #fff !important;  /* fondo blanco */
    color: #000 !important;             /* texto negro */
    border: 1px solid #333 !important;
    border-radius: 6px !important;
    padding: 8px 10px !important;
    font-size: 16px !important;
}

/* ==== INPUT LABELS ===== */
div[data-testid="stTextInput"] label {
    font-size: 16px !important;
    color: #000 !important;
    font-weight: bold !important;
    margin-bottom: 5px;
}

/* ==== BOTÓN ROJO  ===== */
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

/* ==== RADIO BUTTONS ===== */
div[data-baseweb="radio"] label span {
    color: #fff !important;
    font-size: 16px !important;
    font-weight: bold;
}

div[data-baseweb="radio"] > div {
    margin-top: 5px !important;
}

/* ==== ERROR MESSAGES (mantener tu estilo actual) ===== */
/* No modificamos nada, usarás st.error o tu div existente */
</style>
""", unsafe_allow_html=True)


# Apply custom CSS for the text_input label
st.markdown("""
<style>
/* Change the font size and color of text_input labels */
div[data-testid="stTextInput"] label {
    font-size: 50px !important;  /* Adjust size */
    color: #000 !important;   /* Optional: change color */
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
/* Fondo negro para toda la app */
[data-testid="stAppViewContainer"] {
    background-color: #111111 !important; /* Fondo negro oscuro */
}


/* Fondo negro del área principal de contenido */
[data-testid="stMainContent"] {
    background-color: #111111 !important;
    color: #ffffff !important;
}


/* Título principal */
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
            /* Cambiar estilo de st.warning */
            div[data-testid="stAlert"] {
                background-color: #2b2b2b !important;  /* gris oscuro */
                color: #ffffff !important;             /* texto blanco */
                border-left: 5px solid #d72a18 !important; /* borde rojo Netflix */
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

/* Título principal */
.recommendation-section h1 {
    color: #d72a18;         
    font-size: 70px !important;        
    font-family: 'Bebas Neue', sans-serif; 
    font-weight: normal;
    margin-bottom: 15px;
}

/* Subtítulos o texto adicional */
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
/* Fondo negro para toda la app */
[data-testid="stAppViewContainer"] {
    background-color: #111111 !important; /* Fondo negro oscuro */
}


/* Fondo negro del área principal de contenido */
[data-testid="stMainContent"] {
    background-color: #111111 !important;
    color: #ffffff !important;
}

/* Fondo negro de sidebar (si la usas) */
[data-testid="stSidebar"] {
    background-color: #111111 !important;
    color: #ffffff !important;
}

/* Fondo negro de los formularios y bloques */
section {
    background-color: #111111 !important;
}


/* Título principal */
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
/* Animación suave */
@keyframes fadeIn {
    0% { opacity: 0; transform: translateY(10px); }
    100% { opacity: 1; transform: translateY(0); }
}

/* Título principal estilo Netflix */
.netflix-title {
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

    /* Glow + sombra */
    text-shadow:
        0 0 10px rgba(215, 42, 24, 0.6),
        0 0 20px rgba(215, 42, 24, 0.4),
        0 0 30px rgba(215, 42, 24, 0.2);
}

/* Subtítulo blanco con sombra suave */
.netflix-subtitle {
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

    /* sombra suave */
    text-shadow:
        0 0 8px rgba(255, 255, 255, 0.4),
        0 0 14px rgba(255, 255, 255, 0.2);
}
</style>
""", unsafe_allow_html=True)



st.markdown("""
<style>
/* Animación suave */
@keyframes fadeIn {
    0% { opacity: 0; transform: translateY(10px); }
    100% { opacity: 1; transform: translateY(0); }
}

/* Subtítulo sin glow, solo sombra suave */
.netflix-title2 {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-weight: 700;
    font-size: 2.5rem !important;
    text-align: center;
    color: #d72a18 !important;  /* rojo Netflix */
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
st.markdown('<h1 class="netflix-title">Movie Recommendations</h1>', unsafe_allow_html=True)

st.markdown('<h3 class="netflix-subtitle" style="color:#FFFFFF !important;">Checkout Your Personalized Movie Picks!</h3>', unsafe_allow_html=True)



def fix_title(title):
    articles = ["The", "A", "An"]
    for article in articles:
        suffix = f", {article}"
        if title.endswith(suffix):
            title = f"{article} {title[:-len(suffix)]}"
            break
    return title

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
    st.error(f"No se pudo cargar la información: {e}")

# ========================
# RECOMMENDATIONS FOR LOGGED-IN USER
# ========================

st.markdown("")
st.markdown("")
st.markdown("")

st.markdown('<h3 class="netflix-title2" style="font-size:1rem !important;">Recommended Movies For You</h3>', unsafe_allow_html=True)




username = st.session_state.username

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR,"..", "training", "top10_recommendations_with_titles.csv")
try:
    rec_df = pd.read_csv(csv_path)
    user_id = username
    user_recs = rec_df[rec_df['user_id'] == user_id]
    recommended_movies = []
    if not user_recs.empty:
        titles = user_recs.iloc[0, 1:].dropna().tolist()  
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
    st.markdown("""
    <div style="
        background-color: #111111;
        color: #fff;
        border: 1px solid #111111;
        padding: 15px 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 16px;
        font-family: 'Montserrat', sans-serif;
    ">
    No recommendations available for you.
    </div>
    """, unsafe_allow_html=True)


options = st.session_state.df["title"]

st.session_state.selected_movie = None
st.markdown("")
st.markdown("")

# ==== CUSTOM CSS para el MULTISELECT =====
st.markdown("""
<style>

div[data-baseweb="select"] > div {
    background-color: #111 !important;       /* Caja negra */
    border-radius: 8px !important;
    border: 1px solid #fff !important;    /* Borde rojo */
}


div[data-baseweb="select"] svg {
    color: #fff !important;                  /* Flechas blancas */
}

div[data-baseweb="select"] input {
    color: #fff !important;                  /* Texto blanco */
}

div[data-baseweb="select"] span {
    color: #fff !important;                  /* Texto de opciones */
    font-size: 16px !important;

}

/* Opciones desplegadas */
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

/* Chips seleccionados */
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
