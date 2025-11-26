# App/Home.py

import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager
from datetime import datetime, timedelta
import pandas as pd
import requests
import os
import sys
from database import init_db, verify_login, register_user
from model.recommendations import DUMMY_RECOMMENDATIONS

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

# Leer cookies
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
    st.title("Welcome")
    st.header("Please log in or register")

    st.session_state.mode = st.radio("Select:", ["Login", "Register"])
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
# MAIN PAGE — USER LOGGED IN
# ========================
st.title("🎬 Movie Recommendations App")
st.write(f"👤 Logged in as **{st.session_state.username}**")

# ========================
# MOVIE CATALOG
# ========================
st.subheader("📚 Movie Catalog")

try:
    response = requests.get("http://127.0.0.1:8000/movies")
    response.raise_for_status()
    movies = response.json()

    df = pd.DataFrame(movies)
    df["genres"] = df["genres"].str.replace("|", ", ")
    st.dataframe(df.head(10))

except Exception as e:
    st.error(f"No se pudo cargar la información: {e}")

# ========================
# RECOMMENDATIONS FOR LOGGED-IN USER
# ========================
st.subheader("🔥 Recommended Movies For You")

username = st.session_state.username

# Convert username to a numeric id for testing dummy recommendations
# Simple example: hash the username and mod by 10, then +1 to get id 1-10
user_id = (abs(hash(username)) % 10) + 1
recommended_movies = DUMMY_RECOMMENDATIONS.get(user_id, [])

# Mostrar recomendaciones en grid de 3 columnas
if recommended_movies:
    cols = st.columns(3)
    for idx, movie in enumerate(recommended_movies):
        with cols[idx % 3]:
            st.markdown(
                f"""
                <div style='padding: 10px; border: 1px solid #ddd; 
                            border-radius: 10px; margin-bottom: 15px;'>
                    <h4 style='margin-bottom: 5px;'>{movie['title']}</h4>
                    <p style='color: gray;'>🎭 {movie['genre']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
else:
    st.info("No recommendations available for this user.")


options = df["title"]


st.session_state.selected_movie = None
st.session_state.selected_movie = st.multiselect("Search for a movie:", options)

if st.session_state.selected_movie:
    st.switch_page("pages/1_Details.py")
