import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager
import json
import os
import pandas as pd

# --- check login ---
if "authenticated" not in st.session_state or not st.session_state.authenticated:
    st.switch_page("Home.py")

cookies = EncryptedCookieManager(
    prefix="movie_app/",
    password="super_secret_key_123"
)

if not cookies.ready():
    st.stop()

# --- Handle preferences.json file ---
PREF_FILE = "preferences.json"

def load_preferences():
    if not os.path.exists(PREF_FILE):
        return {}
    with open(PREF_FILE, "r") as f:
        return json.load(f)

def save_preferences(prefs):
    with open(PREF_FILE, "w") as f:
        json.dump(prefs, f, indent=4)

def get_user_preferences(username):
    prefs = load_preferences()
    return prefs.get(username, [])

def set_user_preferences(username, genres):
    prefs = load_preferences()
    prefs[username] = genres
    save_preferences(prefs)

# --- frontend ---
col1, col2 = st.columns([3,1])

with col1:
    st.title("Profile")

    st.header("Personal Information")
    st.write("Username:", st.session_state.username)

    username = st.session_state.username
    current_genres = get_user_preferences(username)

    genres = pd.read_csv(r"data/processed/unique_genres.csv")

    selected = st.multiselect(
        "Choose your favourite genres:",
        genres,
        default=current_genres)

    if st.button("Save Preferences"):
        set_user_preferences(username, selected)
        st.success("Preferences saved!")

with col2:
    if st.button("Log out"):
        st.session_state.authenticated = False
        st.session_state.username = None
        cookies["username"] = ""
        cookies["expiry"] = ""
        cookies.save()
        st.rerun()