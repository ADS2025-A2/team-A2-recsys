import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager
import json
import os
import pandas as pd
from database import get_preferences, save_preferences

# --- check login ---
if "authenticated" not in st.session_state or not st.session_state.authenticated:
    st.switch_page("Home.py")

cookies = EncryptedCookieManager(
    prefix="movie_app/",
    password="super_secret_key_123"
)

if not cookies.ready():
    st.stop()

# --- frontend ---
col1, col2 = st.columns([3,1])

with col1:
    st.title("Profile")

    #st.header("Personal Information")
    st.write("Username:", st.session_state.username)

    username = st.session_state.username
    current_genres = get_preferences(username)

    genres = pd.read_csv("unique_genres.csv")

    selected = st.multiselect(
        "Choose your favourite genres:",
        genres,
        default=current_genres)

    if st.button("Save Preferences"):
        save_preferences(username, selected)
        st.success("Preferences saved!")

with col2:
    if st.button("Log out"):
        st.session_state.authenticated = False
        st.session_state.username = None
        cookies["username"] = ""
        cookies["expiry"] = ""
        cookies.save()
        st.rerun()