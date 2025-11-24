import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager
from datetime import datetime, timedelta
import json
import os
import hashlib
from database import init_db, verify_login, register_user

# --- initialise database ---
init_db()

# --- cookie setup ---
cookies = EncryptedCookieManager(
    prefix="movie_app/",
    password="super_secret_key_123"
)

if not cookies.ready():
    st.stop()

# --- Session State ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "mode" not in st.session_state:
    st.session_state.mode = "login"

# --- Check cookie for authentication ---
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

# --- Main ---
if not st.session_state.authenticated:

    st.title("Welcome")
    st.header("Please log in or register")
    st.session_state.mode = st.radio("Select:", ["Login", "Register"])

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

# --- Login ---
    if st.session_state.mode == "Login":
        if st.button("Log in"):
            if verify_login(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username

                # Set cookies if using EncryptedCookieManager
                cookies["username"] = username
                cookies["expiry"] = (datetime.now() + timedelta(hours=1)).isoformat()
                cookies.save()

                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password.")
# --- Register ---
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

# --- Home page after login ---    

st.title("Movie Recommendations")
st.image("Movie theatre.jpg")