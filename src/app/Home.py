import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager
from datetime import datetime, timedelta
import json
import os
import hashlib

# --- cookie setup ---
cookies = EncryptedCookieManager(
    prefix="movie_app/",
    password="super_secret_key_123"
)

if not cookies.ready():
    st.stop()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if not os.path.exists("users.json"):
        return {}
    with open("users.json", "r") as f:
        return json.load(f)

def save_users(users):
    with open("users.json", "w") as f:
        json.dump(users, f, indent=4)

users = load_users()

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
            hashed = hash_password(password)
            if username in users and users[username] == hashed:
                st.session_state.authenticated = True
                st.session_state.username = username
                cookies["username"] = username
                cookies["expiry"] = (datetime.now() + timedelta(hours=1)).isoformat()
                cookies.save()
                st.rerun()
            else:
                st.error("Invalid username or password.")
# --- Register ---
    else:
        confirm = st.text_input("Confirm Password", type="password")
        if st.button("Register"):
            if username in users:
                st.error("User already exists.")
            elif password != confirm:
                st.error("Passwords do not match.")
            else:
                users[username] = hash_password(password)
                save_users(users)
                st.success("Account created. Please log in.")
                st.session_state.mode = "Login"

    st.stop()

# --- Home page after login ---    

st.title("Movie Recommendations")
st.image("Movie theatre.jpg")