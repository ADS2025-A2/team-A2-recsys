import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager
import json
import os
import pandas as pd
from database import get_preferences, save_preferences, get_initial

# --- check login ---
if "authenticated" not in st.session_state or not st.session_state.authenticated:
    st.switch_page("Home.py")

# --- Check if initial movies have been rated ---
if get_initial(st.session_state.username) == 0:
    st.switch_page("Home.py")

cookies = EncryptedCookieManager(
    prefix="movie_app/",
    password="super_secret_key_123"
)

if not cookies.ready():
    st.stop()



# --------------------------
# ESTILOS
# --------------------------
st.markdown(
    """
    <style>
    /* Fondo de toda la app */
    .stApp {
        background-color: #111111;
        color: white;
    }

    /* Título principal */
    h1 {
        color: #d72a18 !important;
        text-align: center;
        font-size: 60px;
        font-family: Arial, sans-serif;
        margin-bottom: 20px;
    }

    h2 {
            color: #FFFFFF !important;
            text-align: left;
            font-size: 18px !important;
            font-family: Arial, sans-serif;
            margin-bottom: 5px;
        }

    h3 {
        color: #d72a18 !important;
        text-align: center;
        font-size: 10px;
        font-family: Arial, sans-serif;
        margin-bottom: 20px;
    }
    
    /* Label del multiselect */
    .st-bk {
        color: #d72a18 !important;
        font-size: 15px !important;
        font-weight: bold;
        text-align: left;
    }


    /* Botones */
    div.stButton > button {
        background-color: #d72a18 !important;  /* rojo */
        color: white !important;               /* texto negro */
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        text-align: center;
    }

    div.stButton > button:hover {
        background-color: #e65b4f !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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

st.markdown(
    """
    <style>
        /* Multiselect options */
    ul {
        background-color: #000 !important;
        border: 1px solid #d72a18 !important;
    }
    ul li {
        color: #FFFFFF !important;
    }
    ul li:hover {
        background-color: #d72a18 !important;
        color: #FFFFFF !important;
    }

        
    </style>
    """,
    unsafe_allow_html=True
)

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
    font-family: 'Montserrat', sans-serif !important;
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
    border: 1px solid #111 !important;
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


# --- frontend ---
col1, col2 = st.columns([3,1])

with col1:
    st.markdown("<h1>Profile</h1>", unsafe_allow_html=True)

    #st.header("Personal Information")
    
    st.markdown("<h2>Username: </h2>", unsafe_allow_html=True)
    st.write("", st.session_state.username)

    username = st.session_state.username
    current_genres = get_preferences(username)

    genres = pd.read_csv("unique_genres.csv")

    st.markdown("<h3>Choose your favorite genres</h3>", unsafe_allow_html=True)


    selected = st.multiselect(
        "",
        genres,
        default=current_genres,
        max_selections=5)

        

    if st.button("Save Preferences"):
        if selected:
            save_preferences(username, selected)
            st.success("Preferences saved!")
        else:
            st.warning("Please choose at least one genre before saving.")

with col2:
    if st.button("Log out"):
        st.session_state.authenticated = False
        st.session_state.username = None
        cookies["username"] = ""
        cookies["expiry"] = ""
        cookies.save()
        st.rerun()
