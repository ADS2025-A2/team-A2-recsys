import streamlit as st
from database import get_initial, get_watchlist, remove_from_watchlist
from api import get_movie_poster
from Home import fix_title

import streamlit as st

st.markdown(
    """
    <style>
    /* Fondo de toda la app */
    .stApp, .css-18e3th9, .css-1outpf7, .css-ffhzg2 {
        background-color: #111;
        color: white;
    }

    /* Títulos */
    h1, h2, h3, h4, h5, h6 {
        color: white;
    }

    /* Texto normal */
    p, span, label {
        color: white;
    }

    /* Botones */
    div.stButton > button {
        background-color: #111111 !important;
        color: #111111 !important;
        border-radius: 8px;
        padding: 8px 16px;
        border: none;
    }

    div.stButton > button:hover {
        background-color: #333333;
        color: #111 !important;
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

# Estilo para botones específicos con fondo rojo, texto negro y alineación
st.markdown(
    """
    <style>
    /* Estilo para todos los botones */
    div.stButton > button {
        background-color: #d72a18 !important;  /* fondo rojo */
        color: black !important;               /* texto negro */
        font-weight: bold;
        border-radius: 8px;
        padding: 8px 16px;
        text-align: center;                    /* centrar texto */
    }

    div.stButton > button:hover {
        background-color: #e65b4f !important;  /* rojo más claro al pasar el mouse */
    }
    </style>
    """,
    unsafe_allow_html=True
)



# Check if user is logged in
if "authenticated" not in st.session_state or not st.session_state.authenticated:
    st.switch_page("Home.py")

# Check if initial movies have been rated
if get_initial(st.session_state.username) == 0:
    st.switch_page("Home.py")

st.markdown(
    """
    <h1 style='
        color: #d72a18;   /* color rojo */
        text-align: center; /* centrado */
        font-size: 60px;   /* tamaño más grande */
        font-family: Arial, sans-serif;
    '>
        Watchlist
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown("")
options = st.session_state.df["title"]
st.markdown(
    "<p style='color:#d72a18; font-size:24px; font-weight:bold; text-align:left; margin-bottom:5px !important;'>Search for a movie</p>",
    unsafe_allow_html=True
)
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

st.session_state.selected_movie = st.multiselect("", options)
st.markdown("")
st.markdown("")

if st.session_state.selected_movie:
    st.switch_page("pages/1_Details.py")

watchlist = get_watchlist(st.session_state.username)

for title, year in watchlist:
    col1, col2 = st.columns([2,1])
    with st.container():
        with col1:
            st.subheader(fix_title(title))
            if st.button("More Information", key=f"info_{title}"):
                st.session_state.selected_movie = [f"{title} ({year})"]
                st.switch_page("pages/1_Details.py")
            if st.button("Remove from Watchlist", key=f"remove_{title}"):
                remove_from_watchlist(st.session_state.username, title)
                st.success("Movie removed from watchlist.")
                st.rerun()
        with col2:
            poster_url = get_movie_poster(title, year)
            st.image(poster_url)




