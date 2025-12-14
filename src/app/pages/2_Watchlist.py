import streamlit as st
from database import get_initial, get_watchlist, remove_from_watchlist
from api import get_movie_poster
from Home import fix_title

import streamlit as st

st.markdown(
    """
    <style>
    .stApp, .css-18e3th9, .css-1outpf7, .css-ffhzg2 {
        background-color: #111;
        color: white;
    }

    h1, h2, h3, h4, h5, h6 {
        color: white;
    }

    p, span, label {
        color: white;
    }

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

[data-testid="stSidebar"] {
    background-color: #111111 !important;
    
}


[data-testid="stSidebarNav"] span {
    color: #FFFFFF !important;      
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

st.markdown(
    """
    <style>
    div.stButton > button {
        background-color: #d72a18 !important;  
        color: black !important;             
        font-weight: bold;
        border-radius: 8px;
        padding: 8px 16px;
        text-align: center;                  
    }

    div.stButton > button:hover {
        background-color: #e65b4f !important;  
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
        color: #d72a18;  
        text-align: center; 
        font-size: 60px;   
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
    font-family: 'Montserrat', sans-serif !important;
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




