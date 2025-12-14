import streamlit as st
import pandas as pd
from api import get_movie_poster, get_movie_summary, get_movie_cast, get_movie_runtime
from streamlit_star_rating import st_star_rating
from database import save_rating, get_rating, add_to_watchlist, get_initial
from Home import fix_title

st.markdown("""
<style>
[data-testid="stSidebar"] svg {
    fill: #111111 !important;   
    color: #111111 !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #111111 !important;
    color: #111111 !important
    
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


st.markdown("""
<style>

section {
    background-color: #111111 !important;
    color: #FFF !important;
}


h1, h2, h3, h4, h5 {
    color: #d72a18 !important;  
    margin-top: 10px;
    margin-bottom: 10px;
    text-align: center;
}


p, span, label {
    color: #FFFFFF !important;
    font-size: 16px;
    margin: 0;
}


div[data-testid="stButton"] button {
    background-color: #d72a18 !important;
    color: #111111 !important;
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: bold;
    cursor: pointer;
    transition: 0.2s ease-in-out;
}
div[data-testid="stButton"] button:hover {
    background-color: #b02114 !important;
    transform: scale(1.02);
}


div[data-testid="stTextInput"] input, div[data-baseweb="select"] > div {
    background-color: #111 !important;
    color: #111 !important;
    border: 1px solid #fff !important;
    border-radius: 6px;
    padding: 6px 10px;
}


ul {
    background-color: #111 !important;
    border: 1px solid #111 !important;
    color: #111111 !important;
}
ul li {
    background-color: #111 !important;
    color: #fff !important;
    border: 1px solid #111 !important;
}
ul li:hover {
    background-color: #d72a18 !important;
    color: #FFFFFF !important;
}
-----

div[data-baseweb="select"] svg {
    color: #fff !important;                  
}

/* Warning / success / error messages */
div[data-testid="stAlert"] {
    border-left: 5px solid #d72a18 !important;
    background-color: #2b2b2b !important;
    color: #FFFFFF !important;
    padding: 12px;
    border-radius: 8px;
    font-weight: bold;
    margin: 10px 0;
}


.stImage img {
    display: block;
    margin-left: auto;
    margin-right: auto;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
[data-testid="stSidebar"] svg {
    fill: #111111 !important;   
    color: #111111 !important;
}

</style>
""", unsafe_allow_html=True)


# Check if user is logged in
if "authenticated" not in st.session_state or not st.session_state.authenticated:
    st.switch_page("Home.py")

# Check if initial movies have been rated
if get_initial(st.session_state.username) == 0:
    st.switch_page("Home.py")

def star_rating(rating, max_stars=5):
    full_star = "★"
    empty_star = "☆"

    stars = full_star * round(rating)
    stars += empty_star * (max_stars - len(stars))
    return stars

def show_info(movie):
    if isinstance(movie, dict):
        full_title = movie["title"]
    elif isinstance(movie, list):
        full_title = movie[0]  
    else:
        full_title = movie 

    if " (" in full_title:
        title, year = full_title.rsplit(" (", 1)
        year = year.replace(")", "")
    else:
        title = full_title
        year = "Unknown"
    duration = get_movie_runtime(title, year)

    col1, col2 = st.columns([2, 1])  # ajustamos proporción para que la poster no sea demasiado grande

    with col1:
        # Título rojo
        st.markdown(f"<h2 style='color:#d72a18; text-align:center;'>{fix_title(title)}</h2>", unsafe_allow_html=True)
        
        # Detalles
        selected_genre = st.session_state.df.loc[st.session_state.df["title"] == full_title, "genres"].values[0]
        st.markdown(f"<p style='color:#FFFFFF; font-size:18px; margin:2px 0;'>Genre: {selected_genre}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#FFFFFF; font-size:18px; margin:2px 0;'>Year: {year}</p>", unsafe_allow_html=True)
        if duration:
            st.markdown(f"<p style='color:#FFFFFF; font-size:18px; margin:2px 0;'>Duration: {duration} mins</p>", unsafe_allow_html=True)

        # Rating promedio
        avg_ratings_df = st.session_state.avg_ratings
        rating_row = avg_ratings_df[avg_ratings_df["title"] == full_title]
        if not rating_row.empty:
            rating = rating_row["rating"].values[0]
            st.markdown(f"<p style='color:#FFFFFF; font-size:18px; margin:2px 0;'>Average Rating: {star_rating(rating)}</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:#FFFFFF; font-size:16px; margin:2px 0;'>Average Rating: N/A</p>", unsafe_allow_html=True)

        st.markdown("")
        st.markdown("")
        summary = get_movie_summary(title, year)
        cast = get_movie_cast(title, year)

        # Resumen y elenco
        if summary:
            st.markdown(f"<p style='color:#FFFFFF; font-size:16px;'>{summary}</p>", unsafe_allow_html=True)
        if cast:
            st.markdown(f"<p style='color:#FFFFFF; font-size:16px;'><strong>Top Cast:</strong> {', '.join(cast)}</p>", unsafe_allow_html=True)

        st.markdown("")
        st.markdown("")

        if st.button("Add to Watchlist", key=f"watchlist_{movie}"):
            add_to_watchlist(st.session_state.username, title, year)
            st.success(f"{title} added to watchlist!")


    with col2:
        
        with st.spinner("Loading movie poster..."):
            poster_url = get_movie_poster(title, year)
            summary = get_movie_summary(title, year)
            cast = get_movie_cast(title, year)

        if poster_url:
            st.image(poster_url, width=250)
        else:
            st.warning("Movie poster not found.")
        st.markdown("")
        st.markdown("")
        st.markdown("<p style='color:#d72a18; font-size:32px; font-weight:bold;'>Your Rating:</p>", unsafe_allow_html=True)

        st.markdown("")

        # Tu rating     
        previous_rating = get_rating(st.session_state.username, title)

        

        rating = st_star_rating(
            label="", 
            maxValue=5, 
            defaultValue=previous_rating, 
            key=f"rating_{movie}",
            size=22,
            dark_theme=True
        )
           # Botones
        st.markdown("""
        <style>
        div[data-testid="stButton"] button {
            background-color: #d72a18 !important;
            color: #111111 !important;
            border-radius: 8px;
            padding: 6px 20px;
            font-weight: bold;
            cursor: pointer;
            margin-right: 10px;
            transition: 0.2s ease-in-out;
        }
        div[data-testid="stButton"] button:hover {
            background-color: #b02114 !important;
            transform: scale(1.02);
        }

        div[data-testid="stButton"] {
            display: flex;
            justify-content: center;
        }
        </style>
        """, unsafe_allow_html=True)


        if st.button("Save Rating", key=f"Save_rating_{movie}"):
            if title and rating > 0:
                save_rating(st.session_state.username, title, rating)
                st.success("Rating saved!")
            else:
                st.warning("Please select a rating to save.")


st.title("Movie Details")

st.session_state.select = []
options = st.session_state.df["title"]
selected = st.multiselect("Search for a movie:", options, default=st.session_state.selected_movie, key="movie_search_select")

if selected:
    for movie_title in selected:
        show_info(movie_title)
    

