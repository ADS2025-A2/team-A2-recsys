import streamlit as st
import pandas as pd
from api import get_movie_poster, get_movie_summary, get_movie_cast, get_movie_runtime
from streamlit_star_rating import st_star_rating
from database import save_rating, get_rating, add_to_watchlist, get_initial

# Check if user is logged in
if "authenticated" not in st.session_state or not st.session_state.authenticated:
    st.switch_page("Home.py")

# Check if initial movies have been rated
if get_initial(st.session_state.username) == 0:
    st.switch_page("Home.py")

def star_rating(rating, max_stars=5):
    full_star = "★"
    empty_star = "☆"
    half_star = "⯨"

    stars = full_star * int(rating)
    if rating - int(rating) >= 0.5:
        stars += half_star
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

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(title)
        selected_genre = st.session_state.df.loc[st.session_state.df["title"] == full_title, "genres"].values[0]
        st.markdown(
                f"<p style='font-size:20px; color:#111; margin:0;'>Genre: {selected_genre}</p>",
                unsafe_allow_html=True
            )
        st.markdown(
                f"<p style='font-size:20px; color:#111; margin:0;'>Year: {year}</p>",
                unsafe_allow_html=True
            )
        if duration:
            st.markdown(
                    f"<p style='font-size:20px; color:#111; margin:0;'>Duration: {duration} mins</p>",
                    unsafe_allow_html=True
                )

        avg_ratings_df = st.session_state.avg_ratings
        rating_row = avg_ratings_df[avg_ratings_df["title"] == full_title]

        if not rating_row.empty:
            rating = rating_row["rating"].values[0]
            st.markdown(
                f"<p style='font-size:20px; color:#111; margin:0;'>Average Rating: {star_rating(rating)}</p>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<p style='font-size:16px; color:#111; margin:0;'>Average Rating: N/A</p>",
                unsafe_allow_html=True
            )
        
        previous_rating = get_rating(st.session_state.username, title)
        rating = st_star_rating("Your Rating:", maxValue=5, defaultValue=previous_rating, key=f"rating_{movie}")
        if st.button("Save Rating", key=f"Save_rating_{movie}"):
            if title and rating > 0:
                save_rating(st.session_state.username, title, rating)
                st.success("Rating saved!")
            else:
                st.warning("Please select a rating to save.")
        
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
        
    st.write(summary)
    st.write("**Top Cast:**", ", ".join(cast))
    

st.title("Movie Details")

st.session_state.select = []
options = st.session_state.df["title"]
selected = st.multiselect("Search for a movie:", options)

if selected:
    for movie_title in selected:
        show_info(movie_title)

if st.session_state.selected_movie:
    show_info(st.session_state.selected_movie)
    st.session_state.selected_movie = None


