import streamlit as st
from api import get_movie_poster, get_movie_summary, get_movie_cast

def show_info(movie):
    full_title = movie[0]
    title, year = full_title.rsplit(" (", 1)
    year = year.replace(")", "")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(title)
        selected_genre = st.session_state.df.loc[st.session_state.df["title"] == full_title, "genres"].values[0]
        st.write("Genre:", selected_genre)
        st.write("Year:", year)

    with col2:
        with st.spinner("Loading movie poster..."):
            poster_url = get_movie_poster(title, year)
            summary = get_movie_summary(title, year)
            cast = get_movie_cast(title, year)

        if poster_url:
            st.image(poster_url, width=200)
        else:
            st.warning("Movie poster not found.")
        
    st.write(summary)
    st.write("**Top Cast:**", ", ".join(cast))
    


if "authenticated" not in st.session_state or not st.session_state.authenticated:
    st.switch_page("Home.py")

st.title("Movie Details")

st.session_state.select = []
options = st.session_state.df["title"]
selected = st.multiselect("Search for a movie:", options)

if selected:
    show_info(selected)

if st.session_state.selected_movie:
    show_info(st.session_state.selected_movie)
    st.session_state.selected_movie = None


