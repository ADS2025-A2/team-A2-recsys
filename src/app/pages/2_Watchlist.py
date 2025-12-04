import streamlit as st
from database import get_initial, get_watchlist, remove_from_watchlist
from api import get_movie_poster

# Check if user is logged in
if "authenticated" not in st.session_state or not st.session_state.authenticated:
    st.switch_page("Home.py")

# Check if initial movies have been rated
if get_initial(st.session_state.username) == 0:
    st.switch_page("Home.py")

st.title("Watchlist")

watchlist = get_watchlist(st.session_state.username)
st.write(watchlist)

for title, year in watchlist:
    col1, col2 = st.columns([2,1])
    with st.container():
        with col1:
            st.subheader(title)
            if st.button("Info", key=f"info_{title}"):
                st.switch_page("pages/1_Details.py")
            if st.button("Remove from Watchlist", key=f"remove_{title}"):
                remove_from_watchlist(st.session_state.username, title)
                st.success("Movie removed from watchlist.")
                st.rerun()
        with col2:
            poster_url = get_movie_poster(title, year)
            st.image(poster_url)


