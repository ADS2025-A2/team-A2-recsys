import streamlit as st

if "authenticated" not in st.session_state or not st.session_state.authenticated:
    st.switch_page("Home.py")

st.title("Watchlist")