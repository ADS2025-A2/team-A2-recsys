
import streamlit as st

def st_star_rating(label, max_stars=5, key=None):
    """
    Star rating simple usando un slider
    """
    rating = st.slider(label, min_value=0, max_value=max_stars, step=1, key=key)
    st.write("⭐" * rating)
    return rating
