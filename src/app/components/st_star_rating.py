import streamlit as st

def st_star_rating(label, maxValue=5, defaultValue=0, key=None):
    """
    Star rating simple usando un slider
    """
    rating = st.slider(label, min_value=0, max_value=maxValue, value=defaultValue, step=1, key=key)
    st.write("⭐" * rating)
    return rating
