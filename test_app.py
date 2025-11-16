import streamlit as st
import pandas as pd

st.title("Hi")

st.subheader("Test")

data = {'a':1, 'b':2, 'c':3}
df = pd.DataFrame([data])
st.write(df)

check = st.checkbox(label='question?')

if check:
    st.subheader("Answer")
