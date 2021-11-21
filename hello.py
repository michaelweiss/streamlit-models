import streamlit as st

st.sidebar.title("Info")
st.sidebar.markdown("Use this sidebar for controls")
st.sidebar.write("More information")

st.title("Quiz")
answer = st.slider("What is the answer to life, the universe and everything?", min_value=1, max_value=50, step=1)
if answer == "42":
	st.write("Your answer is correct")
else:
	st.write("Try again")
