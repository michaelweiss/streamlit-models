import streamlit as st

st.sidebar.title("Info")
st.sidebar.markdown("Use this sidebar for controls")

st.title("Quiz")
answer = st.text_input("What is the answer to life, the universe and everything?")
if answer == "42":
	st.write("Your answer is correct")
else:
	st.write("Try again")
