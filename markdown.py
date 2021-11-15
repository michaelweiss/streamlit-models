import streamlit as st

# see also:
# https://www.pythontutorial.net/python-basics/python-read-text-file/

def read_markdown(path):
	with open(path) as file:
		return file.read()

# see also:
# https://pmbaumgartner.github.io/streamlitopedia/markdown.html

markdown = read_markdown("resources.md")
st.markdown(markdown, unsafe_allow_html=True)