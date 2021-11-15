"""
Also see:
Using Machine Learning with Streamlit, Chapter 4, in Getting Started with Streamlit for Data Science
"""

import streamlit as st
import pandas as pd

import pickle
from PIL import Image

# model

# see streamlit documentation:
# this replaces st.cache when you want to cache data
@st.experimental_memo()
def load_model():
	file = open('iris.pickle', 'rb')
	model = pickle.load(file)
	file.close()
	return model

@st.experimental_memo()
def load_image(species):
	if species == "Setosa":
		return Image.open("images/iris-setosa.jpeg")
	elif species == "Versicolor":
		return Image.open("images/iris-versicolor.jpeg")
	else:
		return Image.open("images/iris-virginica.jpeg")

# view

def show_species(species):
	st.image(load_image(species), width=400,
		caption="The Iris is of the species {}".format(species))

# controller

def get_user_inputs():
	st.sidebar.header("Enter dimensions")
	sepal_length = st.sidebar.slider("Sepal length", min_value=4.0, max_value=8.0)
	sepal_width = st.sidebar.slider("Sepal width", min_value=2.0, max_value=5.0)
	petal_length = st.sidebar.slider("Petal length", min_value=1.0, max_value=7.0)
	petal_width = st.sidebar.slider("Petal width", min_value=0.0, max_value=3.0)
	return sepal_length, sepal_width, petal_length, petal_width

# main

st.title("Find your Iris")
sepal_length, sepal_width, petal_length, petal_width = get_user_inputs()
model = load_model()
prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
species = prediction[0]
show_species(species)



