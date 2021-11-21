"""
Inspired by:
https://github.com/MarcSkovMadsen/awesome-streamlit/blob/master/gallery/iris_classification/iris.py
https://medium.com/swlh/building-interactive-dashboard-with-plotly-and-streamlit-2c390bcfd41a

Also see:
Using Machine Learning with Streamlit, Chapter 4, in Getting Started with Streamlit for Data Science
"""

import streamlit as st
import pandas as pd

# https://plotly.com/python/plotly-express/
import plotly.express as px

# https://scikit-learn.org/stable/
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

import pickle

# model

# See streamlit documentation:
# this replaces st.cache when you want to cache data
@st.experimental_memo()
def load_data():
	return pd.read_csv("data/iris.csv")

def train_model(features, target, model):
	x_train, x_test, y_train, y_test = train_test_split(
		features, target, train_size=0.3, random_state=0)
	model.fit(x_train, y_train)
	accuracy = model.score(x_test, y_test)
	y_predicted = model.predict(x_test)
	conf_matrix = confusion_matrix(y_test, y_predicted)
	return y_predicted, accuracy, conf_matrix

def save_model(model):
	file = open('models/iris.pickle', 'wb')
	pickle.dump(model, file)
	file.close()

# view

def show_data(source_df):
	st.write(source_df)

# Show a scatter plot of two features for the selected species
def show_scatter_plot(selected_species_df):
	st.subheader("Scatter plot")
	axis_x = st.selectbox("Choose feature on axis x", selected_species_df.columns[0:4])
	axis_y = st.selectbox("Choose feature on axis y", selected_species_df.columns[0:4])
	# here we use the plotly library for creating visualizations
	# usually, only a few commands are required to create a plotly visualization
	fig = px.scatter(selected_species_df, x=axis_x, y=axis_y, color="variety")
	st.plotly_chart(fig)

# Show a histogram plot of the selected species and feature
def show_histogram_plot(selected_species_df):
	st.subheader("Histogram plot")
	feature = st.selectbox("Choose feature", selected_species_df.columns[0:4])
	fig = px.histogram(selected_species_df, x=feature, color="variety")
	st.plotly_chart(fig)

# Show the performance of a machine learning model
def show_machine_learning_model(source_df):
	features = source_df[["sepal.length", "sepal.width", "petal.length", "petal.width"]].values
	target = source_df["variety"].values
	show_features_and_target(features, target)
	try:
		model = select_algorithm()
		st.write(model)
		y_predicted, accuracy, conf_matrix = train_model(features, target, model)
		st.write("Accuracy:", accuracy.round(2))
		st.write("Confusion matrix:", conf_matrix)
		save_model(model)
	except:
		st.info("Algorithm not supported yet")

def show_features_and_target(features, target):
	# use containers to put two tables side-by-side
	cols = st.columns(2)
	with cols[0]:
		st.subheader("Features")
		st.write(features)
	with cols[1]:
		st.subheader("Target")
		st.write(target)

# controller

# select the species to explore
def select_species(source_df):
	selected_species = st.multiselect("Select varieties for further exploration", 
		source_df["variety"].unique())
	selected_species_df = source_df[source_df["variety"].isin(selected_species)]
	if not selected_species_df.empty:
		st.write(selected_species_df)
	else:
		st.info("Select one or multiple varieties to explore")
	return selected_species_df

# select the algorithm to use for the model
# this illustrates how you can allow the user to choose between algorithms
def select_algorithm():
	algorithms = ["Decision Tree", "Support Vector Machine", "Logistic Regression"]
	classifier = st.selectbox("Choose the algorithm to use?", algorithms)
	if classifier == "Decision Tree":
		model = DecisionTreeClassifier()
	elif classifier == "Support Vector Machine":
		model = SVC()
	elif classifier == "Logistic Regression":
		model = LogisticRegression()
	else:
		raise NotImplementedError()
	return model

# main

st.title("Iris classifier")	

st.header("Explore the data")
source_df = load_data()
if st.checkbox("Show the full dataset"):
	show_data(source_df)
selected_species_df = select_species(source_df)
if not selected_species_df.empty:
	show_scatter_plot(selected_species_df)
	show_histogram_plot(selected_species_df)

st.header("Train the model")
show_machine_learning_model(source_df)



