# Inspired by the Naive Bayes classifier example in Data Sceince and Analytics with Python
# The data is from the SMS Spam Collection dataset at https://archive.ics.uci.edu/ml/datasets/sms+spam+collection

# Also see the text classification example in:
# Rogel-Salazar, Data Science and Analytics with Python, Chapter 6

import streamlit as st
import pandas as pd

# https://scikit-learn.org/stable/
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix

# saving and loading models
import pickle

# file utilities
import os.path

# model

# See streamlit documentation:
# this replaces st.cache when you want to cache data
@st.experimental_memo()
def load_data():
	return pd.read_csv("data/spam.csv")

# Extract features
# A CountVectorizer is used to extract a bag-of-words representation from the textual data
def extract_features(source_df):
	# Preprocess text of a message
	source_df['text'] = source_df['text'].apply(preprocess_message)
	
	# Extract features using CountVectorizer
	# Usually, we would set lowercase to True. However, here we are dealing with spam, and
	# spam often contains all-caps words, so we don't want to lose this information.
	vectorizer = CountVectorizer(lowercase=False, binary=True)
	# We use fit_transform to create a dictionary of words from the corpus of messages,
	# and then transform each message in the corpus to a bag-of-words
	features = vectorizer.fit_transform(source_df['text'])

	return vectorizer, features

# Put any preprocessing of messages here
# In this case, we don't do any additional preprocessing, but you could use this to
# remove URLs from the message text, for example
def preprocess_message(message):
	return message

def train_model(features, target, model):
	x_train, x_test, y_train, y_test = train_test_split(
		features, target, train_size=0.3, random_state=0)
	model.fit(x_train, y_train)
	accuracy = model.score(x_test, y_test)
	y_predicted = model.predict(x_test)
	conf_matrix = confusion_matrix(y_test, y_predicted)
	return y_predicted, accuracy, conf_matrix

# See streamlit documentation:
# this replaces st.cache when you want to cache data
@st.experimental_memo()
def load_model():
	file = open('models/spam.pickle', 'rb')
	model = pickle.load(file)
	file.close()
	return model

def save_model(model):
	file = open('models/spam.pickle', 'wb')
	pickle.dump(model, file)
	file.close()

# view

def show_data(source_df):
	st.header("Data")
	st.write(source_df)

# Show the performance of a machine learning model
def show_machine_learning_model(source_df):
	st.header("Train model")
	
	# Extract features
	vectorizer, features = extract_features(source_df)
	target = source_df["type"].values
	try:
		# Select algorithm and train model
		model = select_algorithm()
		st.write(model)
		y_predicted, accuracy, conf_matrix = train_model(features, target, model)
		st.write("Accuracy:", accuracy.round(2))
		st.write("Confusion matrix:", conf_matrix)
		if st.button("Save model"):
			save_model(model)
	except:
		st.info("Algorithm not supported yet")

# controller

def select_algorithm():
	algorithms = ["Decision Tree", "Naive Bayes"]
	classifier = st.selectbox("Choose the algorithm to use?", algorithms)
	if classifier == "Decision Tree":
		model = DecisionTreeClassifier()
	elif classifier == "Naive Bayes":
		model = MultinomialNB()
	else:
		raise NotImplementedError()
	return model

# Check if a message is spam
# We need extract the features in the same way as when training the model
def check_message(source_df):
	message = st.text_input("Enter the text of a message")
	vectorizer, features = extract_features(source_df)
	model = load_model()
	# Note that here we use transform, not fit_transform
	prediction = model.predict(vectorizer.transform([message]))
	if prediction[0] == "ham":
		st.info("You are good to go. This looks like a normal message")
	else:
		st.warning("Better look out. This looks like spam")

# main 

st.title("Spam or Ham?")

source_df = load_data()
if os.path.exists('models/spam.pickle'):
	check_message(source_df)
	if st.checkbox("Update the model"):
		if st.checkbox("Show the dataset"):
			show_data(source_df)
		show_machine_learning_model(source_df)
else:
	show_machine_learning_model(source_df)

