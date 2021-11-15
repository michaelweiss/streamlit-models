import streamlit as st
import pandas as pd

# use this library for loading images
from PIL import Image

# use these libraries for pre-processing text
import nltk
from nltk.stem.snowball import SnowballStemmer
import re
import string

# use these libraries to calculate SBS metrics
from collections import Counter
import numpy as np

# use this library for social network analysis
import networkx as nx
from pyvis.network import Network

# use this library for streamlit components
import streamlit.components.v1 as components

# see also:
# https://towardsdatascience.com/calculating-the-semantic-brand-score-with-python-3f94fb8372a6

# model

# Read text documents from example files
# In this example, we used the Guardian API in Orange to collect news articles about NFTs
@st.experimental_memo()
def load_corpus():
	return pd.read_csv("data/guardian-nft.csv")

# Extract text documents from the corpus
def extract_documents(corpus):
	return [headline + ". " + content for headline, content in
		zip(corpus["Headline"], corpus["Content"])]

# Pre-process the text documents
# Exclude brand names from pre-processing
# Don't use these steps automatically, but ask yourself if they apply to your data. For example,
# stemming can cause you to lose information (eg whether the word was a noun a verb)
# You could also include other steps, for example, to keep only the nouns
def preprocess(documents, brands):
	# Convert to lower case
	documents = [document.lower() for document in documents]

	# Remove words and punctuation
	# Note that stopword removal assumes that words are lower case
	documents = remove_words_and_punctuation(documents)

	return documents

def tokenize_and_stem(documents, brands):
	# Tokenize the documents
	documents = tokenize(documents)

	# Stem words in the documents
	documents = stem(documents, brands)

	return documents

# Read stopwords
# Here we are just using the standard stopwords from nltk, but we could also
# use our own stopwords or let the user define additional ones
@st.experimental_memo()
def read_stopwords():
	nltk.download("stopwords")
	return nltk.corpus.stopwords.words('english')

# Remove words and punctuations
# Note that there are other ways to do this, and we will look at some
def remove_words_and_punctuation(documents):
	# Define stopwords
	stopwords = read_stopwords()

	# Remove words that start with HTTP
	documents = [re.sub(r"http\S+", " ", document) for document in documents]

	# Remove words that start with WWW
	documents = [re.sub(r"www\S+", " ", document) for document in documents]

	# Remove punctuation
	regex = re.compile('[%s]' % re.escape(string.punctuation))
	documents = [regex.sub(' ', document) for document in documents]

	# Remove words made of single letters
	# Note that we could also use len(s) to get the number of letters in a string,
	# instead of using a regular expression to do this
	documents = [re.sub(r'\b\w{1}\b', ' ', document) for document in documents]

	# Remove unicode punctuation symbols
	# I added this step since unicode punctuation symbols were not included above
	# Whether you need to do this depends on the source of your data
	# Eg if you collected some of your data in a spreadsheet like Excel, the spreadsheet
	# software often converts normal quotes (") etc. to fancy quotes (“ and ”)
	documents = [re.sub(r'[“”‘’–…]+', ' ', document) for document in documents]

	# Remove stopwords
	# Note that an easier way of doing this would be to tokenize the document first,
	# then check for the words in the list of tokens if they are stopwords
	pattern = re.compile(r'\b(' + r'|'.join(stopwords) + r')\b\s*')
	documents = [pattern.sub(' ', document) for document in documents]

	# Remove extra whitespaces
	documents = [re.sub(' +', ' ' , document) for document in documents]

	return documents

# Tokenize text documents (becomes a list of lists)
# Note that this could also be done by an nltk method that take special cases into account
def tokenize(documents):
	return [document.split() for document in documents]

# Stem words in text documents
# Snowball Stemming
def stem(documents, brands):
	stemmer = SnowballStemmer("english")
	return [[stemmer.stem(word) if word not in brands else word for word in document] for document in documents]

# Calculate prevalence, which counts the frequency of occurrence of each brand name 
def calculate_prevalence(documents, brands):
	# Top n words
	top_n = 100

	# Create a dictionary with frequency counts for each word
	countPR = Counter()
	for document in documents:
		countPR.update(Counter(document))

	# Keep the top_n most common words
	countPR = dict(countPR.most_common(top_n))

	# Calculate average score and standard deviation
	avgPR = np.mean(list(countPR.values()))
	stdPR = np.std(list(countPR.values()))

	# Calculate standardized prevalence for each brand
	prevalence = {}
	for brand in brands:
		prevalence[brand] = (countPR[brand] - avgPR) / stdPR
		
	return prevalence

def frequency_count(documents, top_n=100):
	# Create a dictionary with frequency counts for each word
	count = Counter()
	for document in documents:
		count.update(Counter(document))

	# Keep the top_n most common words
	count = dict(count.most_common(top_n))

	return count

# Transform texts (list of lists of tokens) into a social network where nodes are words and 
# links are weighted according to the number of co-occurrences between each pair of words
def create_co_occurrence_network(documents, brands, top_n=100, link_filter=2):
	# Choose a co-occurrence range
	# Note that this could be a user-settable variable
	co_range = 7

	# Create an undirected network graph
	G = nx.Graph()

	# Collect the most frequent words, including brands
	top_n_words = list(frequency_count(documents, top_n).keys()) + brands

	# Each word is a network node
	# Instead of the following, we only use the top N words
	# nodes = set([word for document in documents for word in document])
	nodes = top_n_words
	G.add_nodes_from(nodes)

	# Add links based on co-occurrences
	for document in documents:
		word_list = []
		length = len(document)

		for k, w in enumerate(document):
			# Skip if the word is not in the top N words or a brand name
			if w not in top_n_words:
				continue

			# Define range, based on document length
			if (k + co_range) >= length:
				superior = length
			else:
			 	superior = k + co_range + 1

			# Create the list of co-occurring words
			if k < length - 1:
				for i in range(k + 1, superior):
					linked_word = document[i]
					if linked_word in nodes:
						word_list.append(linked_word)
				
			# If the list is not empty, create the network links
			if word_list:
				for p in word_list:
					# Do not include loops
					if w != p:
						if G.has_edge(w, p):
							G[w][p]['weight'] += 1
						else:
							G.add_edge(w, p, weight=1)

			word_list = []

	# Remove negligible co-occurrences based on a filter
	# Create a new graph which has only links above the minimum co-occurrence threshold
	G_filtered = nx.Graph() 
	G_filtered.add_nodes_from(G)
	for u, v, data in G.edges(data=True):
	    if data['weight'] >= link_filter:
	        G_filtered.add_edge(u, v, weight=data['weight'])

	# Optional removal of isolates
	isolates = set(nx.isolates(G_filtered))
	isolates -= set(brands)
	G_filtered.remove_nodes_from(isolates)

	return G_filtered

# view

def show_figure(url, width="100%", caption=""):
	figure = """
		<center>
			<figure>
				<img src="{}" width="{}">
				<figcaption>{}</figcaption>
			</figure>
		<center>
		"""
	st.markdown(figure.format(url, width, caption),
		unsafe_allow_html=True)

def show_co_occurrence_network(G):
	network = Network('600px', '600px')
	network.from_nx(G)
	network.show("co_occurrence_network.html")
	with st.container():
		components.html(open("co_occurrence_network.html", 'r', encoding='utf-8').read(), height=600)

# controller

# main

st.title("Semantic Brand Score")

st.sidebar.title("Contents")
st.sidebar.markdown("""
	* [Overview](#overview)
	* [Step 1](#step-1-collect-textual-data)
	* [Step 2](#step-2-remove-punctuation-special-characters-html-tags-and-stopwords)
	* [Step 3](#step-3-tokenization-and-stemming)
	* [Step 4](#step-4-from-texts-to-networks)
	* [Step 5](#step-5-link-filtering)
	* [Metrics](#metrics)
	""")

st.header("Overview")

st.markdown("""
	The code in this app builds on this article:
	[Calculating the Semantic Brand Score](https://towardsdatascience.com/calculating-the-semantic-brand-score-with-python-3f94fb8372a6).
	Computing the Semantic Brand Score involves the following steps:
	""")
show_figure("https://miro.medium.com/max/1400/1*k_6ixQSNVGo3c7xhjDi5zg.jpeg", width="80%")

st.header("Step 1: Collect textual data")

st.markdown("""
	In this example, we used the Guardian API in Orange to collect news articles about NFTs.
	The time period covered by the articles was: Nov 13, 2020 to Nov 13, 2021.
	""")
corpus = load_corpus()
st.write(corpus)

# In the remaining analysis we focus on just the text of the documents
documents = extract_documents(corpus)
brands = ["beeple", "opensea", "cat", "facebook"]

st.header("Step 2: Remove punctuation, special characters, HTML tags, and stopwords")

st.markdown("""
	Remove punctuation, stopwords, and special characters.
	""")
documents = preprocess(documents, brands)
st.write(documents[:3])

st.header("Step 3: Tokenization and stemming")

st.markdown("""
	Tokenize and stem documents.
	""")
documents = tokenize_and_stem(documents, brands)
st.write(documents[:3])

st.header("Step 4: From texts to networks")

st.markdown("""
	Create word co-occurrence network. 
	""")

words = set([word for document in documents for word in document])
percentage_of_words = st.slider("% of words", min_value=0, max_value=100, value=10)
max_words = int(len(words) * percentage_of_words/100)

st.header("Step 5: Link filtering")

st.markdown("Filter network links based on their weight.")

min_link_weight = st.slider("Minimum link weight", min_value=1, max_value=50, value=2)

G = create_co_occurrence_network(documents, brands, top_n=max_words, link_filter=min_link_weight)
st.write("Number of nodes", len(G.nodes))
st.write("Number of edges", len(G.edges))
show_co_occurrence_network(G)

st.header("Metrics")

prevalence = calculate_prevalence(documents, brands)
for brand in brands:
	st.write("prevalence_{}".format(brand), prevalence[brand])


