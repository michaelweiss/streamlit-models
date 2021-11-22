# -*- coding: utf-8 -*-
# simple end-to-end process to create a topic co-occurrence matrix

import streamlit as st
import pandas as pd

# the gensim library is used for topic modeling
from gensim.models import LdaModel
from gensim.corpora import Dictionary

# model

@st.experimental_memo()
def load_corpus(file):
	documents = pd.read_csv(file)
	return documents

def read_stopwords(file):
	file = open(file, 'r')
	return [w.strip() for w in file.read().split('\n')]

def tokenize(document):
	return [w.lower() for w in document.split()]

def corpus_to_tokens(corpus, user_stopwords):
	stopwords_en = read_stopwords("data/stopwords-en.txt")
	user_stopwords = [word.strip() for word in user_stopwords.split('\n')]
	stopwords = stopwords_en + user_stopwords
	# remove stopwords and numbers
	return [[w for w in tokenize(document) if w not in stopwords and w.isalnum()]
		for document in corpus['content']]

def tokens_to_bow(tokens, dictionary):
	return [dictionary.doc2bow(document) for document in tokens]

def fit_lda(tokens, number_of_topics, dictionary):
	return LdaModel(tokens_to_bow(tokens, dictionary), number_of_topics, dictionary)

def ids_to_words(bow, dictionary):
	return [(dictionary.id2token[w], f) for w, f in bow]

def document_topics_matrix(tokens, dictionary):
	return [lda.get_document_topics(bow) for bow in tokens_to_bow(tokens, dictionary)]

def topic_co_occurrence_matrix(dtm, min_weight=0.1):
	return [[t for t, w in topics if w >= min_weight] for topics in dtm]

def tcom_to_sentences(tcom):
	for tco in tcom:
		tco = ["T{}".format(t) for t in tco]
		tco.append('.')
	return "\n".join(tco)

# view

def show_corpus(corpus):
	for i, document in enumerate(corpus['content']):
		print(i, document)

def show_topics(lda, number_of_topics):
	# show top 10 keywords for each topic
	topics_df = pd.DataFrame([[", ".join([tw[0] for tw in lda.show_topic(t, 10)])] 
		for t in range(number_of_topics)], columns=['Keywords'])
	st.table(topics_df)

def show_document_topics_matrix(dtm):
	for i, topics in enumerate(dtm):
		print(i, topics)

# main

st.sidebar.title("TME")

corpus_file = st.sidebar.file_uploader("Corpus", type="csv")
user_stopwords = st.sidebar.text_area("Stopwords (one per line)")

st.header("Corpus")
if corpus_file is not None:
	corpus = load_corpus(corpus_file)
	st.write(corpus)
else:
	st.markdown("Please upload a corpus. The csv file should contain at least a 'name' and a 'content' column.")

st.header("Preprocessed corpus")
tokens = corpus_to_tokens(corpus, user_stopwords)
st.write(tokens[:2])

st.header("Topics")
number_of_topics = st.sidebar.slider("Number of topics", min_value=1, max_value=50, value=10)
dictionary = Dictionary(tokens)
lda = fit_lda(tokens, number_of_topics, dictionary)
show_topics(lda, number_of_topics)

# dtm = document_topics_matrix(tokens, dictionary)
# show_document_topics_matrix(dtm)
# tcom = topic_co_occurrence_matrix(dtm, 0.1)

# tcom_to_sentences(tcom)


