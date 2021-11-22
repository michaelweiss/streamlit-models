# streamlit-models

These are a set of examples of using [streamlit](https://streamlit.io) to create machine learning applications. The code draws on various sources identified in the respective files.

## Hello World

In this first example, we just create a basic streamlit application that collects some user input and does something with it. Run this example using:

```
streamlit run hello.py

```

Then go to your browser and connect it to `localhost:8501`.

## Iris

Train a machine learning model using the Iris dataset. The code is structured into three sections following the MVC (Model View Controller) pattern. The model section is for loading the data, training and saving the machine learning model. The view section is for exploring the data and showing the results of training the machine learning model. The controller section is to ask the user for input. 

To train the machine learning model run:

```
streamlit run iris.py
```

To use the model to find the sepcies of Iris given its features run:

```
streamlit run iris_app.py
```

## Spam

This example combines training a machine learning model with using it in one application. The application trains a spam classifier on the data from the [SMS Spam Collection dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection). The basic structure of the application is similar to that of the Iris example. However, since the data is unstructured, textual data, we need to first convert it into a bag-of-words representation to obtain features that we can then use to train a machine learning model.

To use the application run:

```
streamlit run spam.py

```

## SBS

Computes the Semantic Brand Score from textual data. The SBS measures the importance of a brand. This code builds on the implementation in the post [Calculating the Semantic Brand Score with Python](https://towardsdatascience.com/calculating-the-semantic-brand-score-with-python-3f94fb8372a6). In the streamlit version of the code, we again adopt a MVC architecture. Computing the SBS involves five steps:

1. Collect textual data. We collected news articles from The Guardian on the topic of "NFT" published over the last year through the Guardian API (using Orange). We split each article into sentences and filtered the sentences that mention one of the brands.
2. Remove punctuation, special characters, HTML tags, and stopwords. We use a standard list of English stopwords in this case.
3. Tokenize documents, stem words, and remove all but the most frequent words. 
4. Construct and visualize a word co-occurrence network. Transform texts (list of lists of tokens) into a social network, where nodes are words and links are weighted according to the number of co-occurrences between each pair of words.
5. Filter links. Remove links with less than a given weight. 

Now compute prevalence, diversity, and connectivity metrics, and add them to produce the SBS. 

To produce a network that was easy to interpret, three steps proved to be important: split articles into sentences and focus the co-range for word occurrences to the sentence level, ensure that all sentences used in the analysis mention one of the brands, and remove all but the most frequent words. The resulting network was clearly focused on the brands and the key words associated with those brands. It also took less time to compute the network with those constraints in place.

