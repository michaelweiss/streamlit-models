# streamlit-models

These are a set of examples of using [streamlit](https://streamlit.io) to create machine learning applications. The code draws on various sources identified in the respective files.

## Hello World

The first one is just a example of creating a basic streamlit application that collects some user input and does something with it. Run this example using:

```
streamlit run hello.py

```

Then go to your browser and connect to `localhost:8501`.

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

This example combines training a machine learning model with using it in one app. The application trains a spam classifier on the data from the [SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) dataset. The basic structure of the application is similar to the `iris` example. However, since the data is unstructured, textual data, we need to convert it into a bag-of-words representation to obtain features that we can use to train a machine learning model.

To use the app run:

```
streamlit run spam.py

```
