# streamlit-models

These are a set of examples of using [streamlit](https://streamlit.io) to create machine learning applications. The code draws on various sources identified in the respective files.

## Hello World

The first one is just a example of creating a basic streamlit application that collects some user input and does something with it.

Run the example using:

```
streamlit run hello.py

```

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

