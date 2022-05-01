# Disaster response pipeline - NLP and machine learning

## Project motivation

Appen is an international company that provides data services for artificial intelligent projects. As described in their website, they are “the global leader in data for the AI Lifecycle, with over 25 years of experience in data sourcing, data annotation, and model evaluation by humans”.

One of their data sets contains real messages that were sent during disaster events. 

Following a disaster, a great amount of communication, including text messages, are sent to response organizations. As in these scenarios the human capacity of evaluating and providing help is limited and needs to be correctly prioritized, it is crucial to quickly identify the messages that really require a response action.

In the way that disasters are typically responded to, different organizations are responsible for different parts of the problem.  So it is also important that these messages are classified in a way that allows them to be correctly directed to an appropriate disaster relief agency.

In this context, this project’s objective is to create a machine learning model for an API that classifies disasters text messages using Appen (formally Figure 8) data. 
 

## Data

The data provided by Appen are divided into two sets:
* *disaster_messages.csv*: contains 26219 real text messages sent following disasters. 
* *disaster_categories.csv*: the labels, corresponding to 36 categories of interest that can be found in the messages and are relevant to the response agencies.

The *process_data.py* script is used to load, clean and merge the data. It also stores the resulting table in as SQLite repository with the file name *DisasterResponse.db*. This table is the input for the text processing and the machine learning algorithm.


## Methodology and packages

This project is basically divided into three components:
1. ETL pipeline for loading, cleaning and preparing the data to be used on training process of a machine learning model. For this purpose, the data will be stored in a SQLite database
2. NLP and Machine learning pipeline to split the data into a train and a test set, apply appropriate text processing and transformation, and train a multiclass supervised machine learning model. Parameter tuning is part of the pipeline but due to limited computational capacity, it is not really being used, as it is defined with only one set of parameters. The model is exported in a pickle file to be used in an web application
3. Flask Web App where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

Python 3 was used for building the ETL and machine learning pipelines. NLP algorithms were also used to process text data and prepare the features for the ML model.

Different ML tecniques were tested such as random forrest, logistic regression, naïve Bayes, gradient boosting, among others. However, stochastic gradient descent was the algorithm chosen to integrate the ML pipeline as it showed a slightly better accuracy than all the other techniques and has an efficient processing time.

So the packages used can be summarized as follows:

* ETL pipeline

    + sys, pandas and create_engine (from sqlalchemy)

* Text processing 

    + From nltk - word_tokenize (from tokenize), stopwords (from corpus), PorterStemmer (from stem.porter) 

    + From sklearn.feature_extraction.text – CountVectorizer and TfidfTransformer 

    + re (regular expression package)

* ML pipeline

    + From sklearn – Pipeline (from pipeline), train_test_split (from model_selection), classification_report (from metrics) GridSearchCV (from model_selection), MultiOutputClassifier (from multioutput) and SGDClassifier (from linear_model) 

    + pickle 

* Visualization: Plotly

In addition to the *process_data.py*, used to prepare the data, the *train_classifier.py* script performs the text processing, model training and testing steps. It also saves the model in a pickle file named *classifier.pkl* to be deployed in the web app.
The script that runs the app is also a Python file named *run.py*.


## Instructions

1. Run the following commands in the project's root directory to set up your database and model.

    1.1 To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

    1.2 To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following commands in the project's root directory to initialize the app.

    2.1 Go to `app` directory: `cd app`

    2.2 Run your web app: `python run.py`


## Results

The result generated by this study is a web application that allows the user to input a message and returns classification results for all 36 categories.

All the files used in this project can be found in [this]( https://github.com/captorres/disaster-response.git) Github repository.
