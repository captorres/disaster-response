import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
#nltk.download('punkt')
nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import SGDClassifier
import re
import pickle

def load_data(database_filepath):

    # load data from database
    engine = create_engine('sqlite:///' + database_filepath + '.db')
    df = pd.read_sql("SELECT * FROM MessagesCategories", engine)

    # Defining X and y data
    X = df['message']
    y = df.drop(['id','message','original','genre'], axis=1)

    return X, y, y.columns


def customTokenize(text):

    # Transform to lowercase and remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenize words
    words = word_tokenize(text)

    # Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]

    # Reduce words to their root form
    #words = [WordNetLemmatizer().lemmatize(w) for w in words]
    words = [PorterStemmer().stem(w) for w in words]

    return words


def build_model():

    pipeline = Pipeline([
            ('countvect', CountVectorizer(tokenizer=customTokenize)),
            ('tfidf', TfidfTransformer()),
            ('multi_sgdc', MultiOutputClassifier(SGDClassifier(max_iter=1000, tol=0.001)))
            ])

    parameters = {
       # 'countvect__ngram_range': ((1, 1),(1, 2)),
        'multi_sgdc__estimator__alpha': [0.0001]#, 0.01, 10, 100]
         }

    # create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):

    # predict on test data
    y_pred=model.predict(X_test)
    df_pred = pd.DataFrame(y_pred, columns=category_names)

    for target in category_names:
        print('\n', target)
        print(classification_report(Y_test[target], df_pred[target]))


def save_model(model, model_filepath):
    with open(model_filepath,'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        #print('Best parameters:', model.best_params_)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
