#Import libraries
import sys
import nltk
nltk.download(['punkt', 'wordnet','stopwords','averaged_perceptron_tagger'])

class style:
    BOLD = '\033[1m'
    END = '\033[0m'
    
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import string
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
#from sklearn.svm import SVC
from sklearn.pipeline import Pipeline,FeatureUnion

from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,make_scorer,classification_report,fbeta_score

from sklearn.base import BaseEstimator, TransformerMixin
import pickle


def load_data(database_filepath):
    '''
    Description:
    Load data from the database filepath and read in the table name to extract the data,target and category names of the labels.
    
    Input:
    database filepath
    
    Output:
    X - data,Y - multioutput target,category names(labels)
    '''
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('DisasterMessages', engine)
    X =  df.message
    Y = (df[df.columns[4:]])
    category_names =Y.columns.tolist()
    
    return X, Y, category_names


def tokenize(text):
    '''
    Description :
    Normalize,tokenize,apply stemming and lemmatization to text data.
    
    Input: 
    text(str) containing message that needs to be tokenized.
    
    Output : 
    processed word tokens: normalized,tokenized,lemmatized and stemmed
    '''
    #message = text.translate(str.maketrans('','',string.punctuation)).lower() #normalize
    message=re.sub(r"[^a-zA-Z0-9]"," ",text) #replace special characters with spaces
    words = word_tokenize(message) # tokenize
    words = [w for w in words if w not in stopwords.words('english')] #apply stopwords
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in words] #lemmatize
    stemmed = [PorterStemmer().stem(w).lower().strip() for w in lemmed] # stem
    return stemmed


def build_model():
    '''
    Description:
    Create pipeline object and set the parameters for GridSearchCV
    
    Input:
    None
    
    Output:
    GridSearchCV object with optimal parameters.    
    '''
    #build pipeline
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=1)))
        ])
    
    #parameters for drid search to find optimal parameters
    parameters = {
        'clf__estimator__min_samples_split': [2,4],
        'clf__estimator__min_samples_leaf': [2,3],
        'vect__ngram_range': ((1, 1),(1, 2))
    }
    
    #Grid search for optimal parameters
    cv = GridSearchCV(pipeline, param_grid=parameters,verbose = 10)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate model on accuracy score,fbeta score.Generate classification report
    Print the classification report,mean fbeta score and mean accuracy score
    
    Args:
    model,X_test, Y_test, category_names
    
    Returns:
    None
    '''

    # evaluate all steps on test set
    pred = model.predict(X_test)
    
    #create a dataframe for our multioutput predictions
    predicted_df=pd.DataFrame(pred,columns=category_names)   
    
    #Classification report
    for col in category_names:
        print(style.BOLD + 'Feature : {}\n'.format(col) + style.END,classification_report(Y_test[col],predicted_df[col]), '\n')
    
    #Mean F-beta score for the features in Test data
    print(style.BOLD + 'F-beta score for the features in Test data\n' + style.END)
    f_score_test=[]
    for col in category_names:
        f_score_test.append(fbeta_score(Y_test[col],predicted_df[col],average='weighted',beta=1))
        print(style.BOLD + col + style.END,':', fbeta_score(Y_test[col],predicted_df[col],average='weighted',beta=1))    

    print(style.BOLD +'\nMean fbeta score : {}'.format(np.mean(f_score_test)) +style.END) 
    
    #Mean accuracy score for test data
    acc_score_test=[]
    for col in category_names:
        acc_score_test.append((predicted_df[col].values==Y_test[col].values).mean())
    
    print(style.BOLD +'Mean acc score : {}'.format(np.mean(acc_score_test)) +style.END)  

    
def save_model(model, model_filepath):
    '''
    Save the model with optimal parameters obtained from GridSearchCV
    
    Args:
    final optimized model and the model file path
    
    Returns:
    None
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


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