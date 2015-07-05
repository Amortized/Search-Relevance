from nltk.stem.porter import *
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from nltk.corpus import stopwords
from sklearn import decomposition, pipeline, metrics, grid_search
from util import quadratic_weighted_kappa;
import warnings
import math;

def generateStopWords():
    stop_words      = set(text.ENGLISH_STOP_WORDS) | set(stopwords.words("english"));
    new_stop_words  = ['http','www','img','border','color','style','padding','table',\
                       'font','thi','inch','ha','width','height','0','1','2','3','4',\
                       '5','6','7','8','9'];

    return stop_words | set(new_stop_words);


def preprocess(data, type="train"):
    data_X = [];
    stemmer = PorterStemmer();

    for i in range(len(data.id)):
        #Strip out HTML, split it and lower case
        query = (" ").join([z.lower()  for z in BeautifulSoup(data["query"][i]).get_text(" ").split(" ")]);
        query = re.sub("[^a-zA-Z0-9]"," ", query);    

        #Strip out HTML, split it and lower case
        title = (" ").join([z.lower()  for z in BeautifulSoup(data["product_title"][i]).get_text(" ").split(" ")]);
        title = re.sub("[^a-zA-Z0-9]"," ", title);  

        #Concatenate query and title
        query_title = query + " " + title
        query_title = (" ").join([stemmer.stem(z) for z in query_title.split(" ")]);
        
        data_X.append(query_title);

    if type=="train":    
        return data_X, data['query'].values, data['product_title'].values;    
    else:
        return data_X, data.id.values.astype(int);


def model(train_X, train_Y, test_X, test_ids, train_query,train_title):

    stop_words = generateStopWords();

    tfv = TfidfVectorizer(min_df=3, strip_accents='unicode', analyzer='word',\
                          token_pattern=r'\w{1,}',ngram_range=(1, 5), use_idf=True,smooth_idf=True,\
                          sublinear_tf=True,stop_words = stop_words);


    #Vectorize the data
    tfv.fit(train_X);
    train_X = tfv.transform(train_X)
    test_X  = tfv.transform(test_X)

    #Pipeline set up 
    svd     = TruncatedSVD(); #Latent Semantic Analysis.
    scl     = StandardScaler();
    mymodel = SVC(); #Multi Classifier;

    # Create the pipeline 
    clf = pipeline.Pipeline([('svd', svd),
                             ('scl', scl),
                             ('svm', mymodel)]);

    #Best Parameters for the whole pipeline
    param_grid = {'svd__n_components' : [300,400] , 'svm__C' : [1,5,10,12]}

    kappa_scorer = metrics.make_scorer(quadratic_weighted_kappa, greater_is_better = True)

    # Initialize Grid Search Model
    model = grid_search.GridSearchCV(estimator = clf, param_grid=param_grid, scoring=kappa_scorer,\
                                     verbose=10, n_jobs=-1, iid=True, refit=True, cv=2);

    # Fit Grid Search Model
    model.fit(train_X, train_Y)
    print("Best score: %0.3f" % model.best_score_)
    print("Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    
    # Get best model
    best_model = model.best_estimator_
    
    # Fit model with best parameters optimized for quadratic_weighted_kappa
    best_model.fit(train_X,train_Y);


    #Debug
    f = open("mistakes.dat", "w")
    train_preds = best_model.predict(train_X);
    for i in range(0, len(train_preds)):
        if train_preds[i] != train_Y[i]:
            f.write("\n...................")
            f.write("\nQuery "      + str(train_query[i]));
            f.write("\nTitle "      + str(train_title[i]));
            f.write("\nPredicted  " + str(train_preds[i]));
            f.write("\nActual "     + str(train_Y[i]));
            f.write("\n...................")
    f.close();
    

    preds = best_model.predict(test_X)
    
    # Create your first submission file
    submission = pd.DataFrame({"id": test_ids, "prediction": preds})
    submission.to_csv("submission.csv", index=False)


def do(train_loc, test_loc):
    #load data
    train                            = pd.read_csv(train_loc).fillna("");
    test                             = pd.read_csv(test_loc).fillna("");

    
    train_X,train_query,train_title  = preprocess(train, "train");
    train_Y                          = train.median_relevance.values
    test_X,test_ids                  = preprocess(test, "test");

    model(train_X, train_Y, test_X, test_ids, train_query,train_title);



if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    do("./data/train.csv", "./data/test.csv");

