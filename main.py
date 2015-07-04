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

def generateStopWords():
    stop_words      = set(text.ENGLISH_STOP_WORDS) | set(stopwords.words("english"));
    new_stop_words  = ['http','www','img','border','color','style','padding','table',\
                       'font','thi','inch','ha','width','height','0','1','2','3','4',\
                       '5','6','7','8','9'];

    return stop_words | set(new_stop_words);


def preprocess(data):
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

    return data_X;    






def do(train_loc, test_loc):
    #load data
    train      = pd.read_csv(train_loc).fillna("");
    test       = pd.read_csv(test_loc).fillna("");

    stop_words = generateStopWords();

    train_X    = preprocess(train);
    train_Y    = train.median_relevance.values
    test_X     = preprocess(test);





do("./data/train.csv", "./data/test.csv");

