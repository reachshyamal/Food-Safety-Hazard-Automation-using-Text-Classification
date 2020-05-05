#!/usr/bin/env python
# coding: utf-8

# # This notebook reads text data from data extract created from FSHA Forms and runs predictive Models to predict the value 'Are there any inherent or cross contact allergens or intolerants?' , based on the Input Data

# In[1]:


#import all necessary modules
from __future__ import print_function
import logging
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt
import os
import nltk
import string
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.tokenize import RegexpTokenizer
import re
import numpy as np
from numpy import array
from numpy import argmax
from scipy import signal
import random
import string
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import SVC 
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.utils.extmath import density
from sklearn.decomposition import PCA
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.keras import models
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
import warnings
warnings.simplefilter('ignore')


# ## File name and other important parameters like ngram_range set

# In[ ]:


#command line argument parsing
# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--filename",
              action="store_true", dest="filename",
              help="Input data file name")
op.add_option("--target_call",
              action="store_true", dest="target_call",
              help="Input target variable name")

(opts, args) = op.parse_args()

if len(args) == 0:
    op.error("this script takes at least one argument (--filename).")
    sys.exit(1)
#display filename    
print(opts.filename)
print(args[0])
#display target variable name
print(opts.target_call)
print(args[1])
print(__doc__)
op.print_help()
print()


# In[2]:


#These parameters will be input from command line
ngram_range_inp=(1,2)
#filename = "C:\\Users\\725191\\Desktop\\PepsiCo FSHA data\\RPA data\\FSHA - RPA.xlsm"
filename = str(args[0])
#n_components = 0


# # Define reusable modular method for Text Normalization (removal of stopwords, changing to lower case, removal of punctuation etc) - Q1

# In[3]:


exclude = set(string.punctuation) 
wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')
newStopWords = ['from','dtype','object']
stop_words.extend(newStopWords)
table = str.maketrans('', '', string.punctuation)

porter = PorterStemmer()

def normalize_document(doc):
    # tokenize document
    tokens = doc.split()
    # remove punctuation from each word
    tokens = [w.translate(table) for w in tokens]
    # convert to lower case
    lower_tokens = [w.lower() for w in tokens]
    #remove spaces
    stripped = [w.strip() for w in lower_tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter stopwords out of document
    filtered_tokens = [token for token in words if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

#normalize_corpus = np.vectorize(normalize_document)


# In[11]:


#import spacy
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import re
from bs4 import BeautifulSoup
import unicodedata

#nlp = spacy.load('en', parse = False, tag=False, entity=False)
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')
def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text
def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def remove_special_characters(text):
    text = re.sub('[^a-zA-z0-9\s]', '', text)
    text = text.replace('\n','')
    return text

def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text
def normalize_corpus(corpus, html_stripping=True, contraction_expansion=True,
                     accented_char_removal=True, text_lower_case=True, 
                     text_lemmatization=True, special_char_removal=True, 
                     stopword_removal=True):
    
    normalized_corpus = []
    # normalize each document in the corpus
    for doc in corpus:
        # strip HTML
        if html_stripping:
            doc = strip_html_tags(doc)
        # remove accented characters
        if accented_char_removal:
            doc = remove_accented_chars(doc)

        # lowercase the text    
        if text_lower_case:
            doc = doc.lower()
        # remove extra newlines
        doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc)
        # insert spaces between special characters to isolate them    
        special_char_pattern = re.compile(r'([{.(-)!}])')
        doc = special_char_pattern.sub(" \\1 ", doc)
        # remove special characters    
        if special_char_removal:
            doc = remove_special_characters(doc)  
        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)
        # remove stopwords
        if stopword_removal:
            doc = remove_stopwords(doc, is_lower_case=text_lower_case)
            
        normalized_corpus.append(doc)
        
    return normalized_corpus


# # Read the data extract file (tabular format with Input data(X) and target(Y))

# In[5]:


fsha_data = pd.read_excel(filename)


# In[6]:


fsha_data.columns


# # Based on Analysis select the Features (X)

# In[12]:


#selecting set of columns as Features
data2=fsha_data[['preservatives', 'pH', 'waterActivity', 'packaging','otherFSA',
            'prodStorageDist', 'foodSafetyProdClaims','targetMarket','allergens','newIngredient']]


# In[13]:


data2


# # Replace missing values with NA

# In[14]:


data2.fillna('NA', inplace=True)


# In[15]:


data2


# In[16]:


train_df = data2


# # Define reusable code to Vectorize Text column (ex: Allergens) using TF-IDF Vectorizer, after doing Text data normalization

# In[17]:


# Vectorization of text data using TF-IDF Vectorizer

# Range (inclusive) of n-gram sizes for tokenizing text.
#NGRAM_RANGE 

# Limit on the number of features. We use the top 20K features.
#TOP_K = 20000

# Whether text should be split into word or character n-grams.
# One of 'word', 'char'.
TOKEN_MODE = 'word'

# Minimum document/corpus frequency below which a token will be discarded.
MIN_DOCUMENT_FREQUENCY = 2

# Limit on the length of text sequences. Sequences longer than this
# will be truncated.
MAX_SEQUENCE_LENGTH = 500


def ngram_vectorize(train_texts, train_labels,ngram_range):
    """Vectorizes texts as ngram vectors.

    1 text = 1 tf-idf vector the length of vocabulary of uni-grams + bi-grams.

    # Arguments
        train_texts: list, training text strings.
        train_labels: np.ndarray, training labels.
        val_texts: list, validation text strings.

    # Returns
        x_train, x_val: vectorized training and validation texts
    """
    # Create keyword arguments to pass to the 'tf-idf' vectorizer.
    kwargs = {
            'ngram_range': ngram_range,  # Use 1-grams + 2-grams.
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': TOKEN_MODE,  # Split text into word tokens.
            'min_df': MIN_DOCUMENT_FREQUENCY,
    }
    vectorizer = TfidfVectorizer(**kwargs)

    # Learn vocabulary from training texts and vectorize training texts.
    x_train = vectorizer.fit_transform(train_texts)

    # Select top 'k' of the vectorized features.
    #selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
    #selector.fit(x_train, train_labels)
    #x_train = selector.transform(x_train)

    x_train = x_train.astype('float32')
    return x_train


# # Define method to concatenate normalized Text data

# In[397]:


train_df.columns


# In[18]:


train_df['norm_preservatives'] = normalize_corpus(train_df['preservatives'])
train_df['norm_prodStorage'] = normalize_corpus(train_df['prodStorageDist'])
#train_df['norm_waterActivity'] = normalize_corpus(str(train_df['waterActivity']))
train_df['norm_packaging'] = normalize_corpus(train_df['packaging'])
train_df['norm_otherFSA'] = normalize_corpus(train_df['otherFSA'])
train_df['norm_prodStorageDist'] = normalize_corpus(train_df['prodStorageDist'])
train_df['norm_foodSafety_prodClaims'] = normalize_corpus(train_df['foodSafetyProdClaims'])
train_df['norm_targetMarket'] = normalize_corpus(train_df['targetMarket'])
train_df['norm_newIngredient'] = normalize_corpus(train_df['newIngredient'])
train_df['norm_allergens'] = normalize_corpus(train_df['allergens'])
train_df['all_cols'] = train_df['norm_preservatives']+" "+train_df['norm_prodStorage']+" "+train_df['norm_packaging']+" "+train_df['norm_otherFSA']+" "+train_df['norm_prodStorageDist']+" "+train_df['norm_foodSafety_prodClaims']+" "+train_df['norm_targetMarket']+" "+train_df['norm_newIngredient']+" "+train_df['norm_allergens']
   
#train_df['norm_waterActivity']+" "+


# In[20]:


train_df['all_cols'][2]


# # Imput target with mode value

# In[29]:


import statistics 

def impute_target(fsha_data,targetName):
    train_y=[]
    for i in range (len(fsha_data)):
        if str(fsha_data[targetName][i]).strip().lower() =='yes':
            train_y.append(1)
        elif str(fsha_data[targetName][i]).strip().lower() =='no':
            train_y.append(0)
        else:
            train_y.append(-1)
               
    mode_y = statistics.mode(train_y)

    for i in range (len(fsha_data)):
        if train_y[i]==-1:
            train_y[i] = mode_y
            
    return train_y


# # Four sets of targets 

# In[64]:


target_names = ['potentialMicrobial','chokeHazard','operationalAllergen']

"""
train_y = pd.DataFrame()
for i in range(0,len(target_names)):
    train_y[str(target_names[i])]=impute_target(fsha_data,str(target_names[i]))

"""
train_y =[]
train_y= impute_target(fsha_data,str(args[1]))
#train_y_allergens = impute_target(fsha_data,"crossContactAllergens") - this is having all 1s
#train_y_chokeHazard = impute_target(fsha_data,"chokeHazard")
#train_y_opAllergens = impute_target(fsha_data,"operationalAllergen")


# In[89]:


train_df['Target']=train_y


# In[96]:


#train_df['Target']=train_y_chokeHazard


# In[102]:


#train_df['Target']=train_y_opAllergens


# # Define method to evaluate Machine Learning models with the X and y vectors created above, and check the effectiveness of each. Also store the results in array to be plotted in graph for visualization

# In[50]:


from datetime import datetime
import time
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(len(X_train))
    
    clf_descr = str(clf).split('(')[0]
    print("model name:"+clf_descr)
  
    a = datetime.now()
    
    if clf_descr.__contains__('tensorflow'):
        history = clf.fit(
            X_train,
            y_train,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=(X_test, y_test),
            verbose=2, 
            batch_size=batch_size)
    else:
        clf.fit(X_train, y_train)
    
    b = datetime.now()
    c = a-b
    train_time = c.microseconds
    print("train time: %0.3fs" % train_time)

    pred = clf.predict(X_test)

    pred_train = clf.predict(X_train)
 
    if clf_descr.__contains__('tensorflow'):
        for i in range (len(pred)):
            if (pred[i]>=0.3):
                pred[i]=1
            else:
                pred[i]=0
        for i in range (len(pred_train)):        
            if (pred_train[i]>=0.3):
                pred_train[i]=1
            else:
                pred_train[i]=0
    
    f1_score = metrics.f1_score(y_test, pred)
    print("f1_score:   %0.3f" % f1_score )
    
    f1_score_train = metrics.f1_score(y_train, pred_train)
    print("f1_score_train:   %0.3f" % f1_score_train )
    
    print("classification report:")
    print(classification_report(y_test, pred))
    
    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred))

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

    
    return clf_descr,f1_score_train,f1_score,train_time


# # Perform n-gram vectorization,train test split and apply PCA
# Evaluate Machine Learning Models from a List (using the reusable method defined above). Store the results (Accuracy score - train, accuracy score - test, and training time) in a List for data visualization¶

# In[103]:


#results = pd.DataFrame()
#model_name = pd.DataFrame()
#for i in range(0,len(list(train_y))):
train_labels = train_y
    #Vectorize the text data, n_gram range = (1,1)
x_ngram = ngram_vectorize(train_df['all_cols'], train_labels,(1,1))
x_ngram=x_ngram.toarray()
print(x_ngram.shape)   

    #Standard scaling on vectorized data
#x_ngram_std = StandardScaler().fit_transform(x_ngram)  
X,y = x_ngram,train_df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

"""
#Apply PCA 
whiten = False
random_state = 42
svd_solver="full"
n_comp = 2

pca = PCA(n_components=n_comp,svd_solver=svd_solver,whiten=whiten, random_state=42)
pca.fit(X_train)
x_train_pca = pca.transform(X_train)

x_test_pca = pca.transform(X_test)
    
print(x_train_pca.shape)
print(x_test_pca.shape)
"""  

selector = SelectKBest(f_classif, k='all')
results = []
model_name = []

for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (Perceptron(n_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=5), "kNN"),
        (RandomForestClassifier(n_estimators=100), "Random forest")):
    selector_clf = benchmark(Pipeline([('selector', selector),('classifier', clf)]))
    print('=' * 80)
    print(name)
    results.append(benchmark(clf))
    model_name.append(name)
    #model.append(clf)
    
for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(loss='l2', penalty=penalty,dual=False, tol=1e-3)))
    model_name.append("LinearSVC"+" "+penalty)

    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,penalty=penalty)))
    model_name.append("SGDClassifier"+" "+penalty)


# Train SGD with Elastic Net penalty
    print('=' * 80)
    print("Elastic-Net penalty")
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,penalty="elasticnet")))
    model_name.append("SGD with Elastic Net penalty")


# Train NearestCentroid without threshold
    print('=' * 80)
    print("NearestCentroid (aka Rocchio classifier)")
    results.append(benchmark(NearestCentroid()))
    model_name.append("NearestCentroid (aka Rocchio classifier)")

    # Train sparse Naive Bayes classifiers
    print('=' * 80)
    print("Naive Bayes")
    results.append(benchmark(MultinomialNB(alpha=.01)))
    model_name.append("MultinomialNB")



    results.append(benchmark(BernoulliNB(alpha=.01)))
    model_name.append("BernoulliNB")



# # Define Neural Network model

# In[104]:


def mlp_model(layers, units, dropout_rate, input_shape):
    """Creates an instance of a multi-layer perceptron model.

    # Arguments
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of the layers.
        dropout_rate: float, percentage of input to drop at Dropout layers.
        input_shape: tuple, shape of input to the model.

    # Returns
        An MLP model instance.
    """
    model = models.Sequential()
    model.add(Dropout(rate=dropout_rate, input_shape=input_shape))

    for _ in range(layers-1):
        model.add(Dense(units=units, activation='relu'))
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=1, activation='sigmoid'))
    return model


# # Compile Model

# In[105]:


learning_rate=1e-3
epochs=100
batch_size=128
layers=2
units=64
dropout_rate=0.2
model = mlp_model(layers=layers,units=units,dropout_rate=dropout_rate,input_shape=X.shape[1:])
optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]
model.summary()


# # Train model and store results in array

# In[106]:


print('=' * 80)
print("Keras Dense Neural Network")
results.append(benchmark(model))
model_name.append("Keras Dense Neural Network")


# In[ ]:


"""
import pickle
dir1 = 'C:\\Users\\725191\\Desktop\\PepsiCo FSHA data\\Data extraction format\\Results_08July\\'+str(args[1])+.pkl
results_fileName='C:\\Users\\725191\\Desktop\\PepsiCo FSHA data\\Data extraction format\\Results_08July\\results_potentialmicrobial.pkl'
with open(results_fileName, "wb") as f:
    w = pickle.dump(results,f)
"""


# In[ ]:


"""
import pickle
results_fileName='C:\\Users\\725191\\Desktop\\PepsiCo FSHA data\\Data extraction format\\Results_08July\\results_chokehazard.pkl'
with open(results_fileName, "wb") as f:
    w = pickle.dump(results,f)
"""


# In[107]:



"""
import pickle
results_fileName='C:\\Users\\725191\\Desktop\\PepsiCo FSHA data\\Data extraction format\\Results_08July\\results_opAllergens.pkl'
with open(results_fileName, "wb") as f:
    w = pickle.dump(results,f)
"""


# In[ ]:


"""
import scipy
df_f1scoretrain = pd.DataFrame({"potentialmicrobial":f1_score_train1,"chokehazard":f1_score_train2,"opAllergens":f1_score_train3})
maxval =pd.DataFrame(df_f1scoretrain.apply(max, axis=0))
print("Maximum f1score per target is:"+maxval)
f1harmonic = maxval.apply(scipy.stats.hmean, axis=0)
f1harmonic
print("Harmonic mean of f1-score of all 3 targets is:",f1harmonic)
"""


# In[354]:


#results_microbial=pd.read_pickle('D:/Pepsico/results_microbial_100.pkl')
#results_chokeHazard=pd.read_pickle('D:/Pepsico/results_chokeHazard_100.pkl')
#results_opAllergen=pd.read_pickle('D:/Pepsico/results_opAllergen_100.pkl')


# # Visualize the results from multiple Machine Learning Models (accuracy - train, accuracy - test, training time)

# In[116]:


# make some plots
indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, f1_score_train, f1_score_test, training_time = results

training_time = np.array(training_time) / np.max(training_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, f1_score_train, .2, label="f1_score_train", color='r')
plt.barh(indices+0.3, f1_score_test, .2, label="f1_score_test", color='b')
plt.barh(indices + .6, training_time, .2, label="training time", color='g')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, model_name):
    plt.text(-.3, i, c)

plt.show()


# In[ ]:


index=np.argmax(f1_score_test)
f1scorebest = f1_score_test[index]
print(f1scorebest)
print(model_name[index])
print("Best classifier is:"+model_name[index],"||f1_score",f1scorebest)


# In[ ]:




