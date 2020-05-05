#!/usr/bin/env python
# coding: utf-8

# In[9]:


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
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.tokenize import RegexpTokenizer
import re
import numpy as np
import string
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import SVC 
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
from sklearn import metrics


# In[10]:


#os.chdir('C:\\Users\\574977\\PycharmProjects\\pepsico\\fsha\\turgay')


# In[11]:


#command line argument parsing
# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--filename",
              action="store_true", dest="filename",
              help="Input data file name")
op.add_option("--range_ngram_low",
              action="store_true", dest="range_ngram_low",
              help="Input lower ngram range value")
op.add_option("--range_ngram_upper",
              action="store_true", dest="range_ngram_upper",
              help="Input upper ngram range value")

(opts, args) = op.parse_args()

if len(args) == 0:
    op.error("this script takes at least one argument (--filename).")
    sys.exit(1)
#display filename    
print(opts.filename)
print(args[0])
#display lower limit of ngram
filename = args[0]
print(opts.range_ngram_low)
print(args[1])
#display upper limit of ngram
print(opts.range_ngram_upper)
print(args[2])
#displays essential functions, features etc
print(__doc__)
op.print_help()
print()


# In[ ]:


#read filename from command line argument
fsha_data = pd.read_excel(filename)


# In[ ]:


#text cleaning
exclude = set(string.punctuation) 
wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')
#newStopWords = ['lb','j','df','jdf']
#stop_words.extend(newStopWords)
table = str.maketrans('', '', string.punctuation)
from nltk.stem.porter import PorterStemmer
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
    #apply Stemming
    #stemmed = [porter.stem(word) for word in filtered_tokens]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

normalize_corpus = np.vectorize(normalize_document)


# In[ ]:


#taking model dataframe
dataframe_model = fsha_data[["potentialMicrobial"," projDesc"," center"," cur_tsgStage","WHTD"," CPD-prodNameDesc"," procPlat"]]
train_df = dataframe_model


# In[ ]:


#Processing the categorical variables
lists_=[" center"," cur_tsgStage","WHTD"," CPD-prodNameDesc"," procPlat"]

for k in range(len(lists_)):
    if k == 0:
        independent_undersample = pd.get_dummies(train_df[lists_[k]].astype('category'))
        independent_undersample.index = train_df.index
        independent_undersample.columns = [lists_[k]+"_bucket_"+str(each) for each in range(independent_undersample.shape[1])]
    else:
        cat = pd.get_dummies(train_df[lists_[k]].astype('category'))
        cat.index = train_df.index
        cat.columns = [lists_[k]+"_bucket_"+str(each) for each in cat.columns]
        independent_undersample = pd.concat([independent_undersample,cat],axis=1)


# In[ ]:


#concatenation of target variable and text data
to_be_concatenated = train_df[["potentialMicrobial"," projDesc"]]
train_df = pd.concat([to_be_concatenated,independent_undersample],axis=1)


# In[16]:


# splitting the training dataset 
xdf_train, xdf_test = train_test_split(train_df, test_size=0.3,  random_state=1) 


# In[ ]:


#train test target variable division
train_labels_train = xdf_train["potentialMicrobial"]
train_labels_test = xdf_test["potentialMicrobial"]


# In[ ]:


#normalizing the text data column
clean_text_train = xdf_train[' projDesc'].astype(str)
clean_text_test = xdf_test[' projDesc'].astype(str)
norm_clean_text_train = normalize_corpus(clean_text_train)
norm_clean_text_test = normalize_corpus(clean_text_test)


# In[ ]:


# Vectorization parameters

# Range (inclusive) of n-gram sizes for tokenizing text.
NGRAM_RANGE = (int(args[1]), int(args[2]))

# Limit on the number of features. We use the top 20K features.
TOP_K = 20000

# Whether text should be split into word or character n-grams.
# One of 'word', 'char'.
TOKEN_MODE = 'word'

# Minimum document/corpus frequency below which a token will be discarded.
MIN_DOCUMENT_FREQUENCY = 2

# Limit on the length of text sequences. Sequences longer than this
# will be truncated.
MAX_SEQUENCE_LENGTH = 500


def ngram_vectorize(train_texts, train_labels, val_texts):
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
            'ngram_range': NGRAM_RANGE,  # Use 1-grams + 2-grams.
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': TOKEN_MODE,  # Split text into word tokens.
            'min_df': MIN_DOCUMENT_FREQUENCY,
    }
    vectorizer = TfidfVectorizer(**kwargs)

    # Learn vocabulary from training texts and vectorize training texts.
    x_train = vectorizer.fit_transform(train_texts)

    # Vectorize validation texts.
    x_val = vectorizer.transform(val_texts)

    # Select top 'k' of the vectorized features.
    selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
    selector.fit(x_train, train_labels)
    x_train = selector.transform(x_train)
    x_val = selector.transform(x_val)

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    return x_train, x_val


# In[ ]:


# Vectorize texts.
x_train, x_test = ngram_vectorize(norm_clean_text_train, train_labels_train, norm_clean_text_test)


# In[ ]:


# Create the count vector frequency feature matrix
countVec = CountVectorizer()
#normalize_corpus
feature_matrix = countVec.fit_transform(normalize_corpus(train_df[' projDesc']))
# Show tf-idf feature matrix
feature_matrix.toarray()
countVec.get_feature_names()
#print(len(countVec.get_feature_names()))


# In[ ]:


#Display Freq plot of n-grams
#%matplotlib inline
num_ngrams=int(args[2])
all_ngrams = list(countVec.get_feature_names())
#print(all_ngrams)
num_ngrams = min(num_ngrams, len(all_ngrams))
all_counts = feature_matrix.toarray().sum(axis=0).tolist()
#print(all_counts)
data_tuples = list(zip(all_ngrams,all_counts))
df_cv = pd.DataFrame(data_tuples)
df_cv = df_cv.rename(columns={0: 'Ngram'})
df_cv = df_cv.rename(columns={1: 'Ngram Freq'})
df_cv_sorted = df_cv.sort_values(by='Ngram Freq',ascending=False)
#.plot(kind='bar')
#df_cv_sorted['Ngram Freq']
df_cv_sorted = df_cv_sorted[df_cv_sorted['Ngram Freq'] >= 5]
#df_cv_sorted
x = range(len(df_cv_sorted))
x_labels = df_cv_sorted['Ngram']
axarr = df_cv_sorted['Ngram Freq'].sort_values(ascending=False).plot(kind='bar',figsize=(10,5))
axarr.set_title("N gram freq",fontsize=14,fontweight='bold')
axarr.set_xlabel("Ngrams",fontsize=12)
axarr.set_ylabel("Frequency",fontsize=12)
axarr.tick_params(axis='both',which='major',labelsize=12)
axarr.tick_params(axis='both',which='minor',labelsize=12)
axarr.set_xticklabels(x_labels, rotation='45')


# In[15]:


#train test data load
#data_train=pd.read_csv("C:\\Users\\574977\\PycharmProjects\\pepsico\\fsha\\turgay\\traindata.csv")
#data_test=pd.read_csv("C:\\Users\\574977\\PycharmProjects\\pepsico\\fsha\\turgay\\testdata.csv")
data_train = xdf_train.drop(' projDesc',1)
data_test = xdf_test.drop(' projDesc',1)


# In[13]:


#train test splitting
y_train = data_train["potentialMicrobial"]
y_test = data_test["potentialMicrobial"]
X_train = data_train.drop('potentialMicrobial',1)
X_test = data_test.drop('potentialMicrobial',1)
#X_train = X_train.iloc[:,2:35]
#X_test = X_test.iloc[:,2:35]


# In[14]:


# Benchmark classifiers
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)
    
    print("classification report:")
    print(classification_report(y_test, pred))
    
    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred))

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

    
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (Perceptron(n_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=5), "kNN"),
        (tree.DecisionTreeClassifier(),"Decision Tree" ),
        (RandomForestClassifier(n_estimators=100), "Random forest"),
        (AdaBoostClassifier(n_estimators=100),"Ada Boost")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf))

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(loss='l2', penalty=penalty,
                                            dual=False, tol=1e-3)))

    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                           penalty=penalty)))

# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                       penalty="elasticnet")))

# Train NearestCentroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
results.append(benchmark(NearestCentroid()))

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01)))
results.append(benchmark(BernoulliNB(alpha=.01)))
"""
print('=' * 80)
print("LinearSVC with L1-based feature selection")
# The smaller C, the stronger the regularization.
# The more regularization, the more sparsity.

results.append(benchmark(Pipeline([
  ('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),
  ('classification', LinearSVC())
])))
"""           

# make some plots

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='r')
plt.barh(indices + .3, training_time, .2, label="training time", color='g')
plt.barh(indices + .6, test_time, .2, label="test time", color='b')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()

##Putting maximum accuracy model in a list##

