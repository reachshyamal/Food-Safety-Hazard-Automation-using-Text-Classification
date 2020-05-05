#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import all neccesary modules
from __future__ import print_function
import string
import re
import logging
from optparse import OptionParser
import sys
from time import time
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
import pandas as pd
import string
import re
import nltk
import statistics 
from sklearn.model_selection import train_test_split 
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Dense
import pydot
import pydotplus
import graphviz
import pydot_ng
from keras import backend as K
from keras.preprocessing.text import Tokenizer
exclude = set(string.punctuation) 
wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')
table = str.maketrans('', '', string.punctuation)


# In[ ]:


#command line argument parsing
# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--filename",action="store_true", dest="filename",help="Input data file name")
op.add_option("--target_col",action="store_true", dest="target_col",help="Input target column name")
op.add_option("--mode",action="store_true", dest="mode",help="Input mode value")

(opts, args) = op.parse_args()

if len(args) == 0:
    op.error("this script takes at least one argument (--filename).")
    sys.exit(1)
#display filename    
print(opts.filename)
print(args[0])
#display target column name
print(opts.target_col)
print(args[1])
#display mode
print(opts.mode)
print(args[2])


# In[ ]:


# load doc into memory
def load_file(filename1):
    df = pd.read_excel(filename1)
    df.fillna('NA', inplace=True)
    return df


# In[ ]:


#cleaning and preprocessing
def clean_doc(doc):
	# split into tokens by white space
    doc = doc.replace('\n',' ')
    tokens = doc.split()
	# prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
	# remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
	# remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
    tokens = [w for w in tokens if not w in stop_words]
	# filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
	# remove nn from each word
    tokens = [re.sub('nn',' ',word) for word in tokens]
    return tokens


# In[ ]:


# load doc and add to vocab
def add_doc_to_vocab(filename1, vocab):
	# load doc
    df = load_file(filename1)
    #selecting set of columns as Features
    features_df=df[['preservatives', 'pH', 'waterActivity', 'packaging','otherFSA',
            'prodStorageDist', 'foodSafetyProdClaims','targetMarket','allergens','newIngredient']]
    # clean doc 
    for i in range (len(features_df.columns)):
        col = features_df.columns.get_values()[i]
        tokens = clean_doc(str(features_df[col].values))
        vocab.update(tokens)


# In[ ]:


# save list to file
def save_list(lines, filename1):
	# convert lines to a single blob of text
	data = '\n'.join(lines)
	# open file
	file = open(filename1, 'w')
	# write text
	file.write(data)
	# close file
	file.close()


# In[ ]:


# define vocab
vocab = Counter()
# create vocab
add_doc_to_vocab(str(args[0]),vocab)
# keep tokens with a min occurrence
min_occurance = 2
tokens = [k for k,c in vocab.items() if c >= min_occurance]
# save tokens to a vocabulary file
save_list(tokens, 'C:\\Users\\574977\\PycharmProjects\\pepsico\\anirban\\vocab.txt')


# In[ ]:


def process_data_to_lines(df, vocab):
    lines = list()
    for i in range(len(df)):
        tokens = clean_doc(str(df.iloc[i,:].values))
	# filter by vocab
        tokens = [w for w in tokens if w in vocab]
        tokens = ' '.join(tokens)
        lines.append(tokens)
    return lines


# In[ ]:


# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# In[ ]:


# load doc into memory
def load_doc(filename1):
	# open the file as read only
	file = open(filename1, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text


# In[ ]:


# load the vocabulary
vocab_filename = 'C:\\Users\\574977\\PycharmProjects\\pepsico\\anirban\\vocab.txt'
vocab = load_doc(vocab_filename)
vocab = set(vocab.split())    


# In[ ]:


#imputation for na values
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


# In[ ]:


# load all data
filename1 = str(args[0])
df = pd.read_excel(filename1)
df.fillna('NA', inplace=True)
X = df[['preservatives','pH','waterActivity','packaging','otherFSA','prodStorageDist', 'foodSafetyProdClaims','targetMarket','allergens','newIngredient']]
y = impute_target(df,str(args[1]))
y= pd.get_dummies(y)
        


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y, test_size=0.2)
train_docs = process_data_to_lines(X_train,vocab)
test_docs = process_data_to_lines(X_test,vocab)    


# In[ ]:


# create the tokenizer
tokenizer = create_tokenizer(train_docs)   


# In[ ]:


# encode data
Xtrain,ytrain = tokenizer.texts_to_matrix(train_docs, mode=str(args[2])),y_train
Xtest,ytest = tokenizer.texts_to_matrix(test_docs, mode=str(args[2])),y_test 


# In[ ]:


#define the model

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def define_model(n_words):
	# define network
	model = Sequential()
	model.add(Dense(50, input_shape=(n_words,), activation='relu'))
	model.add(Dense(2, activation='sigmoid'))
	# compile network
	model.compile(loss='binary_crossentropy', optimizer='adam',metrics=[f1_m])
	# summarize defined model
	model.summary()
	#plot_model(model, to_file='model.png', show_shapes=True)
	return model


# In[ ]:


n_words = Xtest.shape[1]
model = define_model(n_words)

# fit network
model.fit(Xtrain, ytrain, epochs=10, verbose=2)
# evaluate
loss,f1_score = model.evaluate(Xtest, ytest, verbose=1)
#loss,acc = model.evaluate(Xtest, ytest, verbose=1)
print('Test f1 score: %f' % (f1_score*100))

