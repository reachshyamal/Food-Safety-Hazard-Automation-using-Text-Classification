#!/usr/bin/env python
# coding: utf-8

# # This notebooks does loading of the model and prediction on test set of the Output : "Are there any inherent or cross contact allergens or intolerants?- actual" in the FSHA form.The text is cleaned, normalized and then vectorized and passed into the model for predictions.

# In[515]:


#import all necessary modules
from __future__ import print_function
import logging
from optparse import OptionParser
import sys
from time import time
import os
import nltk
import string
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize,sent_tokenize,RegexpTokenizer
import re
from nltk.tokenize.toktok import ToktokTokenizer
from bs4 import BeautifulSoup
import unicodedata
import numpy as np
from numpy import array
from numpy import argmax
import random
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import warnings
import pickle
from sklearn.externals import joblib
from socket import socket
import warnings
warnings.simplefilter('ignore')


# In[489]:


#command line argument parsing
# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--path",
              action="store_true", dest="path",
              help="Input path name")
              
op.add_option("--filename",
              action="store_true", dest="filename",
              help="Input data file name")

op.add_option("--out_filename",
              action="store_true", dest="out_filename",
              help="Output data file name")

(opts, args) = op.parse_args()

if len(args) == 0:
    op.error("this script takes at least one argument (--filename).")
    sys.exit(1)
#display filename    
print(opts.filename)
print(args[0])
path = args[0]
filename = args[1]
out_filename = args[2]
print(__doc__)
op.print_help()
print()


# # Define reusable modular method for Text Normalization (removal of stopwords, changing to lower case, removal of punctuation etc) 

# In[490]:


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

# In[491]:


#filename = "C:\\Users\\574977\\Downloads\\PycharmProjects\\pepsico\\shymal\\Inputfiles\\FSHA_input.xlsx"
f1 = path + "\\" + filename
fsha_data = pd.read_excel(f1)


# # Based on Analysis select the Features (X)

# In[492]:


#selecting set of columns as Features
features_df=fsha_data[['allergens','allergensLabeledIMAF']]


# In[493]:


test_df = features_df


# # Normalize each column , by cleaning the text

# In[494]:


def conv_str(x):
   
    x=str(x)
    x=x.lower()
    return (x)

test_df['allergens']=test_df['allergens'].apply(lambda x:conv_str(x))  
test_df['allergensLabeledIMAF']=test_df['allergensLabeledIMAF'].apply(lambda x:conv_str(x))


# In[495]:


test_df['norm_allergens'] = normalize_corpus(test_df['allergens'])
test_df['norm_allergens_M']=normalize_corpus(test_df['allergensLabeledIMAF'])


# In[496]:


test_df = test_df.drop(['allergens','allergensLabeledIMAF'],axis = 1)


# # Perform n-gram vectorization and PCA on text data, columnwise, and concatenate with categorical one-hot encoded data

# In[497]:



def preprocess_text(test_df): 
    scaler = joblib.load(path +"\\"+ "scaler_allergens.pkl")
    vectorize = joblib.load(path +"\\"+ "vectorizer_allergens.pkl")
    pca = joblib.load(path +"\\"+ "pca_allergens.pkl")
    
    x_ngram_allergens = vectorize.transform(test_df['norm_allergens']).toarray()
    x_ngram_allergens_m = vectorize.transform(test_df['norm_allergens_M']).toarray()
    test_df = test_df.drop(['norm_allergens','norm_allergens_M'],axis=1)
    
   
    x_ngram_allergens = scaler.transform(x_ngram_allergens)
    x_ngram_allergens_m = scaler.transform(x_ngram_allergens_m)
    
    x_pca_allergens = pca.transform(x_ngram_allergens)
    x_pca_allergens_m = pca.transform(x_ngram_allergens_m)
    x_test = np.concatenate((test_df,x_pca_allergens,x_pca_allergens_m),axis=1)
    return x_test  


# In[498]:


n_gram_range = (1,2)
n_components = 6
whiten = False
random_state = 42
svd_solver="full"
X_features = preprocess_text(test_df)


# In[499]:


X_features = pd.DataFrame(X_features)


# In[500]:


X_features.reset_index(drop=True, inplace=True)


# # Loading the model for predictions on validation data to be provided

# In[501]:


filename = path +"\\"+ "finalized_model_CrossContact.sav"
loaded_model = joblib.load(filename)


# In[502]:


y_pred = loaded_model.predict(X_features)
y_pred1 = pd.DataFrame(y_pred,columns=['Predicted crosscontact_allergens'])


# In[503]:


test_df['File Name'] = fsha_data['File Name']


# In[504]:


y_test1 = fsha_data['crossContactAllergens']


# In[505]:


y_test1.index = y_pred1.index


# In[506]:


test_df['File Name'].index = y_pred1.index 


# In[507]:


test_df.columns


# In[508]:


output_df = pd.concat([test_df['File Name'],y_test1,y_pred1],axis=1)


# In[509]:


fsha_data = fsha_data.drop(['potentialMicrobial', 'crossContactAllergens', 'chokeHazard',
       'operationalAllergen', 'allergensLabeledIMAF','abuseByConsumer', 'cookstepByConsumer'],axis = 1)


# In[510]:


final_output = pd.merge(fsha_data,output_df, on='File Name', how='inner')


# In[511]:


output_ = pd.read_excel(path + "\\"+"ML_Output_CrossContact.xlsm")


# In[512]:


final_output.columns = output_.columns


# In[513]:


final_output['Are there any inherent or cross contact allergens or intolerants?- predicted'] = final_output['Are there any inherent or cross contact allergens or intolerants?- predicted'].apply(lambda x: 'Yes' if x == 1 else 'No' )


# In[514]:



final_output.to_csv(path +"\\"+ out_filename,index = False)

