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


ngram_range_inp=(1,2)


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


# In[87]:


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
    #corpus = str(corpus)
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


# In[88]:


def clean_doc(corpus):
    normalized_corpus = []
    # normalize each document in the corpus
    #corpus = str(corpus)
    for doc in corpus:
	# split into tokens by white space
        doc=str(doc)
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
        tokens = ' '.join(tokens)
        normalized_corpus.append(tokens)
    return normalized_corpus

# # Read the extract file, check the number of records, first few rows

# In[15]:


#filename = "C:\\Users\\574977\\Downloads\\PycharmProjects\\pepsico\\shymal\\Inputfiles\\FSHA_input.xlsx"
f1 = path + "\\" + filename
fsha_data = pd.read_excel(f1)


# # Based on Analysis select the Features (X)

# In[100]:


#selecting set of columns as Features
fsha_data.fillna('NA', inplace=True)
features_df=fsha_data[['PDA_projName', 'projDesc','formulaNumber','CPD-ProdName-Desc','packMaterial','prodStorageDist','shelfLife',
    'TCG','cookedOrHeated', 'specificStorage','labelingInstructions','mishandled','targetMarket','approvedPackage','File Name']]


# In[101]:


def conv_str(x):
   
    x=str(x)
    x=x.lower()
    return (x)

features_df['PDA_projName']=features_df['PDA_projName'].apply(lambda x:conv_str(x))  
features_df['projDesc']=features_df['projDesc'].apply(lambda x:conv_str(x))
features_df['formulaNumber']=features_df['formulaNumber'].apply(lambda x:conv_str(x)) 
features_df['packMaterial']=features_df['packMaterial'].apply(lambda x:conv_str(x))  
features_df['CPD-ProdName-Desc']=features_df['CPD-ProdName-Desc'].apply(lambda x:conv_str(x))  
features_df['prodStorageDist']=features_df['prodStorageDist'].apply(lambda x:conv_str(x)) 
features_df['shelfLife']=features_df['shelfLife'].apply(lambda x:conv_str(x))  
features_df['TCG']=features_df['TCG'].apply(lambda x:conv_str(x)) 
features_df['cookedOrHeated']=features_df['cookedOrHeated'].apply(lambda x:conv_str(x)) 
features_df['specificStorage']=features_df['specificStorage'].apply(lambda x:conv_str(x)) 
features_df['labelingInstructions']=features_df['labelingInstructions'].apply(lambda x:conv_str(x)) 
features_df['mishandled']=features_df['mishandled'].apply(lambda x:conv_str(x))  
features_df['targetMarket']=features_df['targetMarket'].apply(lambda x:conv_str(x)) 
features_df['approvedPackage']=features_df['approvedPackage'].apply(lambda x:conv_str(x)) 


# In[102]:


test_df = features_df


test_df['PDA_projName']=normalize_corpus(test_df['PDA_projName'])
test_df['projDesc']=normalize_corpus(test_df['projDesc'])
test_df['formulaNumber']=normalize_corpus(test_df['formulaNumber'])
test_df['packMaterial']=normalize_corpus(test_df['packMaterial'])  
test_df['CPD-ProdName-Desc']=normalize_corpus(test_df['CPD-ProdName-Desc'])
test_df['prodStorageDist']=normalize_corpus(test_df['prodStorageDist'])
test_df['shelfLife']=normalize_corpus(test_df['shelfLife'])
test_df['TCG']=normalize_corpus(test_df['TCG'])
test_df['cookedOrHeated']=normalize_corpus(test_df['cookedOrHeated'])
test_df['specificStorage']=normalize_corpus(test_df['specificStorage'])
test_df['labelingInstructions']=normalize_corpus(test_df['labelingInstructions']) 
test_df['mishandled']=normalize_corpus(test_df['mishandled']) 
test_df['targetMarket']=normalize_corpus(test_df['targetMarket'])
test_df['approvedPackage']=normalize_corpus(test_df['approvedPackage'])





def preprocess_text(test_df): 
    scaler_load = joblib.load(path +"\\"+"scaler_oppabuse.pkl")
    pca_load = joblib.load(path +"\\"+"pca_oppabuse.pkl")
    vectorizer_load = joblib.load(path +"\\"+"vectorizer_oppabuse.pkl")
    
    x_ngram_projName=vectorizer_load.transform(test_df['PDA_projName']).toarray()
    x_ngram_projDesc=vectorizer_load.transform(test_df['projDesc']).toarray()
    x_ngram_formula=vectorizer_load.transform(test_df['formulaNumber']).toarray()
    x_ngram_packMaterial=vectorizer_load.transform(test_df['packMaterial']).toarray() 
    x_ngram_CPD_ProdName_Desc=vectorizer_load.transform(test_df['CPD-ProdName-Desc']).toarray()
    x_ngram_prodStorageDist=vectorizer_load.transform(test_df['prodStorageDist']).toarray()
    x_ngram_shelfLife=vectorizer_load.transform(test_df['shelfLife']).toarray()
    x_ngram_TCG=vectorizer_load.transform(test_df['TCG']).toarray() 
    x_ngram_cookedOrHeated=vectorizer_load.transform(test_df['cookedOrHeated']).toarray() 
    x_ngram_specificStorage=vectorizer_load.transform(test_df['specificStorage']).toarray() 
    x_ngram_labelingInstructions=vectorizer_load.transform(test_df['labelingInstructions']).toarray()
    x_ngram_mishandled=vectorizer_load.transform(test_df['mishandled']).toarray() 
    x_ngram_targetMarket=vectorizer_load.transform(test_df['targetMarket']).toarray()
    x_ngram_approvedPackage=vectorizer_load.transform(test_df['approvedPackage']).toarray()
    test_df = test_df.drop(['PDA_projName','projDesc','formulaNumber','packMaterial','CPD-ProdName-Desc','prodStorageDist','shelfLife','TCG','cookedOrHeated','specificStorage','labelingInstructions','mishandled','targetMarket','approvedPackage'],axis=1)
    
    
    
    x_ngram_projName =  scaler_load.transform(x_ngram_projName)
    x_ngram_projDesc =  scaler_load.transform(x_ngram_projDesc)
    x_ngram_formula =  scaler_load.transform(x_ngram_formula)
    x_ngram_packMaterial =  scaler_load.transform(x_ngram_packMaterial)
    x_ngram_CPD_ProdName_Desc =  scaler_load.transform(x_ngram_CPD_ProdName_Desc)
    x_ngram_prodStorageDist =  scaler_load.transform(x_ngram_prodStorageDist)
    x_ngram_shelfLife =  scaler_load.transform(x_ngram_shelfLife)
    x_ngram_TCG =  scaler_load.transform(x_ngram_TCG)
    x_ngram_cookedOrHeated =  scaler_load.transform(x_ngram_cookedOrHeated)
    x_ngram_specificStorage =  scaler_load.transform(x_ngram_specificStorage)
    x_ngram_labelingInstructions =  scaler_load.transform(x_ngram_labelingInstructions)
    x_ngram_mishandled =  scaler_load.transform(x_ngram_mishandled)
    x_ngram_targetMarket =  scaler_load.transform(x_ngram_targetMarket)
    x_ngram_approvedPackage =  scaler_load.transform(x_ngram_approvedPackage)
    
    x_pca_projName = pca_load.transform(x_ngram_projName)
    x_pca_projDesc = pca_load.transform(x_ngram_projDesc)
    x_pca_formula = pca_load.transform(x_ngram_formula)
    x_pca_packMaterial = pca_load.transform(x_ngram_packMaterial)
    x_pca_CPD_ProdName_Desc = pca_load.transform(x_ngram_CPD_ProdName_Desc)
    x_pca_prodStorageDist = pca_load.transform(x_ngram_prodStorageDist)
    x_pca_shelfLife = pca_load.transform(x_ngram_shelfLife)
    x_pca_TCG = pca_load.transform(x_ngram_TCG)
    x_pca_cookedOrHeated = pca_load.transform(x_ngram_cookedOrHeated)
    x_pca_specificStorage = pca_load.transform(x_ngram_specificStorage)
    x_pca_labelingInstructions = pca_load.transform(x_ngram_labelingInstructions)
    x_pca_mishandled = pca_load.transform(x_ngram_mishandled)
    x_pca_targetMarket = pca_load.transform(x_ngram_targetMarket)
    x_pca_approvedPackage = pca_load.transform(x_ngram_approvedPackage)

    x_train = np.concatenate((x_pca_projName,x_pca_projDesc,x_pca_formula,x_pca_packMaterial,x_pca_CPD_ProdName_Desc,
                              x_pca_prodStorageDist,x_pca_shelfLife,x_pca_TCG,x_pca_cookedOrHeated,x_pca_specificStorage,x_pca_labelingInstructions,x_pca_mishandled,x_pca_targetMarket,x_pca_approvedPackage),axis=1)
    #print(x_train.shape)
    return x_train    


X_features = preprocess_text(test_df)

X_features = pd.DataFrame(X_features)





import pickle
from sklearn.externals import joblib
filename = path +"\\"+'finalized_model_oppAbuse.sav'
loaded_model = joblib.load(filename)


# In[117]:



y_pred = loaded_model.predict(X_features)
y_pred1 = pd.DataFrame(y_pred,columns=['Predicted oppAbuse'])
y_pred1.head()


# In[118]:
test_df['File Name'] = fsha_data['File Name']

y_test1 = fsha_data['abuseByConsumer']


# In[119]:


y_test1.index = y_pred1.index 


# In[120]:


test_df['File Name'].index = y_pred1.index 


# In[121]:


output_df = pd.concat([test_df['File Name'],y_test1,y_pred1],axis=1)
output_df.head()


fsha_data = fsha_data.drop(['potentialMicrobial', 'crossContactAllergens', 'chokeHazard',
       'operationalAllergen', 'allergensLabeledIMAF','abuseByConsumer','cookstepByConsumer'],axis = 1)


# In[124]:


final_output = pd.merge(fsha_data,output_df, on='File Name', how='inner')
final_output.head()


# In[125]:


output_ = pd.read_excel(path +"\\"+"ML_Output_Abuse.xlsm")
#print(len(output_.columns))
output_.columns


# In[126]:





# In[127]:


final_output.columns = output_.columns




# In[130]:

final_output['Is there a likely opportunity for abuse by the consumer?- predicted']=final_output['Is there a likely opportunity for abuse by the consumer?- predicted'].apply(lambda x: 'Yes' if x == 1 else 'No' )


# In[131]:


final_output.to_csv(path +"\\"+out_filename,index = False)


# In[ ]:




