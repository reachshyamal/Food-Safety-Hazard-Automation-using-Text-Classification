#!/usr/bin/env python
# coding: utf-8

# # This notebooks does predictive modeling of the Output : "Is there any cook step by the consumer?" in the FSHA form 
# # This notebook applies basic ML techniques like NaiveBayesClassifier, LogisticRegression, SGDClassifier on the concatenated text derived from the set of features. The text is cleaned, normalized and then vectorized and fit into the Model. The target is upsampled to counter the unbalanced data ('Yes' / 'No' values), before applying ML

# In[36]:


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
from nltk.corpus import stopwords
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


# In[14]:


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



# # Read the extract file, check the number of records, first few rows

# In[15]:


#filename = "C:\\Users\\574977\\Downloads\\PycharmProjects\\pepsico\\shymal\\Inputfiles\\FSHA_input.xlsx"
f1 = path + "\\" + filename
df = pd.read_excel(f1)


# # Define reusable modular method for Text Normalization (removal of stopwords, changing to lower case, removal of punctuation etc) 

# In[16]:


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
stop = stopwords.words('english')
stop.extend(['na','n', 'none'])
STOPWORDS=set(stop)

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text


# # Based on Analysis select the Features (X)

# In[17]:


features_df=df[['projDesc','CPD-ProdName','cookedOrHeated','labelingInstructions','CPD-ProdName-Desc','packMaterial']]

#Replace missing values with NA
df.fillna('NA', inplace=True)


# In[18]:


#Convert all text to string and lower case, since some values are coming as float from data extract
def conv_str(x):
   
    x=str(x)
    x=x.lower()
    return (x)  
features_df['projDesc']=features_df['projDesc'].apply(lambda x:conv_str(x))    
features_df['CPD-ProdName-Desc']=features_df['CPD-ProdName-Desc'].apply(lambda x:conv_str(x))  
features_df['CPD-ProdName']=features_df['CPD-ProdName'].apply(lambda x:conv_str(x)) 
features_df['cookedOrHeated']=features_df['cookedOrHeated'].apply(lambda x:conv_str(x))
features_df['labelingInstructions']=features_df['labelingInstructions'].apply(lambda x:conv_str(x))
features_df['packMaterial']=features_df['packMaterial'].apply(lambda x:conv_str(x))
 


# # Concatenate the columns as one single text value

# In[19]:


features_df['concat_text']=features_df.apply(' '.join, axis=1)


# # Cleaning the text

# In[20]:


features_df['concat_text'] = features_df['concat_text'].apply(clean_text)


# # Loading the model for predictions on validation data to be provided

# In[21]:


filename = path +"\\"+ "finalized_model_cookstepByConsumer.sav"
loaded_model = joblib.load(filename)


# In[22]:


y_pred = loaded_model.predict(features_df.concat_text)
y_pred1 = pd.DataFrame(y_pred,columns=['Predicted cookstep'])


# In[23]:


y_test1 =  df['cookstepByConsumer']


# In[24]:


features_df['File Name']= df['File Name']


# In[25]:


y_test1.index = y_pred1.index 


# In[26]:


features_df['File Name'].index = y_pred1.index 


# In[27]:


output_df = pd.concat([features_df['File Name'],y_test1,y_pred1],axis=1)
output_df.head()


# In[28]:


df.index = output_df.index


# In[29]:


df = df.drop(['potentialMicrobial', 'crossContactAllergens', 'chokeHazard',
       'operationalAllergen', 'abuseByConsumer', 'cookstepByConsumer',
       'allergensLabeledIMAF'],axis = 1)


# In[30]:


final_output = pd.merge(df,output_df, on='File Name', how='inner')


# In[31]:


output_ = pd.read_excel(path + "\\"+"ML_Output_cookstep.xlsm")


# In[32]:


final_output.columns = output_.columns


# In[33]:


final_output['Is there a cook step by the consumer?- predicted'] = final_output['Is there a cook step by the consumer?- predicted'].apply(lambda x: 'Yes' if x == 1 else 'No' )


# In[35]:


#out_filename = "Final output format_cookstep.csv" 
final_output.to_csv(path + "\\"+out_filename,index = False)

