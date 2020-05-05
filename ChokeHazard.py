#!/usr/bin/env python
# coding: utf-8

# # This code does loading of the model and prediction on test set of the Output : "Are there any consumer suitability or choke hazard concerns?- actual" in the FSHA form.The text is cleaned, normalized and then vectorized and passed into the model for predictions.


#import all necessary modules
from __future__ import print_function
import logging
from optparse import OptionParser
import sys
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import unicodedata
from sklearn.externals import joblib
import statistics 
import warnings
warnings.simplefilter('ignore')

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

#Read input file with extracted data
f1 = path + "\\" + filename
fsha_data = pd.read_excel(f1)

#selecting set of columns as Features
fsha_data.fillna('NA', inplace=True)
features_df=fsha_data[['projDesc','PDA_projName','packMaterial', 'CPD-ProdName','CPD-ProdName-Desc','TCG', 'labelingInstructions',
        'mishandled', 'targetMarket','File Name']]


def conv_str(x):
   
    x=str(x)
    x=x.lower()
    return (x)

features_df['projDesc']=features_df['projDesc'].apply(lambda x:conv_str(x))  
features_df['PDA_projName']=features_df['PDA_projName'].apply(lambda x:conv_str(x))
features_df['packMaterial']=features_df['packMaterial'].apply(lambda x:conv_str(x))  
features_df['CPD-ProdName']=features_df['CPD-ProdName'].apply(lambda x:conv_str(x))  
features_df['CPD-ProdName-Desc']=features_df['CPD-ProdName-Desc'].apply(lambda x:conv_str(x))  
features_df['TCG']=features_df['TCG'].apply(lambda x:conv_str(x)) 
features_df['labelingInstructions']=features_df['labelingInstructions'].apply(lambda x:conv_str(x)) 
features_df['mishandled']=features_df['mishandled'].apply(lambda x:conv_str(x))  
features_df['targetMarket']=features_df['targetMarket'].apply(lambda x:conv_str(x)) 

test_df = features_df

test_df['projDesc']=normalize_corpus(test_df['projDesc'])
test_df['PDA_projName']=normalize_corpus(test_df['PDA_projName'])
test_df['packMaterial']=normalize_corpus(test_df['packMaterial'])  
test_df['CPD-ProdName']=normalize_corpus(test_df['CPD-ProdName'])  
test_df['CPD-ProdName-Desc']=normalize_corpus(test_df['CPD-ProdName-Desc'])
test_df['TCG']=normalize_corpus(test_df['TCG']) 
test_df['labelingInstructions']=normalize_corpus(test_df['labelingInstructions']) 
test_df['mishandled']=normalize_corpus(test_df['mishandled']) 
test_df['targetMarket']=normalize_corpus(test_df['targetMarket'])




def preprocess_text(test_df):    
    scaler_load = joblib.load(path +"\\"+"scaler_chokehazard.pkl")
    pca_load = joblib.load(path +"\\"+"pca_chokehazard.pkl")
    vectorizer_load = joblib.load(path +"\\"+"vectorizer_chokehazard.pkl")
    x_ngram_projDesc=vectorizer_load.transform(test_df['projDesc']).toarray()
    x_ngram_PDA_projName=vectorizer_load.transform(test_df['PDA_projName']).toarray()
    x_ngram_packMaterial=vectorizer_load.transform(test_df['packMaterial']).toarray() 
    x_ngram_CPD_ProdName=vectorizer_load.transform(test_df['CPD-ProdName']).toarray() 
    x_ngram_CPD_ProdName_Desc=vectorizer_load.transform(test_df['CPD-ProdName-Desc']).toarray()
    x_ngram_TCG=vectorizer_load.transform(test_df['TCG']).toarray() 
    x_ngram_labelingInstructions=vectorizer_load.transform(test_df['labelingInstructions']).toarray()
    x_ngram_mishandled=vectorizer_load.transform(test_df['mishandled']).toarray() 
    x_ngram_targetMarket=vectorizer_load.transform(test_df['targetMarket']).toarray()
    test_df = test_df.drop(['projDesc','PDA_projName','packMaterial','CPD-ProdName','TCG','labelingInstructions','mishandled','targetMarket'],axis=1)
    
    x_ngram_projDesc = scaler_load.transform(x_ngram_projDesc)
    x_ngram_PDA_projName = scaler_load.transform(x_ngram_PDA_projName)
    x_ngram_packMaterial = scaler_load.transform(x_ngram_packMaterial)
    x_ngram_CPD_ProdName = scaler_load.transform(x_ngram_CPD_ProdName)
    x_ngram_CPD_ProdName_Desc = scaler_load.transform(x_ngram_CPD_ProdName_Desc)
    x_ngram_TCG = scaler_load.transform(x_ngram_TCG)
    x_ngram_labelingInstructions = scaler_load.transform(x_ngram_labelingInstructions)
    x_ngram_mishandled = scaler_load.transform(x_ngram_mishandled)
    x_ngram_targetMarket = scaler_load.transform(x_ngram_targetMarket)
    
    x_pca_projDesc = pca_load.transform(x_ngram_projDesc)
    x_pca_PDA_projName = pca_load.transform(x_ngram_PDA_projName)
    x_pca_packMaterial = pca_load.transform(x_ngram_packMaterial)
    x_pca_CPD_ProdName = pca_load.transform(x_ngram_CPD_ProdName)
    x_pca_CPD_ProdName_Desc = pca_load.transform(x_ngram_CPD_ProdName_Desc)
    x_pca_TCG = pca_load.transform(x_ngram_TCG)
    x_pca_labelingInstructions = pca_load.transform(x_ngram_labelingInstructions)
    x_pca_mishandled = pca_load.transform(x_ngram_mishandled)
    x_pca_targetMarket = pca_load.transform(x_ngram_targetMarket)

    x_train = np.concatenate((x_pca_projDesc,x_pca_PDA_projName,x_pca_packMaterial,x_pca_CPD_ProdName,x_pca_CPD_ProdName_Desc,x_pca_TCG,x_pca_labelingInstructions,x_pca_mishandled,x_pca_targetMarket),axis=1)

    return x_train 


X_features = preprocess_text(test_df)


X_features = pd.DataFrame(X_features)


X_features.reset_index(drop=True, inplace=True)



# # Loading the model for predictions on validation data to be provided

filename = path +"\\"+'finalized_model_ChokeHazard.sav'
loaded_model = joblib.load(filename)

print("Saved model loaded")

y_pred = loaded_model.predict(X_features)
y_pred1 = pd.DataFrame(y_pred,columns=['Predicted ChokeHazard'])
y_pred1.head()

print("Predictions done")

test_df['File Name'] = fsha_data['File Name']
y_test1 = fsha_data['chokeHazard']
y_test1.head()

y_pred1.index = y_test1.index

features_df['File Name'].index = y_test1.index

output_df = pd.concat([test_df['File Name'],y_test1,y_pred1],axis=1)
output_df.head()

fsha_data = fsha_data.drop(['potentialMicrobial', 'crossContactAllergens', 'chokeHazard',
       'operationalAllergen', 'allergensLabeledIMAF','abuseByConsumer','cookstepByConsumer'],axis = 1)
final_output = pd.merge(fsha_data,output_df, on='File Name', how='inner')
final_output.head()

output_ = pd.read_excel(path +"\\"+"ML_Output_ChokeHazard.xlsm")
len(output_.columns)
output_.columns


final_output.columns = output_.columns


final_output['Are there any consumer suitability or choke hazard concerns?- predicted'] = final_output['Are there any consumer suitability or choke hazard concerns?- predicted'].apply(lambda x: 'Yes' if x == 1 else 'No' )

final_output.to_csv(path +"\\"+out_filename,index = False)



