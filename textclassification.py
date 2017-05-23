# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 14:27:47 2017

@author: minven2
"""

import re
import numpy as np
import os
import json
import pandas as pd
import random
import seaborn as sn
import pickle
import time
import sys
import logging
import logging_configs.load_logger as lg
from scipy.sparse import csc_matrix

logger = logging.getLogger("tokenization")

def read_text_file(filename):
    taw_text =  []
    f = open(filename, encoding="utf8")
    for line in f:
        taw_text.append(line)
    return taw_text

def read_json_file(filename):
    with open(filename) as data_file:    
        data = json.load(data_file)
    return data

def strip_strings(iterable):
    # remove white spaces from token endings
    modified_iterable = []
    for element in iterable:
        modified_iterable.append(element.strip())
    return modified_iterable

def get_file_names_from_dir(path):
    return os.listdir(path)

def tf(token, document):
    return document.count(token) / len(document)
    
def idf(token, documents):
    # number of documents where the term t appears
    occurences = 0
    for document in documents:
        if token in document:
            occurences += 1
    return np.log(len(documents) / (occurences))

def compute_similarity(bag_of_words_matrix, unique_tokens_train, test_document):
    """
    Compute similarity between bagOfWords and new document
    bag_of_words_matrix - pandas dataframe
    unique_tokens_train - list
    test_document - list
    create empty vector which can be multiplied by bagOfWordsMatrix
    """
    initial_array = np.zeros((1, len(unique_tokens_train)))
    vector = pd.DataFrame(data=initial_array,
                         index=[1],
                         columns=unique_tokens_train)
    for token in test_document:
        if token in unique_tokens_train:   
            vector.set_value(1, token, 1)           
    # calculate similiraties between test document and each train documents
    similarities = bag_of_words_matrix.dot(vector.transpose())
    return similarities
    
def compute_prediction(similarities, N):
    """
    Prediction based on N nearest neighbours
    similarities - pandas series
    get N  document ids where similarity is biggest
    """
    topn = list(similarities.nlargest(N,[1]).index)
    predictions = []
    for n in topn:
        predictions.append(document_ids_train[n])
    predicted_class = max(set(predictions), key=predictions.count) 
    return predicted_class       
        
def bag_of_words(unique_tokens, documents, document_ids):
    # We need to create pandas data frame where (x,y) -> (document_id, token) = tf_idf
    # Additioanlly last columns is category. I needed it due to testing posibility
    initial_array = np.zeros((len(document_ids), len(unique_tokens)))
    pandas_df = pd.DataFrame(data=initial_array,
                             index=document_ids.keys(),
                             columns=unique_tokens)
    only_documents = documents.values()
    # store idf values at dictionary, because it is more efficient, we
    # don't need to recalculate idf value each time
    idf_values = {}
    for document_id in list(documents.keys()):
        document = documents[document_id]
        for token in set(document):
            tf_value = tf(token, document)
            if token not in idf_values:
                idf_value = idf(token, only_documents)
                idf_values[token] = idf_value
                tf_idf = tf_value * idf_value
            else:
                tf_idf= tf_value * idf_values[token]
            # update value pandas_df value (row,col) -> (document_id, token) = tf_idf
            pandas_df.set_value(document_id, token, tf_idf)
    return pandas_df



def bag_of_words_sparse(unique_tokens, documents, document_ids):
    
    rows_count = len(documents)
    col_count = len(unique_tokens)
    size = int(rows_count*col_count*0.02)
    row = np.zeros(size, dtype = np.int32)
    col = np.zeros(size,  dtype = np.int32)
    data = np.zeros(size,  dtype = np.float32)
    
    documents_ids = list(documents.keys())
    document_ids_mapping = dict(zip(document_ids,np.arange(len(document_ids))))
    tokens_mapping = dict(zip(unique_tokens,np.arange(len(unique_tokens))))
    
    only_documents = documents.values()
    # store idf values at dictionary, because it is more efficient, we
    # don't need to recalculate idf value each time
    idf_values = {}
    i = 0
    for document_id in documents_ids:
        document = documents[document_id]
        for token in set(document):
            tf_value = tf(token, document)
            if token not in idf_values:
                idf_value = idf(token, only_documents)
                idf_values[token] = idf_value
                tf_idf = tf_value * idf_value
            else:
                tf_idf= tf_value * idf_values[token]
            # update value pandas_df value (row,col) -> (document_id, token) = tf_idf
            row[i] = document_ids_mapping[document_id]
            col[i] = tokens_mapping[token]
            data[i] = tf_idf                         
            i += 1
    return csc_matrix((data, (row, col)), shape=(rows_count, col_count))

def reduce_bag_of_words(bag_of_words_matrix):
    attributes = list(bag_of_words_matrix.columns.values)
    # check how many documents have attribute inside
    attributes_in_documents_count =  np.sum(bag_of_words_matrix[attributes] != 0)
    # atributes which appears in documents rarelly 
    rare_attributes =  attributes_in_documents_count == 1
    return bag_of_words_matrix[list(rare_attributes[rare_attributes == False].index)]


def classification_model_performance(confusion_matrix):
    # http://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    # TN = confusion_matrix.values.sum() - (FP + FN + TP)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)    
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    return TPR, PPV
      
class TextClassification:
    
    def __init__(self, document, stop_words, endings):
        self.document = " ".join(document)
        self.stop_words = stop_words
        self.endings = endings
        
    def remove_stop_words(self, token):
        if token not in self.stop_words:
            return token
                
    def remove_digits(self, token):
        if not token.isdigit():
            return token
        
    def remove_small_token(self, token):
        if len(token) >= 4:
            return token
        
    def stemming(self, tokens):
        for i in range(len(tokens)):
            token = tokens[i]
            for ending in self.endings:
                mod_token = re.sub("{}$".format(ending), "", token)
                if (token != mod_token) & (len(mod_token) > 1):
                    tokens[i] = mod_token 
                    break
        return tokens
        
    def tokenize(self):
        self.document = self.document.lower()
        tokens = re.split(r'\W+', self.document)
        modified_tokens = []
        for token in tokens:
            token = self.remove_stop_words(token)
            if token:    
                token = self.remove_digits(token)
            if token:
                token = self.remove_small_token(token)
            if token:
                modified_tokens.append(token)
        return sorted(modified_tokens)
    
        
if __name__ == "__main__":
    lg.configure_logging("logging_configs/logging.json")
    stop_words_name = "nlp_helpers/stop_words.csv"
    ending_name = "nlp_helpers/ending.csv"
    stop_words = read_text_file(stop_words_name)
    stop_words = strip_strings(stop_words)
    logger.debug(" {} stop words".format(len(stop_words)))
    endings = read_text_file(ending_name)  
    endings = strip_strings(endings)
    logger.debug(" {} endings".format(len(endings)))
     
    parent_path = os.getcwd()
    categories_path = os.path.join(os.getcwd(), "Documents")
    categories = get_file_names_from_dir(categories_path)
    logger.info("categories {}".format(categories))
    os.chdir(categories_path)
    
    documents_test = {}
    unique_tokens_test = set()
    document_ids_test = {}
    
    documents_train = {}
    unique_tokens_train = set()
    document_ids_train = {}
    document_titles_train = {}
    
    start_time_tokenization = time.time()
    for category in categories:
        documents_path = os.path.join(os.getcwd(), category)        
        document_path = os.path.join(os.getcwd(), category, "{}.json".format(category))
        json_doc = read_json_file(document_path)

        for document in json_doc:
            doc_parser = TextClassification(document["text"], stop_words, endings)
            
            tokens = doc_parser.tokenize()
            tokens = doc_parser.stemming(tokens)
            if tokens:
                # 20% test, 80% train
                if random.randint(1,10) in [1]:
                    documents_test[document["id"]] = tokens
                    document_ids_test[document["id"]] = category
                else:
                    unique_tokens_train.update(set(tokens))
                    documents_train[document["id"]] = tokens
                    document_ids_train[document["id"]] = category
                    document_titles_train[document["id"]] = document["title"]

    
    logger.info("Tokenization %.2f seconds" % (time.time() - start_time_tokenization))
    logger.info("Train documents {}".format(len(documents_train)))
    logger.info("Test documents {}".format(len(documents_test)))

#    start_time_bag_of_words_sparse = time.time()
#    bag_of_words_matrix = bag_of_words_sparse(unique_tokens_train, documents_train, document_ids_train)
#    bag_of_words_matrix = bag_of_words_matrix.toarray()
#    logger.info("BagOfWorfds sparse %.2f seconds" % (time.time() - start_time_bag_of_words_sparse))
#    logger.info("BagOfWorfds sparse matrix size %.2f MB" % (sys.getsizeof(bag_of_words_matrix) / 1000000))
#    logger.info("BagOfWorfds sparse matrix shape {}".format(bag_of_words_matrix.shape))
    
    start_time_bag_of_words = time.time()
    bag_of_words_matrix = bag_of_words(unique_tokens_train, documents_train, document_ids_train)
    logger.info("BagOfWorfds %.2f seconds" % (time.time() - start_time_bag_of_words))
    logger.info("BagOfWorfds matrix size %.2f MB" % (sys.getsizeof(bag_of_words_matrix) / 1000000))
    logger.info("BagOfWorfds matrix shape {}".format(bag_of_words_matrix.shape))
    start_time_reduce_bag_of_words = time.time()
    bag_of_words_matrix = reduce_bag_of_words(bag_of_words_matrix)
    logger.info("Reduction %.2f seconds" % (time.time() - start_time_reduce_bag_of_words))
    logger.info("Reduced matrix size %.2f MB" % (sys.getsizeof(bag_of_words_matrix) / 1000000))
    logger.info("Reduced matrix shape {}".format(bag_of_words_matrix.shape))


    start_time_testing = time.time()
    confusion_matrix_ini = np.zeros((len(categories), len(categories)))
    confusion_matrix = pd.DataFrame(data=confusion_matrix_ini,
                                    index=categories,
                                    columns=categories)
    
    unique_tokens_train = list(bag_of_words_matrix.columns.values)
    for document_id in list(document_ids_test.keys()):
        document_test = documents_test[document_id]
        true_class = document_ids_test[document_id]
        similarities = compute_similarity(bag_of_words_matrix, unique_tokens_train, document_test)    
        predicted_class = compute_prediction(similarities, 5)
        
        new_value = confusion_matrix[true_class][predicted_class] + 1
        confusion_matrix.set_value(true_class, predicted_class, new_value)
        
    logger.info("Testing %.2f seconds" % (time.time() - start_time_testing))
    
    
    os.chdir(parent_path) 
    sns_plot = sn.heatmap(confusion_matrix, annot=True, cmap="YlGnBu", square=True)
    sns_plot.figure.savefig("visualizations/confussion_matrix.png")
     
    # model performance
    TPR, PPV = classification_model_performance(confusion_matrix)
    logger.info("Recall = {} and Precision = {}".format(TPR.mean(), PPV.mean()))
    # save documents dict
    pickle.dump(documents_train, open("pickles/documents_train.p", "wb"))
    pickle.dump(unique_tokens_train, open("pickles/unique_tokens_train.p", "wb"))
    pickle.dump(document_ids_train, open("pickles/document_ids_train.p", "wb"))
    pickle.dump(bag_of_words_matrix, open("pickles/bag_of_words_matrix.p", "wb"))
    pickle.dump(document_titles_train, open("pickles/document_titles_train.p", "wb"))
#    rez = {}
#    for document_id in documents_train.keys():
#        thrashold = max_values[document_id] -0.05
#        thrashold_boolean = subset_matrix.loc[document_id] > thrashold
#        items = subset_matrix.loc[document_id][thrashold_boolean].index.values
#        rez[document_id] = items

    
    