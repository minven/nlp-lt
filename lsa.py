# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 21:31:55 2017

@author: minven2
"""
# provides a method for determining the similarity of meaning of words and passages by
# analysis of large text corpora.
# Used for document classification, clustering, text search, and more

# LSA assumes that words that are close in meaning will occur in similar pieces of text

# A typical example of the weighting of the elements of the matrix is tf-idf 
# the weight of an element of the matrix is proportional to the number of times
# the terms appear in each document, where rare terms are upweighted to reflect their relative importance.
# This mitigates the problem of identifying synonymy, as the rank lowering is
# expected to merge the dimensions associated with terms that have similar meanings. 


# https://en.wikipedia.org/wiki/Latent_semantic_analysis#Rank_lowering
# https://jeremykun.com/2016/04/18/singular-value-decomposition-part-1-perspectives-on-linear-algebra/
# http://webhome.cs.uvic.ca/~thomo/svd.pdf

# U - singular vectors
# S - eigenvalues
# V - singular vectors

import pickle
import numpy as np
import matplotlib.pyplot as pyplot
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd
from gensim import corpora, models, matutils
from sklearn.metrics.pairwise import cosine_similarity



class LSA(object):
    
    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.bag_of_words_df = self._load_pickle("bag_of_words_matrix.p")
        self.features = list(self.bag_of_words_df.columns)
        self.tokens_count, self.documents_count = self.bag_of_words_df.shape
        self.documents_mapping = dict(zip(list(range(self.documents_count)),
                                          list(self.bag_of_words_df.index)))
        self.tokens_mapping = dict(zip(list(self.bag_of_words_df.columns),
                                       list(range(self.tokens_count))))
        self.documents_titles = self._load_pickle("document_titles_train.p")
        # https://stats.stackexchange.com/questions/69157/why-do-we-need-to-normalize-data-before-analysis
        self.bag_of_words_matrix = normalize(self.bag_of_words_df.as_matrix(), axis=1, norm="l2")
        #self.bag_of_words_matrix = self.bag_of_words_df.as_matrix()
        self.components = []
        
    def _load_pickle(self, pickle_name):
        pickle_obj = pickle.load( open( "pickles/{}".format(pickle_name), "rb" ))       
        return pickle_obj

            
    def plot_main_components(self):
        # http://mccormickml.com/2016/03/25/lsa-for-text-classification-tutorial/

        components_numb = len(self.components)
        terms_numb = len(self.components[0])
        fig_col = 2
        fig_row = components_numb // 2
        f, ax = pyplot.subplots(fig_row, fig_col, figsize=(15, 20))
        terms_count = len(self.components[0])
        for i, latent in enumerate(self.components):
            weights = [np.abs(term[1]) for term in latent]
            terms = [term[0] for term in latent]
            positions = np.arange(terms_count) + .5    # the bar centers on the y axis
            ax[i//2, i%2].barh(positions, weights, align='center', alpha=0.5)
            ax[i//2, i%2].set_yticks(positions)
            ax[i//2, i%2].set_yticklabels(terms, rotation="horizontal")                         
            ax[i//2, i%2].set_title("%s principal component"%(i+1))
        f.subplots_adjust(hspace=0.5)
        pyplot.savefig("visualizations/main_term_components.png")

    def explore_bag_of_words_matrix(self):
        doc_means = self.bag_of_words_matrix.mean(1)
        doc_std = self.bag_of_words_matrix.std(1)
        tokens_means = self.bag_of_words_matrix.mean(0)
        tokens_std = self.bag_of_words_matrix.std(0)
        return tokens_means, doc_means, doc_std, tokens_std
            
    def show_topic(self, component_numb, topn):
        # https://stats.stackexchange.com/questions/107533/how-to-use-svd-for-dimensionality-reduction-to-reduce-the-number-of-columns-fea
        # https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/lsimodel.py
        nth_component = np.asarray(self.U.T[component_numb, :]).flatten()
        most_lsa = matutils.argsort(np.abs(nth_component), topn, reverse=True)
        terms = [(lsa_instance.features[weightIndex], nth_component[weightIndex]) for weightIndex in most_lsa]
        return terms        
        
    
    def search_query(self, query):
        """
        search for query and find most related document for query
        http://webhome.cs.uvic.ca/~thomo/svd.pdf
        """
        
        def topN(similarities, N=5):
            return np.argsort(similarities)[::-1][:N]
        
        words = query.split(" ")
        tokens_ids = []
        for word in words:
            try:
                token_id = self.tokens_mapping[word]
            except KeyError:
                print("Token not found in tokens mapping dict")
            else:
                tokens_ids.append(token_id)
        
        query_representation = np.mean(self.tokens_representation[tokens_ids,:], axis=0)
        similarities = cosine_similarity(query_representation, self.documents_representation)
        topN_documents =[self.documents_mapping[index] for index in topN(similarities[0])] 
        return topN_documents
        # compute cosine distance between query_representation and each document 
    
    def generate_components(self, components_numb, topn):
        components = []
        for i in range(components_numb):
            latent = self.show_topic(component_numb = i, topn = topn)
            components.append(latent)
        self.components = components
    
    def truncated_svd(self):
        # https://github.com/chrisjmccormick/LSA_Classification/blob/master/inspect_LSA.py
        svd = TruncatedSVD(self.dimensions)   
        lsa = make_pipeline(svd, Normalizer(copy=False))
        X_reduced = lsa.fit_transform(self.bag_of_words_matrix)
        print(svd.components_[0])
        print(svd.explained_variance_ratio_) 
        print(svd.explained_variance_ratio_.sum())
        
    def randomizedSVD(self):
        # http://scikit-learn.org/stable/modules/decomposition.html#truncated-singular-value-decomposition-and-latent-semantic-analysis
        # http://stackoverflow.com/questions/31523575/get-u-sigma-v-matrix-from-truncated-svd-in-scikit-learn
        U, S, V = randomized_svd(self.bag_of_words_matrix.T, 
                                      n_components=self.dimensions,
                                      n_iter=5,
                                      random_state=None)
        self.U = U
        self.S = S
        self.V = V
        self.tokens_representation = np.matrix(U) * np.diag(S)
        self.documents_representation = (np.diag(S) * np.matrix(V)).T
    
    
    def SVD(self):
        ## https://github.com/josephwilk/semanticpy/blob/master/semanticpy/transform/lsa.py
        bag_of_words_matrix = self.bag_of_words_matrix.T
        rows,cols = self.bag_of_words_matrix.shape
        U, S, V = np.linalg.svd(bag_of_words_matrix, full_matrices=False)
        self.U = U[:,:self.dimensions]
        self.S = S[:self.dimensions]
        self.V = V[:self.dimensions,:]
        #transformed_matrix = np.dot(np.dot(U, linalg.diagsvd(S, len(self.bag_of_words_matrix), len(V))) ,V)

    def gensim(self):
        # https://radimrehurek.com/gensim/dist_lsi.html
        # https://radimrehurek.com/gensim/models/lsimodel.html
        corpus = corpora.MmCorpus('../lda/lda_sources/documents_corpus.mm')        
        id2word = corpora.Dictionary.load('../lda/lda_sources/documents_dictionary.dict')
        lsi = models.LsiModel(corpus, id2word=id2word, num_topics=self.dimensions)
        return lsi
    


if __name__ == "__main__":




#    #gensim approach
#    lsa_instance = LSA(200)
#    gensim_model = lsa_instance.gensim()
#    U_gensim = gensim_model.projection.u
#    S_gensim = gensim_model.projection.s
#    US_gensim = np.dot(U_gensim, linalg.diagsvd(S_gensim,200, 200))
#    # Return a specified topic (=left singular vector),
#    topic_10_gensim = gensim_model.show_topic(3, topn = 5)
#    topics_gensim = gensim_model.show_topics(num_topics = 0, num_words = 6)

    
    # Randomized SVD Approach
    lsa_instance = LSA(dimensions=150)
    documents_mapping = lsa_instance.documents_mapping
    tokens_mapping = lsa_instance.tokens_mapping
    lsa_instance.randomizedSVD()
    U = lsa_instance.U
    S = lsa_instance.S
    V = lsa_instance.V
    tokens_representation = lsa_instance.tokens_representation
    documents_representation = lsa_instance.documents_representation
    all_tokens = lsa_instance.features
   
    
    """
    Plotting main token components
    """
    
    lsa_instance.generate_components(components_numb=6, topn=10)
    components = lsa_instance.components
    lsa_instance.plot_main_components()
    
    
    """
    Searching query example
    """
#    suited_docs = lsa_instance.search_query("apdovanojam automobil")
#    for suited_doc in suited_docs:
#        print(lsa_instance.documents_titles[suited_doc])
    

    
    
    
    
#   tokens_means, doc_means, doc_std, tokens_std = lsa_instance.explore_bag_of_words_matrix()      
#    pyplot.hist(doc_std, bins=10)
#    pyplot.title("documents standart deviation")
#    pyplot.show()
#    pyplot.hist(tokens_std, bins=10)
#    pyplot.title("tokens standart deviation")
#    pyplot.show()
    
#    # SVD Approach
#    lsa_instance = LSA(200)
#    lsa_instance.SVD()
#    U = lsa_instance.U
#    S = lsa_instance.S
#    V = lsa_instance.V
#    
#    lsa_instance.generate_components(components_numb=5, topn=5)
#    components = lsa_instance.components
#    lsa_instance.plot_main_components()
#    tokens_means, doc_means, doc_std, tokens_std = lsa_instance.explore_bag_of_words_matrix() 



