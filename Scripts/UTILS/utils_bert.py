import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
import os
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import sys
sys.path.append('../')
import UTILS.utils as utils
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score

DEFAULT_MIN_NUM_TOKENS = 9
DEFAULT_PARAMS = {"ngram_range" : (1,2), "random_state" : 480, "min_samples" : 10,
               "min_topic_size" : 18, "n_neighbors" : 6, "n_components" : 6} 

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class HyperParameters:

    def __init__(self, parameters = None):
        if parameters is None:
            self.assign_param_value(DEFAULT_PARAMS)
        else:
            self.assign_param_value(parameters)

    def assign_param_value(self, parameters : dict):
        self.ngram_range = parameters.get("ngram_range", DEFAULT_PARAMS["ngram_range"])
        self.random_state = parameters.get("random_state", DEFAULT_PARAMS["random_state"])
        self.min_samples = parameters.get("min_samples", DEFAULT_PARAMS["min_samples"])
        self.min_topic_size = parameters.get("min_topic_size", DEFAULT_PARAMS["min_topic_size"])
        self.n_neighbors = parameters.get("n_neighbors", DEFAULT_PARAMS["n_neighbors"])
        self.n_components = parameters.get("n_components", DEFAULT_PARAMS["n_components"])

def get_model_and_fit_docs(docs):
    vectorizer_model = CountVectorizer(ngram_range=(1,2), stop_words="english")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    topic_model = BERTopic(
        vectorizer_model=vectorizer_model,
        embedding_model=embedding_model,
        language='english', calculate_probabilities=True,
        verbose=False
    )

    docs = docs["text"].to_list()
    topics, probs = topic_model.fit_transform(docs)
    return topic_model


def get_model(parameters : HyperParameters = None):

    if parameters is None:
        parameters = HyperParameters()

    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    vectorizer_model = CountVectorizer(ngram_range=parameters.ngram_range, stop_words="english")

    umap_model = UMAP(n_neighbors=parameters.n_neighbors, n_components=parameters.n_components, metric='cosine', low_memory=False, random_state=parameters.random_state)
    hdbscan_model = HDBSCAN(min_cluster_size=parameters.min_topic_size, metric='euclidean', prediction_data=True, min_samples=parameters.min_samples)
    topic_model = BERTopic(
        vectorizer_model=vectorizer_model,
        embedding_model=embedding_model,
        hdbscan_model=hdbscan_model,
        umap_model=umap_model,
        language='english', calculate_probabilities=True,
        verbose=False)
    
    return topic_model

def filter_docs_by_length(docs_df : pd.DataFrame, min_tokens = DEFAULT_MIN_NUM_TOKENS):
    filtered_docs =  docs_df[docs_df["num_tokens"] >= min_tokens].reset_index(drop=True)
    return filtered_docs

def get_coherence_and_silhouette_score(topic_model : BERTopic, docs_df):

    if(not isinstance(docs_df, list)):
        docs = docs_df["text"].to_list()
    else:
        docs = docs_df
    topics, probs = topic_model.fit_transform(docs)

    vectorizer = topic_model.vectorizer_model
    analyzer = vectorizer.build_analyzer()
    tokens = [analyzer(doc) for doc in docs]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    topics = topic_model.get_topics()

    topic_words = [
        [word for word, _ in topic_model.get_topic(topic) if word != ""] for topic in topics
    ]

    coherence_model = CoherenceModel(topics=topic_words, 
                                texts=tokens, 
                                corpus=corpus,
                                dictionary=dictionary, 
                                coherence='c_v')
    coherence = coherence_model.get_coherence()

    embeddings = topic_model.embedding_model.embed(docs)
    silhouette = silhouette_score(X=embeddings, labels=topic_model.topics_)

    return {"coherence" : coherence, "silhouette" : silhouette, "num_topics" : str(len(topics))}

