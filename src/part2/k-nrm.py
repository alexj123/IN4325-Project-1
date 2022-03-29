import io
import itertools
import random

import numpy as np
import pyterrier as pt
from sklearn.feature_extraction import DictVectorizer
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.ensemble import RandomForestRegressor
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import SGDRegressor

stops = set('for a of the and to in'.split())
model = KeyedVectors.load_word2vec_format("data/glove.6B.50d.txt", no_header=True)

print("Model loaded")


def _try_get_vector(i):
    try:
        return model.get_vector(i, norm=True)
    except KeyError:
        pass


def _remove_stops(sentence):
    terms = sentence.split(" ")
    terms = [t for t in terms if t not in stops]
    return terms


def string_to_vector_array(text):
    text = text.lower()
    terms = _remove_stops(text)
    res = []
    for term in terms:
        val = _try_get_vector(term)
        if val is not None:
            res.append(val)
    return res


def pre_process_query(q):
    return string_to_vector_array(q["query"])


class DocumentVectorCache:
    def __init__(self):
        self.cache = {}

    def lookup(self, docid, text):
        value = self.cache.get(docid, None)
        if value is None:
            value = string_to_vector_array(text)
            self.cache[docid] = value
        return value


def rbf_kernel(m_i, kernel_mean, kernel_var):
    return np.sum(np.exp(-1 * ((m_i - kernel_mean) ** 2) / (2 * kernel_var ** 2)))


# Expect a row Mi
def apply_kernels(row):
    try:
        k_1 = rbf_kernel(row, -0.3, 0.1)
        k_2 = rbf_kernel(row, 0.5, 0.1)
        k_3 = rbf_kernel(row, 1, 0.1)
        return [k_1, k_2, k_3]
    except ValueError:
        print("oof")


# Expects a row kernel pooling
def soft_tf(table):
    s_tf = np.sum(np.log(table), axis=0)
    return s_tf


def compute_sim(row):
    q_vecs = row["query_vec"]
    doc_vecs = vector_cache.lookup(row["docid"], row["text"])

    grid = list(itertools.product(q_vecs, doc_vecs))
    similarity_matrix = np.zeros(len(q_vecs) * len(doc_vecs))
    for i, value in enumerate(grid):
        similarity_matrix[i] = 1 - spatial.distance.cosine(value[0], value[1])
    similarity_matrix = similarity_matrix.reshape((len(q_vecs), len(doc_vecs)))
    matrix = np.apply_along_axis(apply_kernels, 1, similarity_matrix)
    return soft_tf(matrix)


pt.init()
dataset = pt.get_dataset('msmarco_passage')

vector_cache = DocumentVectorCache()

index = pt.IndexFactory.of(
    "E:/Files/uni/in4325/project 1/IN4325-Project-1/src/part2/data/msmarco-passage-index-with-meta")
print(index.getCollectionStatistics().toString())


train_topics = dataset.get_topics("dev")
train_qrels = dataset.get_qrels("dev")

test_topics = dataset.get_topics("test-2019")
test_qrels = dataset.get_qrels("test-2019")

sgd_classifier = SGDRegressor(loss="squared_error", penalty="l2")

pipeline = pt.apply.query_vec(pre_process_query, verbose=True) >> pt.BatchRetrieve(index, wmodel="BM25", verbose=True,
                                                                                   metadata=["docno", "text"]) % 10 >> \
           pt.apply.doc_features(compute_sim, verbose=True) >> pt.ltr.apply_learned_model(sgd_classifier)

print("Fitting model")
pipeline.fit(test_topics, test_qrels)

print("Starting experiment")
res = pipeline.transform(test_topics)

x = 2
