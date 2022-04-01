import itertools

import numpy as np
from gensim.models import KeyedVectors, FastText
from scipy import spatial


class DocumentVectorCache:
    def __init__(self):
        self.cache = {}

    def lookup(self, docid, text):
        value = self.cache.get(docid, None)
        if value is None:
            value = string_to_vector_array(text)
            self.cache[docid] = value
        return value


vector_cache = DocumentVectorCache()
model = KeyedVectors.load_word2vec_format("data/glove.6B.300d.txt", no_header=True)
# model = KeyedVectors.load_word2vec_format("data/wiki-news-300d-1M.vec")
stops = set('for a of the and to in'.split())
print("Vectors loaded")


def rbf_kernel(m_i, kernel_mean, kernel_var):
    return np.sum(np.exp(-1 * ((m_i - kernel_mean) ** 2) / (2 * (kernel_var ** 2))))


# Expect a row Mi
def apply_kernels(row):
    try:
        k_1 = rbf_kernel(row, 1, 0.001)
        k_2 = rbf_kernel(row, 0.9, 0.1)
        k_3 = rbf_kernel(row, 0.7, 0.1)
        k_4 = rbf_kernel(row, 0.5, 0.1)
        k_5 = rbf_kernel(row, 0.3, 0.1)
        k_6 = rbf_kernel(row, 0.1, 0.1)
        k_7 = rbf_kernel(row, -0.1, 0.1)
        k_8 = rbf_kernel(row, -0.3, 0.1)
        k_9 = rbf_kernel(row, -0.5, 0.1)
        k_10 = rbf_kernel(row, -0.7, 0.1)
        k_11 = rbf_kernel(row, -0.9, 0.1)
        return [k_1, k_2, k_3, k_4, k_5, k_6, k_7, k_8, k_9, k_10, k_11]
    except ValueError:
        print("oof")


# Expects a row kernel pooling
def soft_tf(table):
    inter = np.sum(np.log(table), axis=0)
    inter[np.isneginf(inter)] = 0
    return inter


def compute_cosine_similarity(row):
    query_vectors = row["query_vec"]
    document_vectors = vector_cache.lookup(row["docid"], row["text"])
    if len(query_vectors) == 0 or len(document_vectors) == 0:
        return np.zeros(11)

    grid = list(itertools.product(query_vectors, document_vectors))
    similarity_matrix = np.zeros(len(query_vectors) * len(document_vectors))
    
    for i, value in enumerate(grid):
        similarity_matrix[i] = 1 - spatial.distance.cosine(value[0], value[1])
        
    similarity_matrix = similarity_matrix.reshape((len(query_vectors), len(document_vectors)))

    kernel_pool = np.apply_along_axis(apply_kernels, 1, similarity_matrix)
    kernel_features = soft_tf(kernel_pool)

    return kernel_features


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
