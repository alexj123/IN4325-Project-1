import itertools

import numpy as np
from gensim.models import KeyedVectors, FastText
from scipy import spatial

stops = set('for a of the and to in'.split())


def _remove_stops(sentence):
    terms = sentence.split(" ")
    terms = [t for t in terms if t not in stops]
    return terms


class DocumentVectorCache:
    def __init__(self, model):
        self.cache = {}
        self.model = model

    def lookup(self, docid, text):
        value = self.cache.get(docid, None)
        if value is None:
            value = self.string_to_vector_array(text)
            self.cache[docid] = value
        return value

    def string_to_vector_array(self, text):
        text = text.lower()
        terms = _remove_stops(text)
        res = []
        for term in terms:
            val = self._try_get_vector(term)
            if val is not None:
                res.append(val)
        return res

    def pre_process_query(self, q):
        return self.string_to_vector_array(q["query"])

    def _try_get_vector(self, i):
        try:
            return self.model.get_vector(i, norm=True)
        except KeyError:
            pass


class KNRM:
    def __init__(self, vec_path, no_header):
        self.model = KeyedVectors.load_word2vec_format(vec_path, no_header=no_header)
        self.vector_cache = DocumentVectorCache(self.model)
        # model = KeyedVectors.load_word2vec_format("data/wiki-news-300d-1M.vec")
        print("Vectors loaded")

    @staticmethod
    def rbf_kernel(m_i, kernel_mean, kernel_var):
        return np.sum(np.exp(-1 * ((m_i - kernel_mean) ** 2) / (2 * (kernel_var ** 2))))

    # Expect a row Mi
    def apply_kernels(self, row):
        try:
            k_1 = self.rbf_kernel(row, 1, 0.001)
            k_2 = self.rbf_kernel(row, 0.9, 0.1)
            k_3 = self.rbf_kernel(row, 0.7, 0.1)
            k_4 = self.rbf_kernel(row, 0.5, 0.1)
            k_5 = self.rbf_kernel(row, 0.3, 0.1)
            k_6 = self.rbf_kernel(row, 0.1, 0.1)
            k_7 = self.rbf_kernel(row, -0.1, 0.1)
            k_8 = self.rbf_kernel(row, -0.3, 0.1)
            k_9 = self.rbf_kernel(row, -0.5, 0.1)
            k_10 = self.rbf_kernel(row, -0.7, 0.1)
            k_11 = self.rbf_kernel(row, -0.9, 0.1)
            return [k_1, k_2, k_3, k_4, k_5, k_6, k_7, k_8, k_9, k_10, k_11]
        except ValueError:
            print("oof")

    # Expects a row kernel pooling
    @staticmethod
    def soft_tf(table):
        inter = np.sum(np.log(table), axis=0)
        inter[np.isneginf(inter)] = 0
        return inter

    def compute_cosine_similarity(self, row):
        query_vectors = row["query_vec"]
        document_vectors = self.vector_cache.lookup(row["docid"], row["text"])
        if len(query_vectors) == 0 or len(document_vectors) == 0:
            return np.zeros(11)

        grid = list(itertools.product(query_vectors, document_vectors))
        similarity_matrix = np.zeros(len(query_vectors) * len(document_vectors))

        for i, value in enumerate(grid):
            similarity_matrix[i] = 1 - spatial.distance.cosine(value[0], value[1])

        similarity_matrix = similarity_matrix.reshape((len(query_vectors), len(document_vectors)))

        kernel_pool = np.apply_along_axis(self.apply_kernels, 1, similarity_matrix)
        kernel_features = self.soft_tf(kernel_pool)

        return kernel_features

    def pre_process_query(self, q):
        return self.vector_cache.pre_process_query(q)
