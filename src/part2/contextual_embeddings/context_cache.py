from sentence_transformers import SentenceTransformer, util
import pandas as pd


class ContextCache:

    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        self.cache = {}

    def lookup(self, key, text):
        value = self.cache.get(key, None)
        if value is None:
            value = self.model.encode([text.lower()], convert_to_tensor=True)
            self.cache[key] = value
        return value

    def compute_cosine_similarity(self, row):
        query_vector = row["query_vec"]
        document_vector = self.lookup(row["docid"], row["text"])

        return util.cos_sim(query_vector, document_vector)
