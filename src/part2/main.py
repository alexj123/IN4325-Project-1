import io
import random

import numpy as np
import pyterrier as pt
from sklearn.feature_extraction import DictVectorizer
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.ensemble import RandomForestRegressor

stops = set('for a of the and to in'.split())


def pre_process_query(q):
    return _remove_stops(q["query"])


def _remove_stops(sentence):
    terms = sentence.split(" ")
    terms = [t for t in terms if t not in stops]
    return " ".join(terms)


def compute_sim(row):
    q_terms = row["query"].split(" ")
    t_terms = _remove_stops(row["text"]).split(" ")
    sims = []
    sim_matrix = np.fromfunction(lambda i, j: model.relative_cosine_similarity(q_terms[i], t_terms[j]), (len(q_terms), len(t_terms)))

    return sum(sims)


def load_vectors(f_name) -> KeyedVectors:
    _model = KeyedVectors.load_word2vec_format(f_name, no_header=True)
    return _model


model = load_vectors("data/glove.6B.50d.txt")
print("Model loaded")

pt.init()
dataset = pt.get_dataset('msmarco_passage')

index = pt.IndexFactory.of("D:/Projects/Uni/IN4325-Project-1/src/part2/data/msmarco-passage-index-with-meta")
print(index.getCollectionStatistics().toString())

test_topics = dataset.get_topics("test-2019")
test_qrels = dataset.get_qrels("test-2019")

r = pt.apply.query(pre_process_query, verbose=True) >> pt.BatchRetrieve(index, wmodel="BM25", verbose=True, metadata=["docno", "text"]) % 3 >> \
    pt.apply.doc_features(compute_sim, verbose=True)  # Reranking using cos

l = r.transform(test_topics)

# df = pt.Experiment([r], test_topics, test_qrels)
# df.to_csv("l2r_res.csv")

x = 2
