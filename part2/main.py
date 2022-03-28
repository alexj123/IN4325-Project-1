import io
import numpy as np
import pyterrier as pt
from sklearn.feature_extraction import DictVectorizer
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.ensemble import RandomForestRegressor


stops = set('for a of the and to in'.split())


def _remove_stops(q):
    terms = q["query"].split(" ")
    terms = [t for t in terms if t not in stops]
    return " ".join(terms)


def compute_sim(keyFreq, posting, entryStats, collStats):
    x = 2
    return posting.getFrequency()


def load_vectors(f_name) -> KeyedVectors:
    _model = KeyedVectors.load_word2vec_format(f_name, no_header=True)
    return _model


model = load_vectors("data/glove.6B.50d.txt")
print("Model loaded")

pt.init()
dataset = pt.get_dataset('msmarco_passage')

index = pt.IndexFactory.of("E:/Files/uni/in4325/project 1/IN4325-Project-1/src/data/msmarco_passage-index")

test_topics = dataset.get_topics("test-2019")
test_qrels = dataset.get_qrels("test-2019")

r = pt.apply.query(_remove_stops, verbose=True) >> pt.BatchRetrieve(index, wmodel=compute_sim)

l = r.transform(test_topics)

# df = pt.Experiment([r], test_topics, test_qrels)
# df.to_csv("l2r_res.csv")

x = 2