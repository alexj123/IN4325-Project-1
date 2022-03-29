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
import joblib
from k_nrm import DocumentVectorCache, pre_process_query, compute_cosine_similarity


def get_model(preload, path=None):
    if not preload:
        print("New model instantiated")
        return SGDRegressor(loss="squared_error", penalty="l2")
    else:
        if path is None:
            raise ValueError("Path is required if preload is true")
        print("Model loaded")
        return joblib.load(path)


def transform_with_model(preload=False, path=None):
    test_topics = dataset.get_topics("test-2019")
    test_qrels = dataset.get_qrels("test-2019")

    sgd_classifier = get_model(preload, path)

    BM25 = pt.apply.query_vec(pre_process_query, verbose=True) >> pt.BatchRetrieve(index, wmodel="BM25", verbose=True,
                                                                                   metadata=["docno", "text"]) % 10
    pipeline = BM25 >> pt.apply.doc_features(
        compute_cosine_similarity, verbose=True) >> pt.ltr.apply_learned_model(sgd_classifier)

    if not preload:
        print("Fitting model")
        train_topics = dataset.get_topics("dev")
        train_qrels = dataset.get_qrels("dev")
        pipeline.fit(train_topics, train_qrels)

    if not preload and path is not None:
        print("Saving model")
        joblib.dump(sgd_classifier, path)

    print("Starting experiment")
    result = pt.Experiment([BM25, pipeline], test_topics, test_qrels, ["map", "ndcg"], names=["BM25", "Pipeline"])
    return result


if __name__ == '__main__':
    pt.init()
    dataset = pt.get_dataset('msmarco_passage')

    index = pt.IndexFactory.of(
        "E:/Files/uni/in4325/project 1/IN4325-Project-1/src/part2/data/msmarco-passage-index-with-meta")

    res = transform_with_model(preload=False, path='res/model_sgd.pkl')
    # res = transform_with_model(preload=True, path='res/model_sgd.pkl')

    x = 2
