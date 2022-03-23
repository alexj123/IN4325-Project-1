import io
import numpy as np
import pyterrier as pt
from sklearn.feature_extraction import DictVectorizer
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.ensemble import RandomForestRegressor


def load_vectors(f_name) -> {}:
    _model = KeyedVectors.load_word2vec_format(f_name, no_header=True)
    return _model


if __name__ == "__main__":
    model = load_vectors("data/glove.6B.50d.txt")

    pt.init()
    dataset = pt.get_dataset('msmarco_passage')

    index = pt.IndexFactory.of("E:/Files/uni/in4325/project 1/IN4325-Project-1/src/data/msmarco_passage-index")

    train_topics = dataset.get_topics("dev.small")
    train_qrels = dataset.get_qrels("dev.small")

    test_topics = dataset.get_topics("test-2019")
    test_qrels = dataset.get_qrels("test-2019")

    TF_IDF = pt.BatchRetrieve(index, vmodel="TF_IDF")
    BM25 = pt.BatchRetrieve(index, vmodel="BM25")

    pipeline = pt.FeaturesBatchRetrieve(index, wmodel="BM25", features=["WMODEL:Tf"], verbose=True)
    rf_pipe = pipeline >> pt.ltr.apply_learned_model(model)
    rf_pipe.fit(test_topics, test_qrels)

    rf_pipe_fast = rf_pipe.compile()

    df = pt.Experiment([BM25, rf_pipe_fast], test_topics, test_qrels, ["map"], names=["BM25 Baseline", "LTR"])
    df.to_csv("l2r_res.csv")

    x = 2