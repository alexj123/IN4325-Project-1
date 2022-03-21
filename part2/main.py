import io
import numpy as np
import pyterrier as pt
from sklearn.feature_extraction import DictVectorizer


def load_vectors(fname) -> {}:
    embeddings_dict = {}
    with open(fname, 'r', encoding="utf-8") as f:
        for line in f:
            vals = line.split()
            embeddings_dict[vals[0]] = np.array(vals[1:])
    return embeddings_dict


if __name__ == "__main__":
    vecs = load_vectors("data/glove.6B.50d.txt")
    vec = DictVectorizer()

    y = vec.fit_transform(X=vecs).toarray()

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
    rf_pipe = pipeline >> pt.ltr.apply_learned_model(y)
    rf_pipe.fit(test_topics, test_qrels)

    rf_pipe_fast = rf_pipe.compile()

    df = pt.Experiment([BM25, rf_pipe_fast], test_topics, test_qrels, ["map"], names=["BM25 Baseline", "LTR"])
    df.to_csv("l2r_res.csv")

    x = 2