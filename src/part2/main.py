import pyterrier as pt
from sklearn.linear_model import SGDRegressor
import joblib
from k_nrm import pre_process_query, compute_cosine_similarity
from os import path as pat


def get_model(preload, path=None):
    if not preload:
        print("New model instantiated")
        return SGDRegressor(loss="squared_error", penalty="l2")
    else:
        if path is None:
            raise ValueError("Path is required if preload is true")
        print("Model loaded")
        return joblib.load(path)


def transform_with_model(path, preload=False):
    test_topics = dataset.get_topics("test-2019")
    test_qrels = dataset.get_qrels("test-2019")

    if pat.exists(path) and preload is False:
        raise ValueError("Path already exists, delete the existing model or change the path")

    sgd_classifier = get_model(preload, path)

    BM25 = pt.apply.query_vec(pre_process_query, verbose=True) \
           >> pt.BatchRetrieve(index, wmodel="BM25", verbose=True, metadata=["docno", "text"]) % 100
    pipeline = BM25 \
               >> pt.apply.doc_features(compute_cosine_similarity, verbose=True) \
               >> pt.apply.text(drop=True) \
               >> pt.ltr.apply_learned_model(sgd_classifier)

    if not preload:
        print("Fitting model")
        train_topics = dataset.get_topics("dev.small")
        train_qrels = dataset.get_qrels("dev.small")
        pipeline.fit(train_topics, train_qrels)

        print("Saving model")
        joblib.dump(sgd_classifier, path)

    print("Starting experiment")
    result = pt.Experiment([BM25, pipeline], test_topics, test_qrels, ["map", "ndcg", "P_10"], names=["BM25", "Pipeline"])
    return result


if __name__ == '__main__':
    pt.init()
    dataset = pt.get_dataset('msmarco_passage')

    index = pt.IndexFactory.of(
        "E:/Files/uni/in4325/project 1/IN4325-Project-1/src/part2/data/msmarco-passage-index-with-meta")

    res = transform_with_model(preload=False, path='res/model_glove_300d_sgd_bm25_top100_3.pkl')
    # res = transform_with_model(preload=True, path='res/model_fasttext_300d_sgd_bm25_top100.pkl')

    x = 2
