import os

import pyterrier as pt
from sklearn.linear_model import SGDRegressor
import joblib
import k_nrm
from os import path as pat


class Experiment:
    @staticmethod
    def get_model(exists, path=None):
        if not exists:
            print("New model instantiated")
            return SGDRegressor(loss="squared_error", penalty="l2")
        else:
            if path is None:
                raise ValueError("Path is required if preload is true")
            print("Model loaded")
            return joblib.load(path)

    def transform_with_model(self, path, perquery=False, save_dir=None):
        test_topics = self.dataset.get_topics("test-2019")
        test_qrels = self.dataset.get_qrels("test-2019")

        exists = pat.exists(path)
        sgd_classifier = self.get_model(exists, path)

        bm25 = pt.apply.query_vec(self.k_nrm.pre_process_query, verbose=True) >> \
               pt.BatchRetrieve(self.index, wmodel="BM25", verbose=True, metadata=["docno", "text"])

        knmr = ~bm25 % 100 >> pt.apply.doc_features(self.k_nrm.compute_cosine_similarity, verbose=True) \
               >> pt.apply.text(drop=True) \
               >> pt.ltr.apply_learned_model(sgd_classifier)

        if not exists:
            print("Fitting model")
            train_topics = self.dataset.get_topics("dev.small")
            train_qrels = self.dataset.get_qrels("dev.small")
            knmr.fit(train_topics, train_qrels)

            print("Saving model")
            joblib.dump(sgd_classifier, path)

        print("Starting experiment")
        if save_dir is None:
            return pt.Experiment(
                [~bm25, ~knmr],
                test_topics,
                test_qrels,
                ["map", "ndcg", "P_10"],
                names=["BM25", "Pipeline"],
                perquery=perquery
            )
        if not pat.exists(save_dir):
            os.mkdir(save_dir)

        return pt.Experiment(
            [~bm25, ~knmr],
            test_topics,
            test_qrels,
            ["map", "ndcg", "P_10"],
            names=["BM25", "Pipeline"],
            perquery=perquery,
            save_dir=save_dir,
            save_mode="overwrite"
        )

    def run_experiment(self):
        res = self.transform_with_model(
            path=f'res/models/{self.model_name}.pkl',
            perquery=True,
            save_dir=f"res/{self.model_name}"
        )

        res.to_csv(f"res/{self.model_name}/per_query.csv")

    def __init__(self, model_name, vec_path, no_header):
        self.model_name = model_name
        self.dataset = pt.get_dataset('msmarco_passage')
        self.index = pt.IndexFactory.of(
            "D:\\Projects\\Uni\\IN4325-Project-1\\src\\part2\\data\\msmarco-passage-index-with-meta")
        self.k_nrm = k_nrm.KNRM(vec_path, no_header)
