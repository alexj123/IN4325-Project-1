import datetime
import os

import pyterrier as pt
from os import path as pat

from sentence_transformers import SentenceTransformer
from src.part2.contextual_embeddings.context_cache import ContextCache

DISTILBERT = "sentence-transformers/distiluse-base-multilingual-cased-v2"
ALBERT = "sentence-transformers/paraphrase-albert-small-v2"


class Experiment:
    def __init__(self, model_name):
        print(f"retrieving model {model_name}")
        self.model = SentenceTransformer(model_name)
        print("retrieving dataset msmarco_passage")
        self.dataset = pt.get_dataset('msmarco_passage')
        print("retrieving index")
        self.index = pt.IndexFactory.of(
            "D:\\Projects\\Uni\\IN4325-Project-1\\src\\part2\\data\\msmarco-passage-index-with-meta")
        print("initializing context cache")
        self.doc_cache = ContextCache(model_name)
        print("experiment initialized")

    def pre_process_query(self, q):
        return self.model.encode([q[1].lower()], convert_to_tensor=True)

    def transform_with_model(self, perquery=False, save_dir=None):
        test_topics = self.dataset.get_topics("test-2019")
        test_qrels = self.dataset.get_qrels("test-2019")

        # for testing single query
        # test_topics = test_topics[(test_topics["qid"] == "1121709")]

        bm25 = pt.apply.query_vec(self.pre_process_query, verbose=True) >> \
               pt.BatchRetrieve(self.index, wmodel="BM25", verbose=True, metadata=["docno", "text"])

        # pipeline = ~bm25 >> pt.apply.doc_score(self.doc_cache.compute_cosine_similarity, verbose=True) \

        pipeline = ~bm25 % 100 >> pt.apply.doc_score(self.doc_cache.compute_cosine_similarity, verbose=True) \

        print("starting experiment")
        if save_dir is None:
            return pt.Experiment(
                [pipeline],
                test_topics,
                test_qrels,
                ["map", "ndcg", "P_10"],
                names=["Pipeline"],
                perquery=perquery
            )
        if not pat.exists(save_dir):
            os.mkdir(save_dir)

        return pt.Experiment(
            [pipeline],
            test_topics,
            test_qrels,
            ["map", "ndcg", "P_10"],
            names=["Pipeline"],
            perquery=perquery,
            save_dir=save_dir,
            save_mode="overwrite"
        )

    def run_experiment(self, run_name):
        res = self.transform_with_model(
            perquery=True,
            save_dir=f"../res/{run_name}"
        )

        res.to_csv(f"../res/{run_name}/per_query.csv")


if __name__ == "__main__":
    pt.init()
    Experiment(DISTILBERT).run_experiment(f"distilbert-1121709")
    Experiment(ALBERT).run_experiment(f"albert-1121709")
