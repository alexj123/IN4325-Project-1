import pyterrier as pt
import experiment as exp
from src.part1.l2r.learning_to_rank import run_old_l2r

if __name__ == '__main__':
    if not pt.started():
        pt.init()

    # exp.Experiment("model_fasttext_300d_sgd_bm25_top100_kernels11", "data/wiki-news-300d-1M.vec", False)\
    #     .run_experiment()
    run_old_l2r()

