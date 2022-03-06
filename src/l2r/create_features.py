"""
This file was not used in the end result.
"""

import pandas as pd
import pyterrier as pt

pt.init()
dataset = pt.get_dataset('msmarco_passage')

index = pt.IndexFactory.of("D:/Projects/Uni/IN4325-Project-1/src/l2r/data/msmarco-passage-index")

qrels = pd.read_csv('data/qrels.train.tsv', delimiter="\t", names=["qid", "0", "docno", "label"])
qrels.drop(columns=["0"])

queries = pd.read_csv('data/queries.train.tsv', delimiter="\t", names=["qid", "query"])

TF_IDF = pt.BatchRetrieve(index, vmodel="TF_IDF")
BM25 = pt.BatchRetrieve(index, vmodel="BM25")
PL2 = pt.BatchRetrieve(index, vmodel="PL2")

pt.Experiment(
    [TF_IDF, BM25, PL2],
    dataset.get_topics("dev.small"),
    dataset.get_qrels("dev.small"),
    ['map', 'ndcg'])

