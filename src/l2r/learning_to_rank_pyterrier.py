import pandas as pd
import pyterrier as pt
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor

pt.init()
dataset = pt.get_dataset('msmarco_passage')

index = pt.IndexFactory.of("D:/Projects/Uni/IN4325-Project-1/src/l2r/data/msmarco-passage-index")

train_topics = dataset.get_topics("dev.small")
qrels = dataset.get_qrels("dev.small")

test_topics = dataset.get_topics("test-2019")
test_qrels = dataset.get_topics("test-2019")

TF_IDF = pt.BatchRetrieve(index, vmodel="TF_IDF")
BM25 = pt.BatchRetrieve(index, vmodel="BM25")
PL2 = pt.BatchRetrieve(index, vmodel="PL2")

pipeline = pt.FeaturesBatchRetrieve(index, wmodel="BM25", features=["WMODEL:Tf", "WMODEL:PL2"])
rf = RandomForestRegressor(n_estimators=400)
rf_pipe = pipeline >> pt.ltr.apply_learned_model(rf)
rf_pipe.fit(train_topics, qrels)

rf_pipe_fast = rf_pipe.compile()

df: DataFrame = pt.Experiment([BM25, rf_pipe_fast], test_topics, test_qrels, ["map"], names=["BM25 Baseline", "LTR"])
df.to_csv("l2r_res.csv")
