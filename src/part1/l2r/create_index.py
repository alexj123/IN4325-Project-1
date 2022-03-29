import pandas as pd
import pyterrier as pt

pt.init()

df = pd.read_csv('data/collection.tsv', delimiter="\t", names=["docno", "text"])
indexer = pt.IterDictIndexer('D:/Projects/Uni/IN4325-Project-1/src/l2r/data/msmarco-passage-index')

index_ref = indexer.index(df.to_dict(orient="records"))
