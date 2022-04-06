import pandas as pd

docs = pd.read_csv("../data/collections/collection.tsv", delimiter="\t", header=None, names=["docid", "doctext"])
doc_ids = pd.read_csv("../res/albert-1121709/Pipeline.res", delimiter="\t")

doc_ids_list = doc_ids[["Column3", "Column4"]]

res = pd.merge(doc_ids_list, docs, left_on="Column3", right_on="docid")
res.to_csv("../res/albert-1121709/merged_docs.csv")
