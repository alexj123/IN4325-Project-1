import pandas as pd
import pyterrier as pt

pt.init()

df = pd.read_csv('data/collection.tsv', delimiter="\t", names=["docno", "text"])
print("Starting indexing")
indexer = pt.IterDictIndexer(
    'E:/Files/uni/in4325/project 1/IN4325-Project-1/part2/data/msmarco-passage-index-with-meta',
    meta=['docno', 'text'], meta_lengths=[20, 4096])

index_ref = indexer.index(df.to_dict(orient="records"))
