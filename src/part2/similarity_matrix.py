import numpy as np
import pyterrier as pt

pt.init()

dataset = pt.get_dataset('msmarco_passage')

index = pt.IndexFactory.of("D:/Projects/Uni/IN4325-Project-1/src/part2/data/msmarco-passage-index-with-meta")

test_topics = dataset.get_topics("test-2019")

pipeline = pt.BatchRetrieve(index, wmodel=None, verbose=True) % 200 

l = pipeline.transform(test_topics)

x = 2