import csv

from pyserini.index import IndexReader
from pyserini.search import SimpleSearcher

if __name__ == '__main__':

    searcher = SimpleSearcher("data/indexes")
    searcher.set_bm25(0.9, 0.4)

    with open("data/queries/msmarco-test2019-queries.tsv") as queryFile:
        queries = csv.reader(queryFile, delimiter="\t")
        for (qid, query) in queries:
            hits = searcher.search(query)
            print(hits)

    # query = 'atomic'
    # docids = range(0, 100)
    #
    # for i in range(0, len(docids)):
    #     score = index_reader.compute_query_document_score(str(docids[i]), query)
    #     print(f'{i + 1:2} {docids[i]:15} {score:.5f}')
