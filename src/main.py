import csv
import math
from pyserini.index import IndexReader
from pyserini.search import SimpleSearcher


def do_bm25():
    searcher = SimpleSearcher("data/indexes")
    searcher.set_bm25(1.2, 0.75)

    all_hits = []
    with open("data/queries/msmarco-test2019-queries.tsv") as query_file:
        queries = csv.reader(query_file, delimiter="\t")
        for (qid, query) in queries:
            hits = searcher.search(query)
            sum_score = 0
            for hit in hits:
                sum_score += hit.score

            # for hit in hits:
            #     y = math.ceil(hit.score / sum_score * 4.0) - 1
            #     hit.score = y
            all_hits.append((qid, hits))
            # if len(all_hits) == 25:
            #     break

    return all_hits


if __name__ == '__main__':

    res = do_bm25()

    with open('data/qrels/2019qrels-pass.txt') as qrel_file:
        qrels = csv.reader(qrel_file, delimiter=" ")

        qrels_dict = {}
        for (qid, q0, docid, rating) in qrels:
            qrels_dict[f"{qid}{docid}"] = rating

    print("rank pos/qid/docid/score_us/score_them")
    for (qid, hits) in res:
        rank = 0
        for hit in hits:
            if f"{qid}{hit.docid}" in qrels_dict:
                print(f"{rank}/{qid}/{hit.docid}/{hit.score}/{qrels_dict[f'{qid}{hit.docid}']}")
            rank += 1
    # query = 'atomic'
    # docids = range(0, 100)
    #
    # for i in range(0, len(docids)):
    #     score = index_reader.compute_query_document_score(str(docids[i]), query)
    #     print(f'{i + 1:2} {docids[i]:15} {score:.5f}')
