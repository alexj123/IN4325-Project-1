import csv
from pyserini.search import SimpleSearcher


def get_int(val):
    if val > 0.95:
        return 3
    if val > 0.85:
        return 2
    if val > 0.7:
        return 1
    return 0


def do_bm25():
    searcher = SimpleSearcher("data/indexes")
    searcher.set_bm25(1.2, 0.75)

    all_hits = []
    with open("data/queries/msmarco-test2019-queries.tsv") as query_file:
        queries = csv.reader(query_file, delimiter="\t")
        for (qid, query) in queries:
            hits = searcher.search(query)
            max_score = hits[0].score
            for hit in hits:
                y = hit.score / max_score
                all_hits.append([qid, "Q0", hit.docid, get_int(y), hit.score, "our_id"])

    return all_hits


if __name__ == '__main__':

    res = do_bm25()

    with open('data/qrels/2019qrels-pass.txt') as qrel_file:
        qrels = csv.reader(qrel_file, delimiter=" ")

        qrels_dict = {}
        for (qid, q0, docid, rating) in qrels:
            qrels_dict[f"{qid}{docid}"] = rating

    with open("qrels.txt", "w", newline='') as qrel_res:
        write = csv.writer(qrel_res, delimiter=' ')
        for result_row in res:
            write.writerow(result_row)
