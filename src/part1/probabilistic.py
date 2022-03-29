import pyterrier as pt

pt.init()
dataset = pt.get_dataset('msmarco_passage')

index = pt.IndexFactory.of("E:/Files/uni/in4325/project 1/IN4325-Project-1/src/data/msmarco_passage-index")


di = index.getDirectIndex()
doi = index.getDocumentIndex()
lex = index.getLexicon()
docid = 6093904  #docids are 0-based
#NB: postings will be null if the document is empty
for posting in di.getPostings(doi.getDocumentEntry(docid)):
    termid = posting.getId()
    lee = lex.getLexiconEntry(termid)
    print("%s with frequency %d" % (lee.getKey(),posting.getFrequency()))

BM25 = pt.BatchRetrieve(index, wmodel="BM25", controls={"c": 0.75, "bm25.k_1": 1.2, "bm25.k_2": 1.2})

x = pt.Experiment(
    [BM25],
    dataset.get_topics("test-2019"),
    dataset.get_qrels("test-2019"),
    eval_metrics=['num_q', 'num_ret', 'num_rel', 'num_rel_ret', 'map', 'ndcg', 'recip_rank', 'iprec_at_recall'],
    # save_dir="./res",
    # save_mode="overwrite",
    perquery=True
)

y = 2
