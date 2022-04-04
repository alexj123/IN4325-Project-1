import pandas as pd


def print_worst5():
    ft = pd.read_csv("res/model_fasttext_300d_sgd_bm25_top100_kernels11/per_query.csv", delimiter=";")
    gl = pd.read_csv("res/model_glove_300d_sgd_bm25_top100_kernels11/per_query.csv", delimiter=";")
    l2r = pd.read_csv("res/old_l2r/per_query.csv", delimiter=";")

    # Find top 5 worst performers for MAP metric
    worst5_l2r = l2r[(l2r["measure"] == "map") & (l2r["name"] == "LTR")].sort_values(by="value").head(5)[
        ["qid", "value"]]

    # Find them in the ft and gl
    worst5_gl = gl[gl["qid"].isin(worst5_l2r["qid"]) & (gl["measure"] == "map") & (gl["name"] == "Pipeline")][
        ["qid", "value"]]
    worst5_ft = ft[ft["qid"].isin(worst5_l2r["qid"]) & (ft["measure"] == "map") & (gl["name"] == "Pipeline")][
        ["qid", "value"]]

    worst5_all = pd.concat([worst5_l2r, worst5_ft, worst5_gl], axis=1, keys=["l2r", "ft", "gl"])

    print(worst5_all)


docs = pd.read_csv("data/collections/collection.tsv", delimiter="\t", header=None, names=["docid", "doctext"])
queries = pd.read_csv("data/queries/msmarco-test2019-queries.tsv", delimiter="\t", header=None, names=["qid", "text"])


def write_docs_and_queries_to_file(res_file, qid, output_name):
    res = pd.read_csv(res_file, delim_whitespace=True, header=None, names=["qid", "Q0", "docid", "rank", "score", "name"])

    res_docs_and_scores = res[res["qid"] == qid][["qid", "docid", "rank", "score"]]
    queries_merged = pd.merge(res_docs_and_scores, queries, left_on="qid", right_on="qid")
    docs_merged = pd.merge(queries_merged, docs, left_on="docid", right_on="docid")
    docs_merged.to_csv(f"res/analysis/{output_name}.{qid}.csv")


if __name__ == "__main__":
    write_docs_and_queries_to_file("res/old_l2r/LTR.res", 1106007, "LTR")
    write_docs_and_queries_to_file("res/model_glove_300d_sgd_bm25_top100_kernels11/Pipeline.res", 1106007, "glove")
    write_docs_and_queries_to_file("res/model_fasttext_300d_sgd_bm25_top100_kernels11/Pipeline.res", 1106007, "fasttext")
