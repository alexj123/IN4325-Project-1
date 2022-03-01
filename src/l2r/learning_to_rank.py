import pandas as pd
import pyterrier as pt
import tensorflow as tf

import tensorflow_ranking as tfr


def load_data():
    qrels_raw = pd.read_csv('data/qrels.train.tsv', delimiter="\t", names=[0, 1, 2, 3])
    qrels = tf.data.Dataset.from_tensor_slices(dict(qrels_raw))

    documents_raw = pd.read_csv('data/collection.tsv', delimiter="\t", names=[0, 1])
    documents = tf.data.Dataset.from_tensor_slices(dict(documents_raw))

    queries_raw = pd.read_csv('data/queries.train.tsv', delimiter="\t", names=[0, 1])
    queries = tf.data.Dataset.from_tensor_slices(dict(queries_raw))

    qrels.map(lambda x: {
        "query_id": x[0],
        "document_id": x[2],
        "relevant": x[3]
    })

    documents = documents.map(lambda x: x[1])
    queries = queries.map(lambda x: x[1])

    queries_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
    queries_vocabulary.adapt(queries.batch(1000))

    documents_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
    documents_vocabulary.adapt(documents.batch(1000))

    key_func = lambda x: queries_vocabulary(x["query_id"])
    reduce_func = lambda key, dataset: dataset.batch(100)
    ds_train = qrels.group_by_window(key_func=key_func, reduce_func=reduce_func, window_size=100)

    for x in ds_train.take(1):
        for key, value in x.items():
            print(f"Shape of {key}: {value.shape}")
            print(f"Example values of {key}: {value[:5].numpy()}")
            print()


if __name__ == "__main__":
    pt.init()
    load_data()
