from pyserini import collection, index

if __name__ == '__main__':
    collection = collection.Collection('JsonCollection', 'data/jsonl/')
    generator = index.Generator('DefaultLuceneDocumentGenerator')

    for (i, fs) in enumerate(collection):
        for (j, doc) in enumerate(fs):
            parsed = generator.create_document(doc)
            docid = parsed.get('id')            # FIELD_ID
            raw = parsed.get('raw')             # FIELD_RAW
            contents = parsed.get('contents')   # FIELD_BODY
            print('{} {} -> {} {}...'.format(i, j, docid, contents.strip().replace('\n', ' ')[:50]))
