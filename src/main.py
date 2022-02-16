from pyserini import collection, index

if __name__ == '__main__':
    collection = collection.Collection('HtmlCollection', 'collections/cacm/')
    generator = index.Generator('DefaultLuceneDocumentGenerator')
