''' Retrieval Models 
● Implement the Boolean retrieval model and process queries. 
● Implement the vector space model with TF-IDF weighting and cosine similarity. ''' 

import re
from collections import defaultdict

class BooleanRetrieval:
    def __init__(self):
        self.index = defaultdict(set)
        self.documents = {}

    def add_document(self, doc_id, text):
        self.documents[doc_id] = text
        words = self.tokenize(text)
        for word in words:
            self.index[word].add(doc_id)

    def tokenize(self, text):
        return re.findall(r'\b\w+\b', text.lower())

    def parse_query(self, query):
        return re.findall(r'\b\w+\b|AND|OR|NOT', query.upper())

    def process_query(self, query):
        tokens = self.parse_query(query)
        result = None
        current_op = None

        for token in tokens:
            if token in {'AND', 'OR', 'NOT'}:
                current_op = token
            else:
                docs = self.index.get(token.lower(), set())
                if current_op == 'NOT':
                    docs = set(self.documents.keys()) - docs

                if result is None:
                    result = docs
                elif current_op == 'AND':
                    result &= docs
                elif current_op == 'OR':
                    result |= docs

                current_op = None
        return result or set()

# Example
if __name__ == "__main__":
    br = BooleanRetrieval()
    docs = {
        1: "Apple banana orange",
        2: "Banana apple pie",
        3: "Fruit salad with banana and kiwi",
    }
    for doc_id, content in docs.items():
        br.add_document(doc_id, content)

    while True:
        q = input("\nBoolean Query (e.g., apple AND banana OR kiwi, type 'exit' to stop): ")
        if q.lower() == "exit":
            break
        results = br.process_query(q)
        if results:
            print(f"Documents: {sorted(results)}")
            for doc_id in results:
                print(f"{doc_id}: {br.documents[doc_id]}")
        else:
            print("No match.")


''' Vector Space Retrieval '''
import math
from collections import defaultdict, Counter

class VectorSpaceRetrieval:
    def __init__(self):
        self.documents = {}
        self.tf = {}
        self.df = defaultdict(int)
        self.N = 0
        self.vocab = set()

    def add_document(self, doc_id, text):
        tokens = self.tokenize(text)
        self.documents[doc_id] = tokens
        self.N += 1
        term_count = Counter(tokens)
        self.tf[doc_id] = term_count
        for term in term_count:
            self.df[term] += 1
            self.vocab.add(term)

    def tokenize(self, text):
        return re.findall(r'\b\w+\b', text.lower())

    def compute_tfidf(self, tokens):
        tfidf = {}
        term_count = Counter(tokens)
        for term in self.vocab:
            tf = term_count[term]
            df = self.df.get(term, 1)
            idf = math.log((self.N + 1) / df)
            tfidf[term] = tf * idf
        return tfidf

    def cosine_similarity(self, vec1, vec2):
        dot = sum(vec1[t] * vec2[t] for t in self.vocab)
        mag1 = math.sqrt(sum(val ** 2 for val in vec1.values()))
        mag2 = math.sqrt(sum(val ** 2 for val in vec2.values()))
        if mag1 == 0 or mag2 == 0:
            return 0
        return dot / (mag1 * mag2)

    def search(self, query):
        query_tokens = self.tokenize(query)
        query_vec = self.compute_tfidf(query_tokens)
        scores = {}
        for doc_id, tokens in self.documents.items():
            doc_vec = self.compute_tfidf(tokens)
            scores[doc_id] = self.cosine_similarity(query_vec, doc_vec)

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# Example
if __name__ == "__main__":
    vs = VectorSpaceRetrieval()
    docs = {
        1: "The cat sat on the mat",
        2: "Dogs and cats are friends",
        3: "The dog chased the cat",
    }

    for doc_id, content in docs.items():
        vs.add_document(doc_id, content)

    while True:
        q = input("\nSearch query (type 'exit' to stop): ")
        if q.lower() == "exit":
            break
        results = vs.search(q)
        for doc_id, score in results:
            if score > 0:
                print(f"{doc_id}: Score = {score:.4f} → {docs[doc_id]}")
