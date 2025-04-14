import re
from collections import defaultdict

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.documents = dict()

    def add_document(self, doc_id, text):
        self.documents[doc_id] = text
        tokens = self.tokenize(text)
        for token in tokens:
            self.index[token].add(doc_id)

    def tokenize(self, text):
        # Lowercase and split words, remove punctuation
        return re.findall(r'\b\w+\b', text.lower())

    def search(self, query):
        tokens = self.tokenize(query)
        if not tokens:
            return set()
        
        # Find documents that contain all query tokens
        result_docs = self.index.get(tokens[0], set()).copy()
        for token in tokens[1:]:
            result_docs &= self.index.get(token, set())
        return result_docs

    def print_index(self):
        for term, doc_ids in sorted(self.index.items()):
            print(f"{term}: {sorted(doc_ids)}")


# Example usage
if __name__ == "__main__":
    docs = {
        1: "The quick brown fox jumps over the lazy dog.",
        2: "Never jump over the lazy dog quickly.",
        3: "A fast brown fox leaped over sleeping dogs."
    }

    ii = InvertedIndex()
    
    for doc_id, content in docs.items():
        ii.add_document(doc_id, content)

    print("\n--- Inverted Index ---")
    ii.print_index()

    while True:
        query = input("\nEnter search query: ")
        if query.lower() == "exit":
            break
        results = ii.search(query)
        if results:
            print(f"Documents matching '{query}': {sorted(results)}")
            for doc_id in sorted(results):
                print(f"Doc {doc_id}: {ii.documents[doc_id]}")
        else:
            print("No matching documents found.")
