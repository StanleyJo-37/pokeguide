import json
import numpy as np
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import ollama


class OllamaEmbeddings:
    def __init__(self, model="embeddinggemma"):
        self.model = model

    def embed_documents(self, texts):
        """
        Embed a list of texts in batch.
        Returns a list of vectors.
        """
        print(f"Embedding {len(texts)} texts with Ollama...")
        results = ollama.embed(model=self.model, input=texts)
        return [r for r in results.embeddings]

    def embed_query(self, text):
        """
        Embed a single query text.
        Returns a single vector.
        """
        results = ollama.embed(model=self.model, input=[text])
        return results.embeddings[0]


vector_store = FAISS.load_local(
    "vector_store", 
    embeddings=OllamaEmbeddings(model="embeddinggemma"),
    allow_dangerous_deserialization=True
)

# Example query
user_query = "Build a team around incineroar"

# Step 1: embed the query
query_vector = OllamaEmbeddings(model="embeddinggemma").embed_query(user_query)

# Step 2: search FAISS index
D, I = vector_store.index.search(np.array([query_vector], dtype='float32'), k=5)  # top 5 matches

# Step 3: retrieve the Documents
for idx in I[0]:
    doc_id = vector_store.index_to_docstore_id[idx]
    doc = vector_store.docstore.search(doc_id)
    print("----")
    print("Page content:", doc.page_content)
    print("Metadata:", doc.metadata)
    
