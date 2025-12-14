import json
import numpy as np
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import ollama

# -------------------------------
# 1️⃣ Ollama Embedding Wrapper
# -------------------------------
class OllamaEmbeddings:
    def __init__(self, model="embeddinggemma"):
        self.model = model

    def embed_texts(self, texts):
        """
        Embed a list of texts in batch.
        Returns a list of vectors.
        """
        print(f"Embedding {len(texts)} texts with Ollama...")
        results = ollama.embed(model=self.model, input=texts)
        return [r for r in results.embeddings]

# -------------------------------
# 2️⃣ Load JSON datasets
# -------------------------------
print("Loading JSON datasets...")
with open("../knowledge_base/gen9regh.json", "r", encoding="utf-8") as f:
    gen9_data = json.load(f)

with open("../knowledge_base/leads_statistics.json", "r", encoding="utf-8") as f:
    leads_statistics_data = json.load(f)

with open("../knowledge_base/usage_statistics.json", "r", encoding="utf-8") as f:
    usage_statistics_data = json.load(f)
print("Datasets loaded successfully.")

# -------------------------------
# 3️⃣ Prepare Documents
# -------------------------------
print("Preparing Document objects...")
documents = []

# Gen9 data
for i, data in enumerate(gen9_data):
    doc = Document(
        page_content=data["input"],
        metadata={
            "pokemon": data["pokemon"],
            "output": data["output"],
            "source": "gen9regh.json"
        }
    )
    documents.append(doc)
print(f"Added {len(gen9_data)} Gen9 documents.")

# Leads statistics
for i, data in enumerate(leads_statistics_data):
    doc = Document(
        page_content=data["text"],
        metadata={
            "pokemon": data["metadata"]["pokemon"],
            "rank": data["metadata"]["rank"],
            "topic": data["metadata"]["topic"],
            "date": data["metadata"]["date"],
            "raw_json": data["raw_json"],
            "source": "leads_statistics.json"
        }
    )
    documents.append(doc)
print(f"Added {len(leads_statistics_data)} lead statistics documents.")

# Usage statistics
for i, data in enumerate(usage_statistics_data):
    doc = Document(
        page_content=data["text"],
        metadata={
            "pokemon": data["metadata"]["pokemon"],
            "rank": data["metadata"]["rank"],
            "topic": data["metadata"]["topic"],
            "date": data["metadata"]["date"],
            "raw_json": data["raw_json"],
            "source": "usage_statistics.json"
        }
    )
    documents.append(doc)
print(f"Added {len(usage_statistics_data)} usage statistics documents.")
print(f"Total documents: {len(documents)}")

# -------------------------------
# 4️⃣ Embed all documents in batch
# -------------------------------
embeddings_model = OllamaEmbeddings(model="embeddinggemma")
texts = [doc.page_content for doc in documents]
print("Starting embeddings...")
vectors = embeddings_model.embed_texts(texts)
print("Embeddings completed.")

# -------------------------------
# 5️⃣ Create FAISS index and docstore
# -------------------------------
dim = len(vectors[0])
index = faiss.IndexFlatL2(dim)

# Build docstore dictionary and index_to_docstore_id mapping
print("Building docstore and FAISS index...")
docstore_dict = {}
index_to_docstore_id = {}
for i, doc in enumerate(documents):
    doc_id = f"doc_{i}"
    docstore_dict[doc_id] = doc
    index_to_docstore_id[i] = doc_id

# Add all vectors to FAISS index at once
vectors_array = np.array(vectors).astype('float32')
index.add(vectors_array)

# Create the vector store with pre-built components
vector_store = FAISS(
    embedding_function=None,  # already embedded
    index=index,
    docstore=InMemoryDocstore(docstore_dict),
    index_to_docstore_id=index_to_docstore_id
)
print(f"FAISS index created with dimension {dim} and {len(documents)} documents.")

# -------------------------------
# 7️⃣ Save vector store
# -------------------------------
vector_store.save_local("vector_store")
print("✅ Vector store saved successfully!")
