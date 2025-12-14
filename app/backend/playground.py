import ollama

# Generate embedding for a sample text
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

embeddings = OllamaEmbeddings()
texts = ["Hello world!", "Black bird"]
vectors = embeddings.embed_texts(texts)
print(vectors)
print("Embedding dimension:", len(vectors))
