from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import os
import faiss
import numpy as np # Ensure numpy is imported
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import ollama
import numpy as np
import re
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ======================
# Configuration
# ======================

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

USE_MOCK_MODEL = os.environ.get("USE_MOCK_MODEL", "false").lower() == "true"

MODEL_NAME = "./model_cache"
ADAPTER_PATH = "../vgcmaster_lora_model"

SYSTEM_PROMPT = (
    "You are a competitive Pokémon VGC analysis model.\n"
    "You must generate analysis strictly from the information provided in the <DATA> block.\n"
    "Do not use outside knowledge or assumptions.\n"
    "If the data is missing, ambiguous, or does not match the requested Pokémon, respond with a refusal.\n"
    "\n"
    "**FORMATTING RULES:**\n"
    "1. Use **Markdown** for all responses.\n"
    "2. If building a team, format it as:\n"
    "   - **Team Summary**: A brief overview of the archetype.\n"
    "   - **Team Members**: JSON-like exportable text or bullet points for each Pokémon.\n"
    "   - **Strategy**: How to pilot the team.\n"
    "3. **NEVER** use XML tags like <OUTPUT> or </OUTPUT>. output raw text only.\n"
    "4. At the very end of your response, output a JSON list of the Pokémon names mentioned in your team builder, in lowercase.\n"
    "   Format: ::POKEMON_LIST::[\"pikachu\", \"charizard\", ...]"
)

# ======================
# Load model ONCE
# ======================

if USE_MOCK_MODEL:
    print("🧪 Running in MOCK mode - no model loaded")
    tokenizer = None
    model = None
else:
    print("🔄 Loading model...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
    
    print("✅ Model loaded and ready!")


# Global embeddings instance
embeddings = OllamaEmbeddings(model="embeddinggemma")

try:
    vector_store = FAISS.load_local(
        "vector_store", 
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    print("✅ Vector store loaded successfully")
except Exception as e:
    print(f"⚠️ Warning: Could not load vector store: {e}")
    vector_store = None


@torch.inference_mode()
def generate(prompt: str, max_new_tokens: int = 1024) -> tuple[str, list[str]]:
    # Mock mode for testing without model
    if USE_MOCK_MODEL:
        return f"[MOCK RESPONSE] Received prompt about VGC analysis. Prompt length: {len(prompt)} chars", []
    
    context_str = ""
    if vector_store:
        try:
            # Generate query embedding
            query_vector = embeddings.embed_query(prompt)
            
            # Search FAISS
            D, I = vector_store.index.search(np.array([query_vector], dtype='float32'), k=5)
            
            # Retrieve documents
            for idx in I[0]:
                if idx in vector_store.index_to_docstore_id:
                    doc_id = vector_store.index_to_docstore_id[idx]
                    if doc_id in vector_store.docstore._dict:
                        doc = vector_store.docstore.search(doc_id)
                        
                        # Format metadata safely
                        meta_str = ", ".join([f"{k}: {v}" for k, v in doc.metadata.items()])
                        context_str += f"Content: {doc.page_content}\nSource Info: {meta_str}\n\n"
        except Exception as e:
            print(f"Error during retrieval: {e}")
            
    # Construct the final prompt with context
    if context_str:
        full_user_content = (
            f"Context information is below.\n"
            f"---------------------\n"
            f"{context_str}"
            f"---------------------\n"
            f"Given the context information and no prior knowledge, answer the query.\n"
            f"Query: {prompt}"
        )
    else:
        full_user_content = prompt

    # Build messages with system prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": full_user_content}
    ]
    
    # Apply chat template
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    
    # Generate response
    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.3,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Decode only the NEW tokens (skip prompt)
    new_tokens = outputs[0][inputs.shape[-1]:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # Post-processing: remove <OUTPUT> tags if they persist
    response_text = response_text.replace("<OUTPUT>", "").replace("</OUTPUT>", "")
    
    # Extract Pokemon List
    pokemon_names = []
    match = re.search(r"::POKEMON_LIST::(\[.*?\])", response_text, re.DOTALL)
    if match:
        try:
            json_str = match.group(1)
            pokemon_names = json.loads(json_str)
            # Remove the list from the visible text
            response_text = response_text.replace(match.group(0), "").strip()
        except:
            print("Failed to parse Pokemon list JSON")

    return response_text, pokemon_names

# ======================
# Flask app
# ======================

app = Flask(__name__)
CORS(app)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()

    if not data or 'prompt' not in data:
        return jsonify({'error': 'Missing prompt'}), 400

    prompt = data['prompt']
    chat_id = data.get('chatId', 'default')

    response_text, pokemon_names = generate(prompt)

    return jsonify({
        'response': response_text,
        'chatId': chat_id,
        'pokemonNames': pokemon_names
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    print("🚀 Backend running on http://localhost:5000")
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,   # IMPORTANT
        threaded=False
    )
