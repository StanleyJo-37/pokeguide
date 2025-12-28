from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import os
import faiss
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import ollama
import re
import json
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from pydantic import BaseModel, Field
from typing import List
from outlines import Generator, from_transformers

# ======================
# Configuration
# ======================
base_url = "https://pokeapi.co/api/v2/"

class OllamaEmbeddings:
    def __init__(self, model="embeddinggemma"):
        self.model = model

    def embed_documents(self, texts):
        print(f"Embedding {len(texts)} texts with Ollama...")
        results = ollama.embed(model=self.model, input=texts)
        return [r for r in results.embeddings]

    def embed_query(self, text):
        results = ollama.embed(model=self.model, input=[text])
        return results.embeddings[0]

class Pokemon(BaseModel):
    name: str = Field(..., description="The name of the Pokemon")
    ability: str = Field(..., description="The ability of the Pokemon")
    nature: str = Field(..., description="The nature of the Pokemon")
    item: str = Field(..., description="The item of the Pokemon")
    moveset: List[str] = Field(..., description="The 4 moves of the Pokemon")

class PokemonTeam(BaseModel):
    team: List[Pokemon] = Field(..., description="The team of 6 Pokemon")

def get_sprite(pokemon_name: str) -> str:
    sprite_url = f"{base_url}pokemon/{pokemon_name.lower()}"
    response = requests.get(sprite_url)
    data = response.json()
    return data["sprites"]["front_default"]

def is_team_building_request(prompt: str) -> bool:
    """
    Detect if the user is asking to build a team
    """
    team_keywords = [
        'build a team',
        'create a team',
        'make a team',
        'generate a team',
        'suggest a team',
        'team for',
        'build me a team',
        'give me a team',
        'team comp',
        'team building',
        'team suggestion',
    ]
    
    prompt_lower = prompt.lower()
    return any(keyword in prompt_lower for keyword in team_keywords)

# ==========================================

USE_MOCK_MODEL = os.environ.get("USE_MOCK_MODEL", "false").lower() == "true"

MODEL_NAME = "./model_cache"
ADAPTER_PATH = "../vgcmaster_lora_model"

SYSTEM_PROMPT_GENERAL = (
    "You are a competitive Pokémon VGC analysis expert.\n"
    "Answer questions about Pokemon, strategies, movesets, and competitive play.\n"
    "Be concise, accurate, and helpful. Be short in your answer, at most 7 sentences.\n"
    "Don't use complex sentences or long paragraphs.\n"
)

SYSTEM_PROMPT_TEAM = (
    "You are a competitive Pokémon VGC team builder.\n"
    "Generate a team of 6 Pokemon with complete competitive sets.\n"
    "Include: name, ability, nature, held item, and 4 moves for each Pokemon.\n"
)

# ======================
# Load model ONCE
# ======================

if USE_MOCK_MODEL:
    print("🧪 Running in MOCK mode - no model loaded")
    tokenizer = None
    model = None
    outlines_model = None
    generator = None
else:
    print("🔄 Loading model...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    
    # Keep the original PeftModel for text generation
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    
    # ✅ Create structured output generator for team building ONLY
    # Use a separate variable to avoid overwriting the original model
    outlines_model = from_transformers(
        model,
        tokenizer, 
    )
    generator = Generator(outlines_model, PokemonTeam)
    
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


def get_context_from_vector_store(prompt: str) -> str:
    """Retrieve context from vector store"""
    if not vector_store:
        return ""
    
    try:
        query_vector = embeddings.embed_query(prompt)
        D, I = vector_store.index.search(np.array([query_vector], dtype='float32'), k=2)
        
        context_str = ""
        for idx in I[0]:
            if idx in vector_store.index_to_docstore_id:
                doc_id = vector_store.index_to_docstore_id[idx]
                if doc_id in vector_store.docstore._dict:
                    doc = vector_store.docstore.search(doc_id)
                    meta_str = ", ".join([f"{k}: {v}" for k, v in doc.metadata.items()])
                    context_str += f"Content: {doc.page_content}\nSource Info: {meta_str}\n\n"
        
        return context_str
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return ""


@torch.inference_mode()
def generate_team(prompt: str, context: str = "") -> dict:
    """
    Generate structured Pokemon team output using Outlines
    """
    if USE_MOCK_MODEL:
        return {
            "team": [
                {
                    "name": "Pikachu",
                    "ability": "Lightning Rod",
                    "nature": "Timid",
                    "item": "Light Ball",
                    "moveset": ["Thunderbolt", "Volt Switch", "Grass Knot", "Protect"]
                }
            ],
            "sprites": ["https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/25.png"]
        }
    
    # Construct prompt with context
    if context:
        full_prompt = (
            f"{SYSTEM_PROMPT_TEAM}\n\n"
            f"Context information:\n"
            f"---------------------\n"
            f"{context}"
            f"---------------------\n\n"
            f"User request: {prompt}\n\n"
            f"Generate a competitive VGC team in JSON format."
        )
    else:
        full_prompt = f"{SYSTEM_PROMPT_TEAM}\n\nUser request: {prompt}\n\nGenerate a competitive VGC team in JSON format."
    
    try:
        # ✅ Use structured output generator
        result = generator(full_prompt)
        print(f"🔍 Raw generator output type: {type(result)}")
        print(f"🔍 Raw generator output: {result}")
        
        # Outlines may return a string - parse it if needed
        if isinstance(result, str):
            print("📝 Parsing JSON string from generator...")
            result = json.loads(result)
        
        # Validate output with Pydantic
        pokemon_team = PokemonTeam.model_validate(result)
        print(f"✅ Validated team: {pokemon_team}")
        
        # Convert Pydantic model to dict
        team_dict = pokemon_team.model_dump()
        print(f"📦 Team dict: {team_dict}")
        
        # Get sprites for each Pokemon
        sprites = []
        for pokemon in team_dict['team']:
            try:
                sprite = get_sprite(pokemon['name'])
                sprites.append(sprite)
            except Exception as e:
                print(f"Error fetching sprite for {pokemon['name']}: {e}")
                sprites.append(None)
        
        return {
            "type": "team",
            "team": team_dict['team'],
            "sprites": sprites
        }
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        print(f"❌ Raw output was: {result}")
        return {
            "type": "error",
            "error": f"Failed to parse team JSON: {str(e)}",
            "team": [],
            "sprites": []
        }
    except Exception as e:
        print(f"Error during team generation: {e}")
        return {
            "type": "error",
            "error": str(e),
            "team": [],
            "sprites": []
        }


@torch.inference_mode()
def generate_text(prompt: str, context: str = "", max_new_tokens: int = 1024) -> dict:
    """
    Generate regular text response (for analysis, questions, etc.)
    """
    if USE_MOCK_MODEL:
        return {
            "type": "text",
            "response": f"[MOCK] This is a text response about: {prompt}"
        }
    
    # Construct prompt with context
    if context:
        full_user_content = (
            f"Context information is below.\n"
            f"---------------------\n"
            f"{context}"
            f"---------------------\n"
            f"Given the context information, answer the query precisely based on the context information.\n"
            f"Query: {prompt}"
        )
    else:
        full_user_content = prompt

    # Build messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_GENERAL},
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
        temperature=0.1,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Decode only the NEW tokens (skip prompt)
    new_tokens = outputs[0][inputs.shape[-1]:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    response_text = response_text.replace('<OUTPUT>', '')
    response_text = response_text.replace('</OUTPUT>', '')
    return {
        "type": "text",
        "response": response_text.strip()
    }


def generate(prompt: str) -> dict:
    """
    Main generation function that routes to team building or text generation
    Returns both the result and the context used for RAG
    """
    # Get context from vector store
    context = get_context_from_vector_store(prompt)
    print(f"📚 Context retrieved: {len(context)} chars" if context else "📚 No context retrieved")
    
    # Check if user wants a team
    if is_team_building_request(prompt):
        print("🎮 Detected team building request - using structured output")
        result = generate_team(prompt, context)
    else:
        print("💬 General query - using text generation")
        result = generate_text(prompt, context)
    
    # Include context in the result
    result["context"] = context if context else None
    print(f"📤 Sending response with context: {bool(result.get('context'))}")
    return result


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

    result = generate(prompt)

    return jsonify({
        'response': result,
        'chatId': chat_id
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    print("🚀 Backend running on http://localhost:5000")
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=False
    )