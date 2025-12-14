from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import os

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ======================
# Configuration
# ======================

USE_MOCK_MODEL = os.environ.get("USE_MOCK_MODEL", "false").lower() == "true"

MODEL_NAME = "./model_cache"
ADAPTER_PATH = "../vgcmaster_lora_model"

SYSTEM_PROMPT = (
    "You are a competitive Pokémon VGC analysis model.\n"
    "You must generate analysis strictly from the information provided in the <DATA> block.\n"
    "Do not use outside knowledge or assumptions.\n"
    "If the data is missing, ambiguous, or does not match the requested Pokémon, "
    "respond with a refusal.\n"
    "Follow the requested format exactly."
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

@torch.inference_mode()
def generate(prompt: str, max_new_tokens: int = 256) -> str:
    # Mock mode for testing without model
    if USE_MOCK_MODEL:
        return f"[MOCK RESPONSE] Received prompt about VGC analysis. Prompt length: {len(prompt)} chars"
    
    # Build messages with system prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
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
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

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

    response = generate(prompt)

    return jsonify({
        'response': response,
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
        debug=False,   # IMPORTANT
        threaded=False
    )
