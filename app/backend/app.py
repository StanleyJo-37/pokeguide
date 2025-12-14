from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests


def get_model_response(prompt: str) -> str:
    """
    Placeholder function for the model response.
    This is where you would integrate your actual AI model.
    
    Args:
        prompt: The user's message/question
        
    Returns:
        A response string from the model
    """
    # TODO: Replace with actual model inference
    # This is a placeholder that returns Pokemon-themed responses
    
    prompt_lower = prompt.lower()
    
    if 'pikachu' in prompt_lower:
        return "Pika pika! ⚡ Pikachu is an Electric-type Pokémon and the mascot of the franchise! It evolves from Pichu when leveled up with high friendship and can evolve into Raichu using a Thunder Stone."
    
    elif 'strongest' in prompt_lower or 'powerful' in prompt_lower:
        return "In terms of raw power, Arceus is often considered the strongest Pokémon as the creator of the Pokémon universe. However, Mega Rayquaza, Ultra Necrozma, and Eternamax Eternatus are also incredibly powerful! 🌟"
    
    elif 'starter' in prompt_lower:
        return "Great question about starters! 🔥💧🌿 The classic Kanto starters are Bulbasaur, Charmander, and Squirtle. Each generation introduces new starters - which generation are you curious about?"
    
    elif 'team' in prompt_lower or 'build' in prompt_lower:
        return "Building a balanced team is key! 🎮 Try to cover different types and have a mix of physical/special attackers, a tank, and maybe a support Pokémon. What's your play style - offensive, defensive, or balanced?"
    
    elif 'hello' in prompt_lower or 'hi' in prompt_lower:
        return "Hello, trainer! 👋 Welcome to PokéChat! I'm here to help with anything Pokémon-related. Ask me about battles, team building, Pokémon stats, or just chat about your favorites!"
    
    else:
        return f"That's an interesting question about Pokémon! 🎯 While I'm currently just a placeholder assistant, the real model will be able to give you detailed information about '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'. Stay tuned for the full experience!"


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Chat endpoint that receives a prompt and returns a model response.
    
    Request body:
        {
            "prompt": "User's message",
            "chatId": "Optional chat session ID"
        }
    
    Response:
        {
            "response": "Model's response",
            "chatId": "Chat session ID"
        }
    """
    data = request.get_json()
    
    if not data or 'prompt' not in data:
        return jsonify({'error': 'Missing prompt in request body'}), 400
    
    prompt = data['prompt']
    chat_id = data.get('chatId', 'default')
    
    # Get response from placeholder model
    response = get_model_response(prompt)
    
    return jsonify({
        'response': response,
        'chatId': chat_id
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'service': 'pokechat-backend'})


if __name__ == '__main__':
    print("🎮 PokéChat Backend Starting...")
    print("📡 API available at http://localhost:5000/api/chat")
    app.run(host='0.0.0.0', port=5000, debug=True)
