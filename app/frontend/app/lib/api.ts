const API_BASE = "http://localhost:5000";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  pokemonNames?: string[];
}

export interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: Date;
}

export interface ChatResponse {
  response: string;
  chatId: string;
  pokemonNames?: string[];
}

/**
 * Send a message to the backend chat endpoint
 * This is a placeholder that calls the backend model endpoint
 */
export async function sendMessage(
  prompt: string,
  chatId?: string
): Promise<ChatResponse> {
  try {
    const response = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        prompt,
        chatId: chatId || null,
      }),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error("Failed to send message:", error);
    // Return a fallback response for demo purposes
    return {
      response:
        "Pika pika! 🔌 It looks like I'm having trouble connecting to my Poké Center. Please make sure the backend server is running!",
      chatId: chatId || generateId(),
    };
  }
}

/**
 * Generate a unique ID for messages and chats
 */
export function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
}
