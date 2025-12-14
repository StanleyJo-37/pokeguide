'use client';

import { useState, useCallback } from 'react';
import ChatSidebar from './components/ChatSidebar';
import ChatArea from './components/ChatArea';
import { sendMessage, generateId, ChatMessage, ChatSession } from './lib/api';

export default function Home() {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  // Get current session's messages
  const activeSession = sessions.find(s => s.id === activeSessionId);
  const messages = activeSession?.messages || [];

  // Create a new chat session
  const handleNewChat = useCallback(() => {
    const newSession: ChatSession = {
      id: generateId(),
      title: 'New Chat',
      messages: [],
      createdAt: new Date(),
    };
    setSessions(prev => [newSession, ...prev]);
    setActiveSessionId(newSession.id);
  }, []);

  // Select a chat session
  const handleSelectSession = useCallback((sessionId: string) => {
    setActiveSessionId(sessionId);
  }, []);

  // Send a message
  const handleSendMessage = useCallback(async (content: string) => {
    // If no active session, create one
    let currentSessionId = activeSessionId;
    if (!currentSessionId) {
      const newSession: ChatSession = {
        id: generateId(),
        title: content.slice(0, 30) + (content.length > 30 ? '...' : ''),
        messages: [],
        createdAt: new Date(),
      };
      setSessions(prev => [newSession, ...prev]);
      setActiveSessionId(newSession.id);
      currentSessionId = newSession.id;
    }

    // Add user message
    const userMessage: ChatMessage = {
      id: generateId(),
      role: 'user',
      content,
      timestamp: new Date(),
    };

    setSessions(prev => prev.map(session => {
      if (session.id === currentSessionId) {
        // Update title if this is the first message
        const newTitle = session.messages.length === 0 
          ? content.slice(0, 30) + (content.length > 30 ? '...' : '')
          : session.title;
        return {
          ...session,
          title: newTitle,
          messages: [...session.messages, userMessage],
        };
      }
      return session;
    }));

    // Send to backend and get response
    setIsLoading(true);
    try {
      const response = await sendMessage(content, currentSessionId);
      
      const assistantMessage: ChatMessage = {
        id: generateId(),
        role: 'assistant',
        content: response.response,
        timestamp: new Date(),
      };

      setSessions(prev => prev.map(session => {
        if (session.id === currentSessionId) {
          return {
            ...session,
            messages: [...session.messages, assistantMessage],
          };
        }
        return session;
      }));
    } catch (error) {
      console.error('Failed to get response:', error);
    } finally {
      setIsLoading(false);
    }
  }, [activeSessionId]);

  return (
    <div className="flex h-screen bg-[var(--background)]">
      {/* Sidebar */}
      <ChatSidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelectSession={handleSelectSession}
        onNewChat={handleNewChat}
      />
      
      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col min-w-0">
        <ChatArea
          messages={messages}
          onSendMessage={handleSendMessage}
          isLoading={isLoading}
        />
      </main>
    </div>
  );
}
