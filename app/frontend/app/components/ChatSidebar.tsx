'use client';

import { ChatSession } from '../lib/api';

interface ChatSidebarProps {
  sessions: ChatSession[];
  activeSessionId: string | null;
  onSelectSession: (sessionId: string) => void;
  onNewChat: () => void;
}

export default function ChatSidebar({ 
  sessions, 
  activeSessionId, 
  onSelectSession, 
  onNewChat 
}: ChatSidebarProps) {
  return (
    <aside className="w-72 h-full flex flex-col glass-panel p-4 gap-3">
      {/* Logo & Title */}
      <div className="flex items-center gap-3 px-2 pb-4 border-b border-white/10">
        <div className="pokeball-icon shrink-0" />
        <h1 className="text-xl font-bold bg-gradient-to-r from-[#E3350D] to-[#FFCB05] bg-clip-text text-transparent">
          PokéChat
        </h1>
      </div>

      {/* New Chat Button */}
      <button
        onClick={onNewChat}
        className="btn-primary flex items-center justify-center gap-2 w-full"
      >
        <svg 
          className="w-5 h-5" 
          fill="none" 
          stroke="currentColor" 
          viewBox="0 0 24 24"
        >
          <path 
            strokeLinecap="round" 
            strokeLinejoin="round" 
            strokeWidth={2} 
            d="M12 4v16m8-8H4" 
          />
        </svg>
        New Chat
      </button>

      {/* Chat History */}
      <div className="flex-1 overflow-y-auto mt-2">
        <p className="text-xs uppercase tracking-wider text-[var(--text-muted)] px-2 mb-2">
          Chat History
        </p>
        
        {sessions.length === 0 ? (
          <p className="text-sm text-[var(--text-muted)] px-2 py-4 text-center">
            No chats yet. Start a new conversation!
          </p>
        ) : (
          <ul className="flex flex-col gap-1">
            {sessions.map((session) => (
              <li key={session.id}>
                <button
                  onClick={() => onSelectSession(session.id)}
                  className={`sidebar-item w-full text-left truncate ${
                    activeSessionId === session.id ? 'active' : ''
                  }`}
                >
                  <span className="block truncate text-sm">
                    {session.title || 'New Chat'}
                  </span>
                  <span className="block text-xs text-[var(--text-muted)] mt-0.5">
                    {session.messages.length} message{session.messages.length !== 1 ? 's' : ''}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Footer */}
      <div className="pt-3 border-t border-white/10 text-center">
        <p className="text-xs text-[var(--text-muted)]">
          Powered by Pokémon AI ⚡
        </p>
      </div>
    </aside>
  );
}
