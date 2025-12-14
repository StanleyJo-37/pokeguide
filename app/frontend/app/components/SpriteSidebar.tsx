'use client';

import { useEffect, useState } from 'react';

interface SpriteSidebarProps {
  pokemonNames: string[];
}

export default function SpriteSidebar({ pokemonNames }: SpriteSidebarProps) {
  const [spriteUrls, setSpriteUrls] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchSprites = async () => {
      if (!pokemonNames || pokemonNames.length === 0) {
        setSpriteUrls([]);
        return;
      }

      setLoading(true);
      const urls: string[] = [];

      try {
        await Promise.all(
          pokemonNames.map(async (name) => {
            try {
              // Fetch pokemon data to get the distinct ID (needed for the specific sprite URL requested)
              const response = await fetch(`https://pokeapi.co/api/v2/pokemon/${name.toLowerCase()}`);
              if (response.ok) {
                const data = await response.json();
                // Use the specific sprite path requested by the user
                const id = data.id;
                urls.push(`https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/${id}.png`);
              }
            } catch (err) {
              console.error(`Failed to fetch sprite for ${name}`, err);
            }
          })
        );
        // Sort to maintain somewhat consistent order or just set them
        setSpriteUrls(urls);
      } catch (error) {
        console.error('Error fetching sprites:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchSprites();
  }, [pokemonNames]);

  // if (!pokemonNames || pokemonNames.length === 0) return null; // Always show sidebar

  return (
    <div className="w-full h-full bg-[var(--background-secondary)] border-l border-white/10 flex flex-col items-center py-6 gap-2 overflow-y-auto animate-fade-in custom-scrollbar transition-all duration-300">
      <h3 className="text-xs font-bold text-[var(--text-secondary)] uppercase tracking-wider mb-2 text-center">
        Team Sprites
      </h3>
      
      {(!pokemonNames || pokemonNames.length === 0) && !loading && (
        <p className="text-xs text-[var(--text-muted)] text-center px-4">
          No sprites to display
        </p>
      )}

      {loading ? (
        <div className="flex flex-col gap-4">
           {[...Array(3)].map((_, i) => (
             <div key={i} className="w-16 h-16 rounded-full bg-white/5 animate-pulse" />
           ))}
        </div>
      ) : (
        <div className="flex flex-col gap-y-4">
          {spriteUrls.map((url, index) => (
            <div 
              key={index} 
              className="relative w-16 h-16 transition-transform hover:scale-110 cursor-pointer group"
            >
              <div className="absolute inset-0 bg-white/5 rounded-full filter blur-md opacity-0 group-hover:opacity-100 transition-opacity" />
              <img 
                src={url} 
                alt="Pokemon Sprite" 
                className="w-full h-full object-contain relative z-10 pixelated rendering-pixelated"
                onError={(e) => {
                    (e.target as HTMLImageElement).style.display = 'none';
                }}
              />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
