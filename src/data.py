"""
Chess PNG dataset loading and tokenization

Pipeline: 
    1. Parse PNG files into lists of moves (e.g. ["e4", "e5", "Nf3", ...])
    2. Build a vocabolary of all unique moves
    3. Encode game as integer sequences
    4. Serve (input, target) pairs for autoregressive training (future steps)
"""

import os
import re

## Parsing PNG - based after first look at original dataset
def parse_png(png_data: str) -> List[List[str]]:
    """ Parse a PNG string into a list of games, each a list of moves."""

    games = []

    # Split into individual game blocks 
    # Each game is separated by a blank line between the moves and the next game's header.
    
    # Normalize line endings
    pgn_text = pgn_text.replace('\r\n', '\n').replace('\r', '\n')

    raw_blocks = re.split(r'\n\n+', pgn_text.strip())

    # Separate each play block
    i=0
    game_blocks = []
    while i < len(raw_blocks):
        block = raw_blocks[i].strip()
        if block.startswith('['):
            if i + 1 < len(raw_blocks):
                moves_block = raw_blocks[i + 1].strip()
                if not moves_block.startswith('['):
                    game_blocks.append(moves_block)
                    i += 2
                    continue
        i+=1

    # Clean each block   
    for move_text in game_blocks:
        move_text = re.sub(r'\{[^}]*\}', ' ', move_text, flags=re.DOTALL)
        move_text = re.sub(r'\([^)]*\)', ' ', move_text)
        move_text = re.sub(r'\$\d+', ' ', move_text)
        move_text = re.sub(r'(1-0|0-1|1/2-1/2|\*)', ' ', move_text)
        move_text = re.sub(r'\d+\.{1,3}', ' ', move_text)
        move_text = move_text.replace('#', '').replace('+', '')
        moves = [m.strip() for m in move_text.split() if m.strip()]

        # Skip very short or empty games
        if len(moves) >= 5:
            games.append(moves)

    return games



if __name__ == 'main':
    pass

