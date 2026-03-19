"""
Chess PGN dataset loading and tokenization

Pipeline:
    1. Parse PGN files into lists of moves (e.g. ["e4", "e5", "Nf3", ...])
    2. Build a vocabulary of all unique moves
    3. Encode game as integer sequences
    4. Serve (input, target) pairs for autoregressive training (future steps)
"""

import re
import json
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset

### Special Tokens
PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"   # beginning of game
EOS_TOKEN = "<EOS>"   # end of game
UNK_TOKEN = "<UNK>"

SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]



## Parsing PGN - based after first look at original dataset
def parse_pgn(pgn_text: str) -> List[List[str]]:
    """ Parse a PGN string into a list of games, each a list of moves."""

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

## Tokenizer 

class ChessTokenizer: 
    """
    Simple vocabulary-based tokenizer for chess moves.

    Moves are already discrete tokens (e.g. "e4", "Nf3", "O-O"),
    so we just need a string -> int mapping.
    Vocabulary size is typically ~2000 for standard chess.
    """

    def __init__(self):
        self.token2id = {}
        self.id2token = {}
        self._add_special_tokens()

    def _add_special_tokens(self):
        for token in SPECIAL_TOKENS:
            self._add_token(token)

    def _add_token(self, token:str) -> int:
        if token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token
        return self.token2id[token]
    
    def build_from_games(self, games: List[list[str]]):
        """Build the vocabulary from a list of tokenized games"""
        for game in games:
            for move in game:
                self._add_token(move)
        print(f"Vocabulary size: {len(self.token2id)} tokens")

    def encode(self, moves: List[str], add_special: bool = True) -> List[int]:
        """Convert moves list to integer ids."""
        ids = [self.token2id.get(m, self.token2id[UNK_TOKEN]) for m in moves]
        if add_special:
            ids = [self.token2id[BOS_TOKEN]] + ids + [self.token2id[EOS_TOKEN]]
        return ids

    def decode(self, ids: List[int]) -> List[str]:
        """Convert integer ids to moves."""
        return [self.id2token.get(i, UNK_TOKEN) for i in ids]

    @property  
    def vocab_size(self) -> int:
        return len(self.token2id)

    @property
    def pad_id(self) -> int:
        return self.token2id[PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.token2id[BOS_TOKEN]
    
    @property
    def eos_id(self) -> int:
        return self.token2id[EOS_TOKEN]
    
    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"token2id": self.token2id, "id2token": {int(k): v for k, v in self.id2token.items()}}, f)


    @classmethod
    def load(cls, path: str) -> "ChessTokenizer":
        tok = cls.__new__(cls)
        tok.token2id = {}
        tok.id2token = {}
        with open(path) as f:
            data = json.load(f)
        tok.token2id = data["token2id"]
        tok.id2token = {int(k): v for k, v in data["id2token"].items()}
        return tok
    

## Chess Dataset class
class ChessDataset(Dataset):
    """
    Autoregressive dataset: given moves 0..T-1, predict moves 1..T.

    Each sample is a (input_ids, target_ids) pair of length `seq_len`.
    Games shorter than seq_len are padded; longer games are chunked.
    """

    def __init__(self, games: List[List[int]], seq_len: int=128, pad_id: int=0):
        self.seq_len = seq_len
        self.pad_id = pad_id
        self.samples = self._build_samples(games)

    def _build_samples(self, games: List[List[int]]) -> List[Tuple[List[int], List[int]]]:
        samples = []
        for game in games:
            # Chunk long games into overlapping windows
            for start in range(0, max(1, len(game) - 1), self.seq_len // 2):
                chunk = game[start: start + self.seq_len + 1]
                if len(chunk) < 3:
                    continue
                # Pad if needed
                padded = chunk + [self.pad_id] * (self.seq_len + 1 - len(chunk))
                inp = padded[:self.seq_len]
                tgt = padded[1: self.seq_len + 1]
                # Replace padding in targets with -1 (ignored by cross_entropy)
                tgt = [t if t != self.pad_id else -1 for t in tgt]
                samples.append((inp, tgt))
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        inp, tgt = self.samples[idx]
        return (torch.tensor(inp, dtype=torch.long), torch.tensor(tgt, dtype=torch.long))



### Full pipeline
def load_data(
    pgn_path: str,
    seq_len: int = 128,
    max_games: Optional[int] = None,
    max_bytes: Optional[int] = None,
    train_split: float = 0.9,
):
    with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read(max_bytes) if max_bytes else f.read()

    # Trim to last complete game
    last_newline = text.rfind('\n\n')
    if last_newline != -1:
        text = text[:last_newline]

    print("Parsing PGN...")
    games = parse_pgn(text)
    print(f"Parsed {len(games)} games")

    if max_games:
        games = games[:max_games]

    # Split at the game level — before building overlapping windows.
    split = int(len(games) * train_split)
    train_games = games[:split]
    val_games   = games[split:]

    # Build vocabulary from training games only
    tokenizer = ChessTokenizer()
    tokenizer.build_from_games(train_games)

    train_encoded = [tokenizer.encode(g) for g in train_games]
    val_encoded   = [tokenizer.encode(g) for g in val_games]

    train_dataset = ChessDataset(train_encoded, seq_len=seq_len, pad_id=tokenizer.pad_id)
    val_dataset   = ChessDataset(val_encoded,   seq_len=seq_len, pad_id=tokenizer.pad_id)

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    return train_dataset, val_dataset, tokenizer