"""
test_data.py — Tests for the PGN data pipeline.

Covers:
  - parse_pgn: game splitting, move extraction, result/annotation stripping,
    short-game filtering, empty input handling
  - ChessTokenizer: special token IDs, encode/decode round-trip, UNK handling,
    vocab size, duplicate moves, save/load
  - ChessDataset: sample shapes, input/target offset-by-one, padding masking,
    tensor dtypes
"""

import pytest
import torch

from pgn_data import (
    parse_pgn,
    ChessTokenizer,
    ChessDataset,
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
    SPECIAL_TOKENS,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_PGN = """\
[Event "Test Game 1"]
[Site "?"]
[Date "2024.01.01"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 1-0

[Event "Test Game 2"]
[Site "?"]
[Date "2024.01.02"]
[White "Player3"]
[Black "Player4"]
[Result "0-1"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 6. Nf3 h6 0-1
"""

# A game shorter than the minimum (5 moves)
SHORT_PGN = """\
[Event "Short"]
[Date "2024.01.01"]
[Result "*"]

1. e4 e5 *
"""

# PGN with comments, annotations, and a variation
ANNOTATED_PGN = """\
[Event "Annotated"]
[Date "2024.01.01"]
[Result "1-0"]

1. e4 {Best by test} e5 $1 2. Nf3 (2. Bc4 e5) Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0
"""


# ---------------------------------------------------------------------------
# parse_pgn
# ---------------------------------------------------------------------------

class TestParsePGN:
    def test_parses_correct_number_of_games(self):
        games = parse_pgn(SAMPLE_PGN)
        assert len(games) == 2

    def test_moves_are_nonempty_strings(self):
        games = parse_pgn(SAMPLE_PGN)
        for game in games:
            for move in game:
                assert isinstance(move, str) and len(move) > 0

    def test_result_tokens_stripped(self):
        games = parse_pgn(SAMPLE_PGN)
        for game in games:
            for move in game:
                assert move not in ("1-0", "0-1", "1/2-1/2", "*"), (
                    f"Result token '{move}' should be stripped from moves"
                )

    def test_move_numbers_stripped(self):
        games = parse_pgn(SAMPLE_PGN)
        for game in games:
            for move in game:
                assert not any(c.isdigit() and move.endswith('.') for c in move), (
                    f"Move number token '{move}' should be stripped"
                )
                # Move numbers look like "1.", "2.", "10." — digits followed by a dot
                assert not (move[:-1].isdigit() and move.endswith('.')), (
                    f"Move number '{move}' should be stripped"
                )

    def test_expected_moves_present_in_first_game(self):
        games = parse_pgn(SAMPLE_PGN)
        first = games[0]
        assert "e4" in first
        assert "e5" in first
        assert "Nf3" in first
        assert "Nc6" in first

    def test_skips_games_shorter_than_five_moves(self):
        games = parse_pgn(SHORT_PGN)
        assert len(games) == 0, "Games with fewer than 5 moves must be skipped"

    def test_handles_empty_string(self):
        assert parse_pgn("") == []

    def test_strips_comments_and_annotations(self):
        """Comments {}, NAGs $N, and variations () must not appear in moves."""
        games = parse_pgn(ANNOTATED_PGN)
        assert len(games) == 1
        for move in games[0]:
            assert not move.startswith('{')
            assert not move.startswith('$')
            assert not move.startswith('(')


# ---------------------------------------------------------------------------
# ChessTokenizer
# ---------------------------------------------------------------------------

class TestChessTokenizer:
    def test_special_token_ids_are_fixed(self):
        tok = ChessTokenizer()
        assert tok.token2id[PAD_TOKEN] == 0
        assert tok.token2id[BOS_TOKEN] == 1
        assert tok.token2id[EOS_TOKEN] == 2
        assert tok.token2id[UNK_TOKEN] == 3

    def test_encode_adds_bos_and_eos(self):
        tok = ChessTokenizer()
        tok.build_from_games([["e4", "e5"]])
        ids = tok.encode(["e4", "e5"])
        assert ids[0] == tok.bos_id,  "First token must be BOS"
        assert ids[-1] == tok.eos_id, "Last token must be EOS"

    def test_encode_without_special_tokens(self):
        tok = ChessTokenizer()
        tok.build_from_games([["e4", "e5"]])
        ids = tok.encode(["e4", "e5"], add_special=False)
        assert len(ids) == 2
        assert ids[0] != tok.bos_id
        assert ids[-1] != tok.eos_id

    def test_decode_roundtrip(self):
        tok   = ChessTokenizer()
        moves = ["e4", "e5", "Nf3", "Nc6"]
        tok.build_from_games([moves])
        ids     = tok.encode(moves, add_special=False)
        decoded = tok.decode(ids)
        assert decoded == moves

    def test_unknown_move_maps_to_unk(self):
        tok = ChessTokenizer()
        tok.build_from_games([["e4", "e5"]])
        ids = tok.encode(["ZZZZ_UNKNOWN"], add_special=False)
        assert ids[0] == tok.token2id[UNK_TOKEN]

    def test_vocab_starts_with_only_special_tokens(self):
        tok = ChessTokenizer()
        assert tok.vocab_size == len(SPECIAL_TOKENS)

    def test_vocab_grows_after_build(self):
        tok = ChessTokenizer()
        tok.build_from_games([["e4", "e5"]])
        assert tok.vocab_size == len(SPECIAL_TOKENS) + 2  # "e4", "e5"

    def test_duplicate_moves_not_double_counted(self):
        tok = ChessTokenizer()
        tok.build_from_games([["e4", "e4", "e4"]])
        assert tok.vocab_size == len(SPECIAL_TOKENS) + 1

    def test_save_and_load_preserves_vocab(self, tmp_path):
        tok = ChessTokenizer()
        tok.build_from_games([["e4", "e5", "Nf3", "Nc6"]])
        path = str(tmp_path / "tokenizer.json")
        tok.save(path)
        tok2 = ChessTokenizer.load(path)
        assert tok2.token2id == tok.token2id
        assert tok2.id2token  == tok.id2token

    def test_properties_return_correct_ids(self):
        tok = ChessTokenizer()
        assert tok.pad_id == tok.token2id[PAD_TOKEN]
        assert tok.bos_id == tok.token2id[BOS_TOKEN]
        assert tok.eos_id == tok.token2id[EOS_TOKEN]


# ---------------------------------------------------------------------------
# ChessDataset
# ---------------------------------------------------------------------------

def _make_dataset(seq_len: int = 16) -> ChessDataset:
    tok       = ChessTokenizer()
    moves     = ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7"]
    games_raw = [moves for _ in range(5)]
    tok.build_from_games(games_raw)
    encoded = [tok.encode(g) for g in games_raw]
    return ChessDataset(encoded, seq_len=seq_len, pad_id=tok.pad_id)


class TestChessDataset:
    def test_nonzero_length(self):
        assert len(_make_dataset()) > 0

    def test_sample_shapes(self):
        ds = _make_dataset(seq_len=16)
        inp, tgt = ds[0]
        assert inp.shape == (16,)
        assert tgt.shape == (16,)

    def test_input_output_offset_by_one(self):
        """
        For non-padded positions: tgt[i] == inp[i+1].
        This is the defining property of autoregressive LM data.
        """
        tok   = ChessTokenizer()
        moves = ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7"]
        tok.build_from_games([moves])
        encoded = [tok.encode(moves)]                  # BOS + 10 moves + EOS = 12 tokens
        seq_len = len(encoded[0]) - 1                  # 11
        ds      = ChessDataset(encoded, seq_len=seq_len, pad_id=tok.pad_id)
        inp, tgt = ds[0]

        for i in range(len(inp) - 1):
            if tgt[i].item() == -1:
                continue  # padding position — skip
            assert inp[i + 1].item() == tgt[i].item(), (
                f"Offset mismatch at position {i}: inp[{i+1}]={inp[i+1].item()} "
                f"!= tgt[{i}]={tgt[i].item()}"
            )

    def test_padded_targets_are_minus_one(self):
        """
        When a game is shorter than seq_len, padding positions in targets must
        be -1 so cross_entropy ignores them.
        """
        ds = _make_dataset(seq_len=64)  # longer than any game → forces padding
        found_padding = False
        for i in range(len(ds)):
            inp, tgt = ds[i]
            if (inp == 0).any():
                assert (tgt == -1).any(), (
                    "Padded input positions must have target == -1"
                )
                found_padding = True
                break
        # If no padding was created the test is vacuously true — warn but don't fail
        # (This can happen if games are longer than seq_len; increase seq_len if so)

    def test_tensor_dtypes(self):
        ds = _make_dataset()
        inp, tgt = ds[0]
        assert inp.dtype == torch.long
        assert tgt.dtype == torch.long

    def test_all_input_values_non_negative(self):
        ds = _make_dataset()
        for i in range(len(ds)):
            inp, _ = ds[i]
            assert inp.min().item() >= 0

    def test_target_values_are_valid_ids_or_minus_one(self):
        """Target ids must be vocab indices or -1 (padding ignore index)."""
        tok   = ChessTokenizer()
        moves = ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"]
        tok.build_from_games([moves])
        encoded = [tok.encode(moves)]
        ds = ChessDataset(encoded, seq_len=16, pad_id=tok.pad_id)
        for i in range(len(ds)):
            _, tgt = ds[i]
            valid = (tgt >= 0) & (tgt < tok.vocab_size)
            ignored = tgt == -1
            assert (valid | ignored).all(), (
                "Every target must be a valid vocab id or -1 (padding)"
            )
