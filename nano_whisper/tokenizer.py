"""Simple character-level tokenizer."""

import re


class CharTokenizer:
    """Character-level tokenizer with special tokens."""

    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3

    def __init__(self, vocab: str = "abcdefghijklmnopqrstuvwxyz '."):
        self.char_to_id = {
            "<pad>": self.PAD_ID,
            "<bos>": self.BOS_ID,
            "<eos>": self.EOS_ID,
            "<unk>": self.UNK_ID,
        }
        for ch in vocab:
            if ch not in self.char_to_id:
                self.char_to_id[ch] = len(self.char_to_id)
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id)

    def encode(self, text: str) -> list[int]:
        text = text.lower().strip()
        text = re.sub(r"[^a-z '.]", "", text)
        return [self.BOS_ID] + [self.char_to_id.get(ch, self.UNK_ID) for ch in text] + [self.EOS_ID]

    def decode(self, ids: list[int]) -> str:
        chars = []
        for idx in ids:
            if idx in (self.PAD_ID, self.BOS_ID, self.EOS_ID):
                continue
            chars.append(self.id_to_char.get(idx, "<unk>"))
        return "".join(chars)
