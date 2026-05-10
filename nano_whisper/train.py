"""Minimal training script for NanoWhisper."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from nano_whisper.model import NanoWhisper
from nano_whisper.audio import log_mel_spectrogram
from nano_whisper.tokenizer import CharTokenizer


class DummyASRDataset(Dataset):
    """Dummy dataset for testing; replace with real data."""

    def __init__(self, manifest_path: str, tokenizer: CharTokenizer, max_audio_len: int = 16000 * 10):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_audio_len = max_audio_len
        with open(manifest_path) as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        audio = torch.randn(self.max_audio_len)  # dummy audio; replace with load
        mel = log_mel_spectrogram(audio)
        tokens = torch.tensor(self.tokenizer.encode(item["text"]), dtype=torch.long)
        return mel, tokens


def collate_fn(batch):
    mels, tokens = zip(*batch)
    max_mel_len = max(m.shape[1] for m in mels)
    max_tok_len = max(len(t) for t in tokens)

    mel_padded = torch.zeros(len(batch), mels[0].shape[0], max_mel_len)
    tok_padded = torch.full((len(batch), max_tok_len), CharTokenizer.PAD_ID, dtype=torch.long)

    for i, (m, t) in enumerate(zip(mels, tokens)):
        mel_padded[i, :, : m.shape[1]] = m
        tok_padded[i, : len(t)] = t

    return mel_padded, tok_padded


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    for mel, tokens in dataloader:
        mel, tokens = mel.to(device), tokens.to(device)
        optimizer.zero_grad()

        logits = model(mel, tokens[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tokens[:, 1:].reshape(-1),
            ignore_index=CharTokenizer.PAD_ID,
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Path to JSON-lines manifest")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save_dir", default="checkpoints")
    args = parser.parse_args()

    tokenizer = CharTokenizer()
    model = NanoWhisper(vocab_size=tokenizer.vocab_size)
    model.to(args.device)

    dataset = DummyASRDataset(args.manifest, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, dataloader, optimizer, args.device)
        print(f"Epoch {epoch}/{args.epochs} | loss={loss:.4f}")
        ckpt = Path(args.save_dir) / f"epoch_{epoch}.pt"
        torch.save(model.state_dict(), ckpt)


if __name__ == "__main__":
    main()
