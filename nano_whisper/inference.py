"""Minimal inference script for NanoWhisper."""

import argparse

import torch
import numpy as np

from nano_whisper.model import NanoWhisper
from nano_whisper.audio import log_mel_spectrogram
from nano_whisper.tokenizer import CharTokenizer


def load_audio(path: str, sample_rate: int = 16000) -> np.ndarray:
    """Load audio file to mono waveform at target sample rate.

    Requires soundfile or librosa; falls back to dummy data if unavailable.
    """
    try:
        import soundfile as sf
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != sample_rate:
            # Simple linear resample
            from math import gcd
            g = gcd(sr, sample_rate)
            audio = np.interp(
                np.linspace(0, len(audio), int(len(audio) * sample_rate / sr)),
                np.arange(len(audio)),
                audio,
            )
        return audio
    except Exception:
        raise RuntimeError("Install soundfile to load real audio: pip install soundfile")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    tokenizer = CharTokenizer()
    model = NanoWhisper(vocab_size=tokenizer.vocab_size)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.to(args.device)
    model.eval()

    audio = load_audio(args.audio)
    mel = log_mel_spectrogram(audio).unsqueeze(0).to(args.device)
    text = model.transcribe(mel, tokenizer)
    print(text)


if __name__ == "__main__":
    main()
