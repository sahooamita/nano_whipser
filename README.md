# Nano Whisper

A minimal, from-scratch implementation of an encoder-decoder speech-to-text (ASR) model inspired by [OpenAI Whisper](https://github.com/openai/whisper). Designed for educational purposes and easy hacking.

## Features

- **Tiny transformer** encoder-decoder (~4M params with default config)
- **Log-mel spectrogram** extraction in pure PyTorch + NumPy
- **Character-level tokenizer** with no external dependencies
- **Greedy decoding** for transcription
- Clean, readable code suitable for learning or prototyping

## Installation

```bash
git clone https://github.com/sahooamita/nano_whipser.git
cd nano_whipser
pip install -r requirements.txt
```

## Quick Start

### Training

Prepare a JSON-lines manifest where each line is `{"audio": "path/to.wav", "text": "hello world"}`:

```bash
python -m nano_whisper.train \
  --manifest data/train.jsonl \
  --epochs 10 \
  --batch_size 4 \
  --device cpu
```

### Inference

```bash
python -m nano_whisper.inference \
  --audio samples/hello.wav \
  --checkpoint checkpoints/epoch_10.pt \
  --device cpu
```

## Project Structure

```
nano_whisper/
├── __init__.py
├── model.py       # NanoWhisper encoder-decoder transformer
├── audio.py       # STFT and log-mel spectrogram
├── tokenizer.py   # Character-level tokenizer
├── train.py       # Minimal training loop
└── inference.py   # Greedy decoding transcription
```

## License

MIT
