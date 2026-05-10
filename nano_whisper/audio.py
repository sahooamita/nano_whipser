"""Audio preprocessing utilities."""

import torch
import torch.nn.functional as F
import numpy as np


def hanning_window(window_length: int) -> torch.Tensor:
    """Compute a Hann window."""
    return torch.hann_window(window_length)


def stft(
    waveform: torch.Tensor,
    n_fft: int = 400,
    hop_length: int = 160,
    win_length: int | None = None,
) -> torch.Tensor:
    """Short-time Fourier transform."""
    if win_length is None:
        win_length = n_fft
    window = hanning_window(win_length).to(waveform.device)
    return torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
        center=True,
        pad_mode="reflect",
    )


def mel_filters(
    n_mels: int = 80,
    n_fft: int = 400,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """Generate a simple mel filterbank matrix."""
    # Simplified mel filterbank (triangular filters)
    f_min = 0.0
    f_max = float(sample_rate) / 2
    mel_min = 2595 * np.log10(1 + f_min / 700)
    mel_max = 2595 * np.log10(1 + f_max / 700)
    mels = np.linspace(mel_min, mel_max, n_mels + 2)
    freqs = 700 * (10 ** (mels / 2595) - 1)
    fft_freqs = np.linspace(0, f_max, n_fft // 2 + 1)

    filters = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        left, center, right = freqs[i], freqs[i + 1], freqs[i + 2]
        for j, f in enumerate(fft_freqs):
            if left < f < right:
                filters[i, j] = max(0.0, 1.0 - abs(f - center) / (center - left if f <= center else right - center))
    return torch.from_numpy(filters).float()


def log_mel_spectrogram(
    audio: np.ndarray | torch.Tensor,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """Convert raw audio to log-mel spectrogram.

    Args:
        audio: Raw audio waveform at 16kHz, shape (T,)
        n_mels: Number of mel bins
        n_fft: FFT size
        hop_length: Hop length between frames
        sample_rate: Sample rate of the audio

    Returns:
        Log-mel spectrogram of shape (n_mels, n_frames)
    """
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio).float()
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)  # (1, T)

    spec = stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitudes = spec.abs().pow(2)

    mel_basis = mel_filters(n_mels, n_fft, sample_rate).to(audio.device)
    mel_spec = mel_basis @ magnitudes.squeeze(0)

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec
