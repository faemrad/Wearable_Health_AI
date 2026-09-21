from __future__ import annotations

import numpy as np
from scipy.signal import butter, decimate, filtfilt, iirnotch, medfilt, resample

from .paths import HUBERT_CROP, HUBERT_DOWNSAMPLE, TARGET_FS, TARGET_LEN


def filter_bandpass_founder(signal: np.ndarray, fs: int) -> np.ndarray:
    """ECGFounder bandpass: 50 Hz notch, 0.67–40 Hz, median baseline."""
    b, a = iirnotch(50, 30, fs)
    filtered = np.zeros_like(signal)
    for c in range(signal.shape[0]):
        filtered[c] = filtfilt(b, a, signal[c])

    b, a = butter(N=4, Wn=[0.67, 40], btype="bandpass", fs=fs)
    for c in range(filtered.shape[0]):
        filtered[c] = filtfilt(b, a, filtered[c])

    baseline = np.zeros_like(filtered)
    kernel_size = int(0.4 * fs) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    for c in range(filtered.shape[0]):
        baseline[c] = medfilt(filtered[c], kernel_size=kernel_size)
    return filtered - baseline


def z_score(signal: np.ndarray) -> np.ndarray:
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)


def upsample_to_500(ecg_12: np.ndarray, fs_in: int) -> np.ndarray:
    if fs_in == TARGET_FS:
        x = ecg_12
    else:
        n_out = int(round(ecg_12.shape[1] * TARGET_FS / fs_in))
        x = resample(ecg_12, n_out, axis=1)
    if x.shape[1] >= TARGET_LEN:
        return x[:, :TARGET_LEN]
    pad = TARGET_LEN - x.shape[1]
    return np.pad(x, ((0, 0), (0, pad)))


def founder_1lead(ecg_12_500: np.ndarray) -> np.ndarray:
    lead_ii = ecg_12_500[1:2]
    x = filter_bandpass_founder(lead_ii, TARGET_FS)
    return z_score(x).astype(np.float32)


def hubert_minmax(seq: np.ndarray, smooth: float = 1e-8) -> np.ndarray:
    return (
        2
        * (seq - np.min(seq, axis=1)[None].T)
        / (np.max(seq, axis=1) - np.min(seq, axis=1) + smooth)[None].T
        - 1
    )


def hubert_bandpass(ecg_12: np.ndarray, fs: int) -> np.ndarray:
    """FIR 0.05–47 Hz. Prefer biosppy; fall back to Butterworth."""
    try:
        from biosppy.signals.tools import filter_signal

        order = int(0.3 * fs)
        out, _, _ = filter_signal(
            signal=ecg_12,
            ftype="FIR",
            band="bandpass",
            order=order,
            frequency=[0.05, 47],
            sampling_rate=fs,
        )
        return np.asarray(out)
    except Exception:
        nyq = fs / 2.0
        high = min(47.0, nyq * 0.95)
        b, a = butter(N=4, Wn=[0.05, high], btype="bandpass", fs=fs)
        return filtfilt(b, a, ecg_12, axis=1)


def hubert_input(ecg_12_500: np.ndarray, random_crop: bool = False) -> np.ndarray:
    x = hubert_bandpass(ecg_12_500, TARGET_FS)
    x = hubert_minmax(x)
    if x.shape[1] < HUBERT_CROP:
        x = np.pad(x, ((0, 0), (0, HUBERT_CROP - x.shape[1])))
    if random_crop and x.shape[1] > HUBERT_CROP:
        start = np.random.randint(0, x.shape[1] - HUBERT_CROP + 1)
        x = x[:, start : start + HUBERT_CROP]
    else:
        x = x[:, :HUBERT_CROP]
    flat = x.reshape(-1)
    return decimate(flat, HUBERT_DOWNSAMPLE).astype(np.float32)
