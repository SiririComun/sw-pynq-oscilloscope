"""
pynq_oscilloscope.audio_utils: Audio conditioning, DC removal, normalization,
and in-memory WAV byte stream generation.
"""

from typing import Tuple, Optional
import io
import wave
import numpy as np


def deinterleave_stereo(raw_interleaved: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits interleaved dual-channel uint16 samples into separate Channel 1 (A0) and Channel 2 (A1) arrays.
    """
    raw_ch1 = raw_interleaved[0::2]
    raw_ch2 = raw_interleaved[1::2]
    return raw_ch1, raw_ch2


def raw_to_voltages(raw_samples: np.ndarray) -> np.ndarray:
    """
    Converts 12-bit left-aligned XADC codes (shifted left by 4) to physical voltages (0.0V - 3.3V).
    """
    return (raw_samples >> 4) * (3.3 / 4095.0)


def remove_dc_offset(voltages: np.ndarray, method: str = "mean") -> np.ndarray:
    """
    Removes the DC bias (resting midpoint voltage) from the analog waveform.

    :param voltages: 1D NumPy array of physical voltages.
    :param method: 'mean' for zero-centering, or 'highpass' for recursive 1st-order DC blocker.
    :return: 1D NumPy array of AC-coupled audio.
    """
    if len(voltages) == 0:
        return np.empty(0, dtype=np.float64)

    if method == "mean":
        return voltages - np.mean(voltages)
    elif method == "highpass":
        # 1st-order DC-blocking IIR filter: y[n] = x[n] - x[n-1] + R * y[n-1] (R = 0.995)
        r = 0.995
        y = np.zeros_like(voltages)
        for n in range(1, len(voltages)):
            y[n] = voltages[n] - voltages[n - 1] + r * y[n - 1]
        return y
    else:
        raise ValueError(f"Invalid method '{method}'. Choose from: 'mean', 'highpass'.")


def normalize_audio(
    ac_signal: np.ndarray,
    peak_ref: float = 1.65,
    auto_gain: bool = False
) -> np.ndarray:
    """
    Normalizes AC-coupled audio to the standard floating-point audio range [-1.0, +1.0].

    :param ac_signal: AC-coupled audio waveform.
    :param peak_ref: Reference peak voltage for 0 dBFS (defaults to 1.65V, the MAX4466/XADC dynamic range).
    :param auto_gain: If True, scales peak amplitude to exactly 0.95 to maximize volume without clipping.
    :return: Float64 array clipped to [-1.0, +1.0].
    """
    if len(ac_signal) == 0:
        return np.empty(0, dtype=np.float64)

    if auto_gain:
        max_abs = np.max(np.abs(ac_signal))
        if max_abs > 1e-5:
            scaled = (ac_signal / max_abs) * 0.95
        else:
            scaled = ac_signal
    else:
        scaled = ac_signal / max(1e-3, peak_ref)

    return np.clip(scaled, -1.0, 1.0)


def audio_to_wav_bytes(
    audio_data: np.ndarray,
    sample_rate_hz: int,
    num_channels: int = 1
) -> bytes:
    """
    Encodes normalized float [-1.0, +1.0] audio into standard 16-bit PCM WAV format in-memory.

    :param audio_data: 1D array (mono) or 2D array (stereo shape: [samples, 2]) in range [-1.0, 1.0].
    :param sample_rate_hz: Sampling frequency in Hertz (e.g. 50000).
    :param num_channels: 1 for Mono, 2 for Stereo.
    :return: Bytes object containing complete RIFF WAV data.
    """
    # 1. Convert float [-1.0, 1.0] to int16 [-32767, 32767]
    pcm_int16 = (audio_data * 32767.0).astype(np.int16)

    # 2. Write WAV headers and payload into in-memory buffer
    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(2) # 16 bits = 2 bytes
        wav_file.setframerate(int(sample_rate_hz))
        wav_file.writeframes(pcm_int16.tobytes())

    return wav_io.getvalue()


def process_raw_recording(
    raw_interleaved: np.ndarray,
    channel: int = 1,
    target_samples: Optional[int] = None,
    peak_ref: float = 1.65,
    auto_gain: bool = False
) -> np.ndarray:
    """
    Full DSP pipeline: De-interleaves, converts to voltages, removes DC offset,
    trims to target sample count, and normalizes to [-1.0, +1.0].

    :param raw_interleaved: Raw multi-frame uint16 DMA buffer.
    :param channel: 1 for Channel 1 (A0), 2 for Channel 2 (A1), or 0 for Stereo tuple.
    :param target_samples: Exact sample count to truncate to.
    :param peak_ref: Voltage reference for normalization.
    :param auto_gain: Auto-scale peak amplitude.
    :return: Normalized float64 audio array [-1.0, 1.0].
    """
    raw_ch1, raw_ch2 = deinterleave_stereo(raw_interleaved)

    if channel == 1:
        selected_raw = raw_ch1
    elif channel == 2:
        selected_raw = raw_ch2
    elif channel == 0:
        # Stereo mode
        v1 = raw_to_voltages(raw_ch1)
        v2 = raw_to_voltages(raw_ch2)
        ac1 = remove_dc_offset(v1)
        ac2 = remove_dc_offset(v2)
        if target_samples is not None:
            ac1 = ac1[:target_samples]
            ac2 = ac2[:target_samples]
        s1 = normalize_audio(ac1, peak_ref=peak_ref, auto_gain=auto_gain)
        s2 = normalize_audio(ac2, peak_ref=peak_ref, auto_gain=auto_gain)
        return np.column_stack((s1, s2))
    else:
        raise ValueError(f"Invalid channel '{channel}'. Choose 1 (A0), 2 (A1), or 0 (Stereo).")

    # Mono pipeline
    voltages = raw_to_voltages(selected_raw)
    ac_signal = remove_dc_offset(voltages)

    if target_samples is not None and len(ac_signal) > target_samples:
        ac_signal = ac_signal[:target_samples]

    normalized = normalize_audio(ac_signal, peak_ref=peak_ref, auto_gain=auto_gain)
    return normalized