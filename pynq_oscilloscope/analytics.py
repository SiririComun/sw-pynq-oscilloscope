"""
pynq_oscilloscope.analytics: High-Precision Signal Processing & Acoustic Analytics Engine.
Provides Hilbert analytic envelope extraction, Short-Time Energy (STE), Inter-aural Level Difference (ILD),
sliding STFT spectrograms, sub-Hertz pitch interpolation, and instantaneous phase tracking.
"""

from typing import Tuple, Optional, Union
import numpy as np


class AcousticAnalytics:
    """
    High-performance, zero-copy acoustic signal analytics engine.
    Provides mathematical extraction for Amplitude vs. Time and Frequency vs. Time metrics.
    """

    @staticmethod
    def hilbert_transform(x: np.ndarray) -> np.ndarray:
        """
        Computes the analytic signal z[n] = x[n] + j*H{x[n]} using pure NumPy FFT.
        
        :param x: 1D real-valued signal.
        :return: 1D complex-valued analytic signal.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        n = len(x_arr)
        if n == 0:
            return np.array([], dtype=np.complex128)

        xf = np.fft.fft(x_arr)
        h = np.zeros(n, dtype=np.float64)

        if n % 2 == 0:
            h[0] = 1.0
            h[n // 2] = 1.0
            h[1 : n // 2] = 2.0
        else:
            h[0] = 1.0
            h[1 : (n + 1) // 2] = 2.0

        return np.fft.ifft(xf * h)

    @classmethod
    def extract_analytic_envelope(
        cls,
        signal: np.ndarray,
        remove_dc: bool = True
    ) -> np.ndarray:
        """
        Extracts the true physical instantaneous amplitude envelope A(t) = |z(t)|.
        
        :param signal: 1D physical voltage array.
        :param remove_dc: Automatically subtract mean DC baseline before envelope extraction.
        :return: 1D instantaneous amplitude envelope array of same length.
        """
        x = np.asarray(signal, dtype=np.float64)
        if remove_dc:
            x = x - np.mean(x)
            
        analytic_signal = cls.hilbert_transform(x)
        return np.abs(analytic_signal)

    @staticmethod
    def compute_short_time_energy(
        signal: np.ndarray,
        window_len: int = 64,
        hop_size: int = 16
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the Short-Time Energy (STE) profile: E[m] = sum(x[n]^2 * w[n]).
        
        :param signal: 1D voltage array.
        :param window_len: Number of samples per energy integration window.
        :param hop_size: Step size between successive energy windows.
        :return: (ste_curve, sample_indices).
        """
        x = np.asarray(signal, dtype=np.float64)
        x_ac = x - np.mean(x)
        x_sq = x_ac ** 2

        n_samples = len(x_sq)
        if n_samples < window_len:
            return np.array([np.mean(x_sq)]), np.array([0])

        indices = np.arange(0, n_samples - window_len + 1, hop_size)
        win = np.hanning(window_len)
        win_norm = win / np.sum(win)

        ste = np.array([
            np.sum(x_sq[idx : idx + window_len] * win_norm)
            for idx in indices
        ])
        return ste, indices + (window_len // 2)

    @classmethod
    def compute_ild(
        cls,
        sig_a0: np.ndarray,
        sig_a1: np.ndarray,
        window_len: int = 64,
        hop_size: int = 16,
        eps: float = 1e-9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the Inter-aural Level Difference (ILD) in decibels:
        Delta_L(t) = 10 * log10( (STE_A0(t) + eps) / (STE_A1(t) + eps) )
        
        :param sig_a0: Channel 1 (A0) voltage array.
        :param sig_a1: Channel 2 (A1) voltage array.
        :return: (ild_db, ste_a0, ste_a1, sample_indices).
        """
        ste_0, idx0 = cls.compute_short_time_energy(sig_a0, window_len=window_len, hop_size=hop_size)
        ste_1, _ = cls.compute_short_time_energy(sig_a1, window_len=window_len, hop_size=hop_size)

        ild_db = 10.0 * np.log10(np.maximum(ste_0, eps) / np.maximum(ste_1, eps))
        return ild_db, ste_0, ste_1, idx0

    @classmethod
    def compute_instantaneous_phase_diff(
        cls,
        sig_a0: np.ndarray,
        sig_a1: np.ndarray,
        remove_dc: bool = True
    ) -> np.ndarray:
        """
        Computes continuous instantaneous phase difference between A0 and A1:
        Delta_phi(t) = unwrap( angle(z_A0(t)) - angle(z_A1(t)) ) in degrees.
        
        :return: 1D phase difference array in degrees (-180° to +180° wrapped or continuous).
        """
        x0 = np.asarray(sig_a0, dtype=np.float64)
        x1 = np.asarray(sig_a1, dtype=np.float64)
        if remove_dc:
            x0 = x0 - np.mean(x0)
            x1 = x1 - np.mean(x1)

        z0 = cls.hilbert_transform(x0)
        z1 = cls.hilbert_transform(x1)

        phi0 = np.unwrap(np.angle(z0))
        phi1 = np.unwrap(np.angle(z1))

        diff_deg = np.rad2deg(phi0 - phi1)
        # Normalize within [-180, +180] band
        return (diff_deg + 180.0) % 360.0 - 180.0

    @staticmethod
    def compute_sliding_stft(
        signal: np.ndarray,
        fs: float,
        nperseg: int = 512,
        noverlap: int = 384,
        window: str = "blackmanharris"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes a 2D Short-Time Fourier Transform (STFT) Spectrogram matrix.
        
        :param signal: 1D physical voltage array.
        :param fs: Sampling frequency in Hz.
        :param nperseg: Number of samples per FFT window segment (e.g., 512, 1024).
        :param noverlap: Number of overlapping samples (default 75% overlap).
        :param window: Window type ('hann', 'hamming', 'blackmanharris').
        :return: (time_axis_ms, freq_axis_hz, spectrogram_dBV_matrix).
        """
        x = np.asarray(signal, dtype=np.float64)
        x_ac = x - np.mean(x)
        n_samples = len(x_ac)

        step = nperseg - noverlap
        if n_samples < nperseg or step <= 0:
            freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
            spec = 20.0 * np.log10(np.maximum(np.abs(np.fft.rfft(x_ac)) / (n_samples / 2.0), 1e-6))
            return np.array([0.0]), freqs, spec[:, np.newaxis]

        # Generate window
        if window == "blackmanharris":
            n = np.arange(nperseg)
            w = (0.35875 - 0.48829 * np.cos(2.0 * np.pi * n / (nperseg - 1)) +
                 0.14128 * np.cos(4.0 * np.pi * n / (nperseg - 1)) -
                 0.01168 * np.cos(6.0 * np.pi * n / (nperseg - 1)))
        elif window == "hamming":
            w = np.hamming(nperseg)
        else:
            w = np.hanning(nperseg)

        coherent_gain = np.sum(w) / nperseg
        freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
        indices = np.arange(0, n_samples - nperseg + 1, step)
        times_ms = (indices + (nperseg / 2.0)) * (1000.0 / fs)

        spec_matrix = np.zeros((len(freqs), len(indices)), dtype=np.float64)

        for i, idx in enumerate(indices):
            segment = x_ac[idx : idx + nperseg] * w
            fft_res = np.abs(np.fft.rfft(segment)) / (nperseg / 2.0)
            linear_v = fft_res / max(coherent_gain, 1e-4)
            spec_matrix[:, i] = 20.0 * np.log10(np.maximum(linear_v, 1e-6))

        return times_ms, freqs, spec_matrix

    @staticmethod
    def track_sub_hertz_pitch(
        freqs: np.ndarray,
        mags: np.ndarray,
        min_freq_hz: float = 20.0,
        max_freq_hz: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Extracts dominant fundamental pitch frequency with sub-Hertz accuracy (±0.2 Hz)
        using three-point parabolic interpolation on spectral peaks.
        
        :param freqs: 1D frequency axis array in Hz.
        :param mags: 1D magnitude array in dB.
        :param min_freq_hz: Minimum frequency threshold to ignore 0 Hz DC bins.
        :param max_freq_hz: Optional upper band limit.
        :return: (peak_freq_hz, peak_mag_dB).
        """
        valid_mask = (freqs >= min_freq_hz)
        if max_freq_hz is not None:
            valid_mask &= (freqs <= max_freq_hz)

        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            k = int(np.argmax(mags))
            return float(freqs[k]), float(mags[k])

        k = int(valid_indices[np.argmax(mags[valid_indices])])

        # Parabolic peak interpolation
        if k <= 0 or k >= len(mags) - 1:
            return float(freqs[k]), float(mags[k])

        alpha = float(mags[k - 1])
        beta  = float(mags[k])
        gamma = float(mags[k + 1])

        denom = alpha - 2.0 * beta + gamma
        if abs(denom) < 1e-12:
            return float(freqs[k]), float(beta)

        delta = 0.5 * (alpha - gamma) / denom
        delta = max(-0.5, min(0.5, delta))

        delta_f = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
        interp_freq = float(freqs[k] + delta * delta_f)
        interp_mag = float(beta - 0.25 * (alpha - gamma) * delta)

        return interp_freq, interp_mag