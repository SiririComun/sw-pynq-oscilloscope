"""
pynq_oscilloscope.fft_dma: Advanced FFT & Spectral Analysis Engine for PYNQ-Z2.
Features multi-windowing (Hann, Hamming, Blackman, Flat-Top), sub-bin quadratic
interpolation, and hardware CORDIC DMA magnitude stream processing.
"""

from typing import Tuple, Optional, Dict
import numpy as np
from pynq import allocate


class StreamingFFT:
    """
    High-level driver for FFT analysis, windowing, and CORDIC magnitude streaming.
    Supports dynamic sampling rates, window functions, and sub-Hertz peak tracking.
    """

    WINDOWS = ["hann", "hamming", "blackman", "flattop", "rectangular"]

    def __init__(self, overlay, fft_points: int = 2048, sample_rate_hz: float = 50_000.0):
        """
        Initialize the FFT controller.
        
        :param overlay: Loaded pynq.Overlay or OscilloscopeOverlay instance.
        :param fft_points: Transform length N (default: 2048).
        :param sample_rate_hz: Sampling frequency fs (default: 50.0 kSPS).
        """
        self.fft_points = int(fft_points)
        self.sample_rate_hz = float(sample_rate_hz)
        self.num_bins = self.fft_points // 2  # Single-sided spectrum bins (1024)
        self.active_window = "hann"
        
        # Grid parameters
        self.delta_f = self.sample_rate_hz / self.fft_points
        self.freq_axis = np.arange(self.num_bins) * self.delta_f

        # Dynamic binding to axi_dma_1 (by name or base address 0x40410000)
        if hasattr(overlay, "axi_dma_1"):
            self.dma = overlay.axi_dma_1
        else:
            dma_blocks = [
                getattr(overlay, ip) for ip, details in overlay.ip_dict.items()
                if "dma" in ip.lower() and details.get("phys_addr") == 0x40410000
            ]
            if dma_blocks:
                self.dma = dma_blocks[0]
            else:
                all_dmas = [ip for ip in overlay.ip_dict.keys() if "dma" in ip.lower()]
                if len(all_dmas) > 1:
                    self.dma = getattr(overlay, all_dmas[1])
                elif all_dmas:
                    self.dma = getattr(overlay, all_dmas[0])
                else:
                    raise RuntimeError("No AXI DMA block found for FFT stream (axi_dma_1).")

        # Allocate contiguous CMA buffer
        self._buffer = allocate(shape=(self.fft_points,), dtype="u2")
        self._window_cache: Dict[str, np.ndarray] = {}

    def update_configuration(self, fft_points: Optional[int] = None, sample_rate_hz: Optional[float] = None):
        """
        Updates the active FFT transform length and ADC sampling rate parameters.
        """
        if fft_points is not None and int(fft_points) != self.fft_points:
            self.fft_points = int(fft_points)
            self.num_bins = self.fft_points // 2
            if hasattr(self, "_buffer") and self._buffer is not None:
                try:
                    self._buffer.close()
                except Exception:
                    pass
            self._buffer = allocate(shape=(self.fft_points,), dtype="u2")

        if sample_rate_hz is not None:
            self.sample_rate_hz = float(sample_rate_hz)

        self.delta_f = self.sample_rate_hz / self.fft_points
        self.freq_axis = np.arange(self.num_bins) * self.delta_f

    # =========================================================================
    # 1. Windowing Engine
    # =========================================================================

    def set_window(self, window_type: str = "hann"):
        """
        Sets the active window function.
        
        :param window_type: 'hann', 'hamming', 'blackman', 'flattop', or 'rectangular'.
        """
        win_clean = window_type.lower().strip()
        if win_clean not in self.WINDOWS:
            raise ValueError(f"Invalid window '{window_type}'. Choose from: {self.WINDOWS}")
        self.active_window = win_clean

    def get_window_vector(self, length: int, window_type: Optional[str] = None) -> np.ndarray:
        """Generates or retrieves a cached window vector of given length."""
        win_name = (window_type or self.active_window).lower().strip()
        cache_key = f"{win_name}_{length}"
        
        if cache_key in self._window_cache:
            return self._window_cache[cache_key]

        n = np.arange(length)
        if win_name == "hann":
            w = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / (length - 1))
        elif win_name == "hamming":
            w = 0.54 - 0.46 * np.cos(2.0 * np.pi * n / (length - 1))
        elif win_name == "blackman":
            w = (0.42 - 0.5 * np.cos(2.0 * np.pi * n / (length - 1)) +
                 0.08 * np.cos(4.0 * np.pi * n / (length - 1)))
        elif win_name == "flattop":
            w = (0.21557895 - 0.41663158 * np.cos(2.0 * np.pi * n / (length - 1)) +
                 0.277263158 * np.cos(4.0 * np.pi * n / (length - 1)) -
                 0.083578947 * np.cos(6.0 * np.pi * n / (length - 1)) +
                 0.006947368 * np.cos(8.0 * np.pi * n / (length - 1)))
        else:  # rectangular
            w = np.ones(length, dtype=np.float64)

        self._window_cache[cache_key] = w
        return w

    # =========================================================================
    # 2. Software Spectral Computation from Time Samples
    # =========================================================================

    def compute_spectrum(
        self,
        time_samples: np.ndarray,
        unit: str = "dBV",
        window_type: Optional[str] = None,
        remove_dc: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes a windowed, zero-centered, power-normalized RFFT spectrum.
        
        :param time_samples: 1D numpy array of physical voltages.
        :param unit: 'dBV', 'dBFS', or 'Linear' (mV).
        :param window_type: Window name (defaults to active_window).
        :param remove_dc: Subtract mean baseline before FFT (default: True).
        :return: (freq_axis_hz, magnitude_array).
        """
        x = np.array(time_samples, dtype=np.float64)
        if remove_dc:
            x = x - np.mean(x)
            
        n = len(x)
        w = self.get_window_vector(n, window_type=window_type)
        windowed_x = x * w

        coherent_gain = np.sum(w) / n
        fft_complex = np.fft.rfft(windowed_x)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate_hz)
        
        linear_volts = (np.abs(fft_complex) / (n / 2.0)) / max(coherent_gain, 1e-4)

        unit_clean = unit.strip().upper()
        if unit_clean == "DBV":
            mags = 20.0 * np.log10(np.maximum(linear_volts, 1e-6))
        elif unit_clean == "DBFS":
            mags = 20.0 * np.log10(np.maximum(linear_volts / 1.65, 1e-6))
        elif unit_clean == "LINEAR":
            mags = linear_volts * 1000.0  # Millivolts
        else:
            raise ValueError(f"Invalid unit '{unit}'. Choose from: 'dBV', 'dBFS', 'Linear'.")

        return freqs, mags

    # =========================================================================
    # 3. Hardware Spectrum DMA Streaming
    # =========================================================================

    def capture_raw(self) -> np.ndarray:
        """Triggers hardware S2MM DMA transfer of 2048 magnitude points from PL."""
        self.dma.recvchannel.transfer(self._buffer)
        self.dma.recvchannel.wait()
        return np.array(self._buffer, copy=True)

    def process_buffer(
        self,
        cma_buffer,
        unit: str = "dBV",
        window: str = "hann",
        ref_voltage: float = 3.3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Processes a raw CORDIC magnitude buffer into calibrated spectrum magnitudes.
        """
        raw_full = np.array(cma_buffer, copy=True)
        raw_half = raw_full[:self.num_bins].astype(np.float64)

        linear_volts = (raw_half / 2048.0) * (ref_voltage / 4095.0)

        unit_clean = unit.strip().upper()
        if unit_clean == "DBV":
            safe_linear = np.maximum(linear_volts, 1e-6)
            magnitudes = 20.0 * np.log10(safe_linear)
        elif unit_clean == "DBFS":
            safe_raw = np.maximum(raw_half, 1.0)
            magnitudes = 20.0 * np.log10(safe_raw / 65535.0)
        elif unit_clean == "LINEAR":
            magnitudes = linear_volts * 1000.0
        else:
            raise ValueError(f"Invalid unit '{unit}'. Choose from: 'dBV', 'dBFS', 'Linear'.")

        return self.freq_axis, magnitudes

    def capture_spectrum(self, unit: str = "dBV", ref_voltage: float = 3.3) -> Tuple[np.ndarray, np.ndarray]:
        """Captures hardware FFT frame and returns (frequencies, magnitudes)."""
        raw_full = self.capture_raw()
        return self.process_buffer(raw_full, unit=unit, ref_voltage=ref_voltage)

    # =========================================================================
    # 4. Sub-Bin Quadratic Peak Interpolation
    # =========================================================================

    @staticmethod
    def get_peak_frequency(
        freqs: np.ndarray,
        mags: np.ndarray,
        min_freq_hz: float = 20.0,
        interpolate: bool = True
    ) -> Tuple[float, float]:
        """
        Finds the dominant pitch / fundamental frequency with sub-Hertz accuracy
        using three-point parabolic interpolation on spectral peaks.
        """
        valid_indices = np.where(freqs >= min_freq_hz)[0]
        if len(valid_indices) == 0:
            k = int(np.argmax(mags))
        else:
            k = int(valid_indices[np.argmax(mags[valid_indices])])

        if not interpolate or k <= 0 or k >= len(mags) - 1:
            return float(freqs[k]), float(mags[k])

        alpha = float(mags[k - 1])
        beta  = float(mags[k])
        gamma = float(mags[k + 1])

        denom = alpha - 2.0 * beta + gamma
        if abs(denom) < 1e-12:
            return float(freqs[k]), float(beta)

        delta = 0.5 * (alpha - gamma) / denom
        delta = max(-0.5, min(0.5, delta))

        delta_f = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 24.414
        interp_freq = float(freqs[k] + delta * delta_f)
        interp_mag = float(beta - 0.25 * (alpha - gamma) * delta)

        return interp_freq, interp_mag

    def close(self):
        """Free allocated CMA buffer."""
        if hasattr(self, "_buffer") and self._buffer is not None:
            try:
                self._buffer.close()
                self._buffer = None
            except Exception:
                pass

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()