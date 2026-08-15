from typing import Tuple, Optional
import numpy as np
from pynq import allocate

class StreamingFFT:
    """
    High-level driver for the PL-accelerated 2048-point FFT & CORDIC Magnitude Engine.
    
    Streams pre-calculated magnitude bins from the FPGA via AXI DMA (axi_dma_1),
    formats the single-sided spectrum (0 Hz to 500 kHz), and provides logarithmic
    (dBV / dBFS) and linear power spectrum transformations.
    """

    def __init__(self, overlay, fft_points: int = 2048, sample_rate_hz: float = 1_000_000.0):
        """
        Initialize the FFT DMA controller from the loaded PYNQ overlay.
        
        :param overlay: Loaded pynq.Overlay or OscilloscopeOverlay instance.
        :param fft_points: FFT transform length matching the PL xfft core (default: 2048).
        :param sample_rate_hz: ADC sampling frequency (default: 1.0 MSPS).
        """
        self.fft_points = fft_points
        self.sample_rate_hz = float(sample_rate_hz)
        self.num_bins = fft_points // 2  # Single-sided spectrum bins (1024)
        
        # Pre-compute frequency axis grid (0 to 500 kHz for 1 MSPS @ 2048 pts)
        self.freq_axis = np.arange(self.num_bins) * (self.sample_rate_hz / self.fft_points)

        # Dynamic binding to axi_dma_1 (by name or base address 0x40410000)
        if hasattr(overlay, "axi_dma_1"):
            self.dma = overlay.axi_dma_1
        else:
            # Fallback: search IP dict for secondary DMA or address 0x40410000
            dma_blocks = [
                getattr(overlay, ip) for ip, details in overlay.ip_dict.items()
                if "dma" in ip.lower() and details.get("phys_addr") == 0x40410000
            ]
            if dma_blocks:
                self.dma = dma_blocks[0]
            else:
                all_dmas = [ip for ip in overlay.ip_dict.keys() if "dma" in ip.lower()]
                if len(all_dmas) > 1:
                    # Choose the second DMA engine
                    self.dma = getattr(overlay, all_dmas[1])
                elif all_dmas:
                    self.dma = getattr(overlay, all_dmas[0])
                else:
                    raise RuntimeError("No AXI DMA block found for FFT stream (axi_dma_1).")

        # Allocate contiguous CMA buffer ONCE for fast real-time streaming
        self._buffer = allocate(shape=(self.fft_points,), dtype="u2")

    def capture_raw(self) -> np.ndarray:
        """
        Triggers a high-speed S2MM DMA transfer of 2048 magnitude points from PL.
        Blocks until the frame completes (asserted by CORDIC/FFT TLAST).
        """
        self.dma.recvchannel.transfer(self._buffer)
        self.dma.recvchannel.wait()
        return np.array(self._buffer, copy=True)

    def capture_spectrum(self, unit: str = "dBV", ref_voltage: float = 3.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Captures a hardware FFT frame and returns (frequencies, magnitudes).
        
        :param unit: Output unit: 'dBV', 'dBFS', or 'Linear' (Volts Peak).
        :param ref_voltage: Full-scale reference voltage of ADC (default: 3.3V).
        :return: Tuple of (freq_axis_hz, magnitude_array) of length 1024.
        """
        raw_full = self.capture_raw()
        
        # 1. Take single-sided spectrum (first N/2 bins from 0 to Nyquist)
        raw_half = raw_full[:self.num_bins].astype(np.float64)

        # 2. Scale raw 16-bit CORDIC magnitude to equivalent peak Volts
        # Scaled xfft + CORDIC translate normalized to ADC range (0V - 3.3V)
        linear_volts = (raw_half / 32768.0) * ref_voltage

        # 3. Convert to requested unit
        unit_clean = unit.strip().upper()
        if unit_clean == "DBV":
            # 20 * log10(V / 1V RMS equivalent)
            safe_linear = np.maximum(linear_volts, 1e-6)
            magnitudes = 20.0 * np.log10(safe_linear)
        elif unit_clean == "DBFS":
            # Relative to full scale (0 dBFS max)
            safe_raw = np.maximum(raw_half, 1.0)
            magnitudes = 20.0 * np.log10(safe_raw / 65535.0)
        elif unit_clean == "LINEAR":
            magnitudes = linear_volts
        else:
            raise ValueError(f"Invalid unit '{unit}'. Choose from: 'dBV', 'dBFS', 'Linear'.")

        return self.freq_axis, magnitudes

    @staticmethod
    def get_peak_frequency(freqs: np.ndarray, mags: np.ndarray, min_freq_hz: float = 1000.0) -> Tuple[float, float]:
        """
        Finds the fundamental/dominant frequency and peak magnitude,
        ignoring near-DC components below min_freq_hz.
        
        :return: (peak_frequency_hz, peak_magnitude)
        """
        # Exclude DC and low-frequency offset bins if requested
        valid_indices = np.where(freqs >= min_freq_hz)[0]
        if len(valid_indices) == 0:
            peak_idx = np.argmax(mags)
        else:
            peak_idx = valid_indices[np.argmax(mags[valid_indices])]

        return float(freqs[peak_idx]), float(mags[peak_idx])

    def close(self):
        """Free the allocated contiguous memory (CMA) buffer."""
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