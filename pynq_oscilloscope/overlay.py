from pathlib import Path
from typing import Union, Optional, Tuple
import numpy as np
from pynq import Overlay

from pynq_oscilloscope.loader import HardwareLoader
from pynq_oscilloscope.xadc_dma import StreamingXADC
from pynq_oscilloscope.fft_dma import StreamingFFT
from pynq_oscilloscope.hw_trigger import HardwareTrigger
from pynq_oscilloscope.ad3_wavegen import AD3SignalGenerator

class OscilloscopeOverlay(Overlay):
    """
    Unified Custom Overlay for the PYNQ-Z2 1 MSPS Oscilloscope & Spectrum Analyzer.
    
    Subclasses pynq.Overlay to provide unified access to:
      • self.trigger   -> HardwareTrigger (AXI-Lite edge & threshold registers)
      • self.xadc      -> StreamingXADC (1 MSPS DMA Stream receiver, 16,384 samples)
      • self.fft       -> StreamingFFT (PL-accelerated 2048-point FFT & CORDIC Magnitude)
      • self.wavegen   -> AD3SignalGenerator (Digilent AD3 waveform generator)
      • self.dashboard() -> Launches multi-tab Plotly + IPywidgets UI (Scope / Spectrum / Dual)
    """

    def __init__(
        self,
        bitfile_name: Optional[Union[str, Path]] = None,
        version: Optional[str] = None,
        packet_size: int = 16384,
        fft_points: int = 2048,
        **kwargs
    ):
        """
        Initialize the Oscilloscope & Spectrum Analyzer Overlay.

        :param bitfile_name: Path to a local .bit file (e.g. './pynq_z2.bit').
                             If None, automatically detects board and fetches pinned release.
        :param version: Release tag override (e.g. 'v1.2.0-rc1').
        :param packet_size: Number of time-domain samples per DMA frame (Default: 16,384).
        :param fft_points: Number of FFT points computed in PL (Default: 2048).
        """
        # Resolve bitstream path
        if bitfile_name is None:
            resolved_bit = str(HardwareLoader.get_overlay_path(version=version))
        else:
            resolved_bit = str(Path(bitfile_name).resolve())

        # Program FPGA fabric via base pynq.Overlay
        super().__init__(resolved_bit, **kwargs)

        # Initialize sub-drivers
        self.packet_size = packet_size
        self.fft_points = fft_points
        
        self.trigger = HardwareTrigger(self)
        self.xadc = StreamingXADC(self, default_packet_size=packet_size)
        self.fft = StreamingFFT(self, fft_points=fft_points)
        self.wavegen = AD3SignalGenerator()

    def capture_both(
        self,
        unit: str = "dBV",
        crop_startup_samples: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Synchronously captures both Time-Domain and Frequency-Domain frames without deadlock.
        Primes both S2MM DMA channels simultaneously before waiting for hardware burst completion.
        
        :param unit: Spectrum magnitude unit ('dBV', 'dBFS', 'Linear').
        :param crop_startup_samples: Number of startup edge samples to discard from time series.
        :return: (voltages_time, freqs_hz, mags_freq)
        """
        # 1. Queue both DMA channels concurrently (Non-blocking)
        self.xadc.dma.recvchannel.transfer(self.xadc._buffer)
        self.fft.dma.recvchannel.transfer(self.fft._buffer)

        # 2. Wait for both transfers to complete
        self.xadc.dma.recvchannel.wait()
        self.fft.dma.recvchannel.wait()

        # 3. Process Time-Domain raw data (12-bit left-aligned to 16 bits -> 0V to 3.3V)
        raw_time = np.array(self.xadc._buffer)
        voltages = (raw_time >> 4) * (3.3 / 4095.0)
        if crop_startup_samples > 0 and len(voltages) > crop_startup_samples:
            voltages = voltages[crop_startup_samples:]

        # 4. Process Frequency-Domain raw data (single-sided 1024 bins from CORDIC)
        raw_fft = np.array(self.fft._buffer, copy=True)
        raw_half = raw_fft[:self.fft.num_bins].astype(np.float64)
        linear_volts = (raw_half / 32768.0) * 3.3

        unit_clean = unit.strip().upper()
        if unit_clean == "DBV":
            mags = 20.0 * np.log10(np.maximum(linear_volts, 1e-6))
        elif unit_clean == "DBFS":
            mags = 20.0 * np.log10(np.maximum(raw_half, 1.0) / 65535.0)
        elif unit_clean == "LINEAR":
            mags = linear_volts
        else:
            raise ValueError(f"Invalid unit '{unit}'. Choose from: 'dBV', 'dBFS', 'Linear'.")

        return voltages, self.fft.freq_axis, mags

    def capture(self, crop_startup_samples: int = 8) -> np.ndarray:
        """Trigger a Time-Domain DMA capture and return a NumPy array of voltages."""
        voltages, _, _ = self.capture_both(crop_startup_samples=crop_startup_samples)
        return voltages

    def capture_fft(self, unit: str = "dBV") -> Tuple[np.ndarray, np.ndarray]:
        """Trigger a Frequency-Domain DMA capture and return (frequencies_hz, magnitudes)."""
        _, freqs, mags = self.capture_both(unit=unit)
        return freqs, mags

    def dashboard(self, display_window: int = 1024):
        """
        Instantiate and display the interactive multi-tab Dashboard.
        """
        from pynq_oscilloscope.dashboard import OscilloscopeDashboard
        dash = OscilloscopeDashboard(
            overlay=self,
            packet_size=self.packet_size,
            fft_points=self.fft_points,
            display_window=display_window
        )
        dash.display()
        return dash

    def close(self):
        """Cleanly release all CMA memory buffers and stop active wavegen."""
        if hasattr(self, "xadc") and self.xadc is not None:
            self.xadc.close()
        if hasattr(self, "fft") and self.fft is not None:
            self.fft.close()
        if hasattr(self, "wavegen") and self.wavegen is not None:
            self.wavegen.stop()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()