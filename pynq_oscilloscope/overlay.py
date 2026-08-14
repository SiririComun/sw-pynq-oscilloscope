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

    def capture(self, crop_startup_samples: int = 8) -> np.ndarray:
        """Trigger a Time-Domain DMA capture and return a NumPy array of voltages."""
        return self.xadc.capture(crop_startup_samples=crop_startup_samples)

    def capture_fft(self, unit: str = "dBV") -> Tuple[np.ndarray, np.ndarray]:
        """Trigger a Frequency-Domain DMA capture and return (frequencies_hz, magnitudes)."""
        return self.fft.capture_spectrum(unit=unit)

    def dashboard(self, display_window: int = 1024):
        """
        Instantiate and display the interactive multi-tab Dashboard.
        """
        from pynq_oscilloscope.dashboard import OscilloscopeDashboard
        dash = OscilloscopeDashboard(
            overlay=self,
            packet_size=self.packet_size,
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