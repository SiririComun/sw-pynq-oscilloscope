from pathlib import Path
from typing import Union, Optional, Tuple
import time
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
    """

    def __init__(
        self,
        bitfile_name: Optional[Union[str, Path]] = None,
        version: Optional[str] = None,
        packet_size: int = 2048,
        fft_points: int = 2048,
        **kwargs
    ):
        if bitfile_name is None:
            resolved_bit = str(HardwareLoader.get_overlay_path(version=version))
        else:
            resolved_bit = str(Path(bitfile_name).resolve())

        super().__init__(resolved_bit, **kwargs)

        self.packet_size = packet_size
        self.fft_points = fft_points
        
        self.trigger = HardwareTrigger(self)
        self.xadc = StreamingXADC(self, default_packet_size=packet_size)
        self.fft = StreamingFFT(self, fft_points=fft_points)
        self.wavegen = AD3SignalGenerator()

    def capture_both(
        self,
        unit: str = "dBV",
        crop_startup_samples: int = 0,
        timeout: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Synchronously captures both Time-Domain and Frequency-Domain frames using the Arm-on-Demand sequence.
        """
        # Ensure channels are idle before arming
        t_wait = time.time()
        while not self.xadc.dma.recvchannel.idle or not self.fft.dma.recvchannel.idle:
            time.sleep(0.001)
            if time.time() - t_wait > timeout:
                try:
                    self.xadc.dma.mmio.write(0x30, 0x04)
                    self.fft.dma.mmio.write(0x30, 0x04)
                    time.sleep(0.01)
                    self.xadc.dma.recvchannel.start()
                    self.fft.dma.recvchannel.start()
                except Exception:
                    pass
                break

        # 1. Queue both DMAs FIRST
        self.xadc.dma.recvchannel.transfer(self.xadc._buffer)
        self.fft.dma.recvchannel.transfer(self.fft._buffer)

        # 2. Arm Hardware Trigger
        self.trigger.arm()

        # 3. Wait for Hardware Completion
        self.xadc.dma.recvchannel.wait()
        self.fft.dma.recvchannel.wait()

        # 4. Process Time-Domain data
        raw_time = np.array(self.xadc._buffer)
        voltages = (raw_time >> 4) * (3.3 / 4095.0)
        if crop_startup_samples > 0 and len(voltages) > crop_startup_samples:
            voltages = voltages[crop_startup_samples:]

        # 5. Process Frequency-Domain data
        raw_fft = np.array(self.fft._buffer, copy=True)
        raw_half = raw_fft[:self.fft.num_bins].astype(np.float64)
        
        unit_clean = unit.strip().upper()
        if unit_clean == "LINEAR":
            mags = (raw_half / 2048.0) * (3.3 / 4095.0) * 1000.0
        elif unit_clean == "DBFS":
            mags = 20.0 * np.log10(np.maximum(raw_half, 1.0) / 65535.0)
        else:
            linear_v = (raw_half / 2048.0) * (3.3 / 4095.0)
            mags = 20.0 * np.log10(np.maximum(linear_v, 1e-6))

        return voltages, self.fft.freq_axis, mags

    def capture(self, crop_startup_samples: int = 8) -> np.ndarray:
        voltages, _, _ = self.capture_both(crop_startup_samples=crop_startup_samples)
        return voltages

    def capture_fft(self, unit: str = "dBV") -> Tuple[np.ndarray, np.ndarray]:
        _, freqs, mags = self.capture_both(unit=unit)
        return freqs, mags

    def dashboard(self, display_window: int = 1024):
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