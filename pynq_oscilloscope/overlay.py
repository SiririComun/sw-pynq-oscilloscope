"""
pynq_oscilloscope.overlay: Unified Custom Overlay for the PYNQ-Z2 Oscilloscope & Audio Spectrum Analyzer.
"""

from pathlib import Path
from typing import Union, Optional, Tuple
import time
import numpy as np
from pynq import Overlay, allocate

from pynq_oscilloscope.loader import HardwareLoader
from pynq_oscilloscope.xadc_dma import StreamingXADC
from pynq_oscilloscope.fft_dma import StreamingFFT
from pynq_oscilloscope.hw_trigger import HardwareTrigger
from pynq_oscilloscope.ad3_wavegen import AD3SignalGenerator
from pynq_oscilloscope.dashboard import OscilloscopeDashboard
from pynq_oscilloscope.audio_dashboard import AudioDashboard


class OscilloscopeOverlay(Overlay):
    """
    Unified Custom Overlay for the PYNQ-Z2 Dual-Channel 1 MSPS Oscilloscope & 50 kSPS Audio Spectrum Analyzer.
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
        self.fs_per_ch = 50_000.0
        
        self.trigger = HardwareTrigger(self)
        self.xadc = StreamingXADC(self, default_packet_size=packet_size)
        self.fft = StreamingFFT(self, fft_points=fft_points, sample_rate_hz=self.fs_per_ch)
        self.wavegen = AD3SignalGenerator()

    def capture_stereo(
        self,
        crop_startup_samples: int = 8,
        timeout: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Synchronously captures dual-channel decimated time-domain waveforms (Ch1 on A0, Ch2 on A1)
        using safe non-blocking polling on Time DMA 0 with automatic XADC sequencer initialization.
        """
        # 1. Initialize XADC continuous sequence for Vaux1 (A0) and Vaux9 (A1)
        if hasattr(self, "xadc_wiz_0"):
            self.xadc_wiz_0.mmio.write(0x304, 0x2000)  # DRP 0x41 = Continuous Sequence Mode
            self.xadc_wiz_0.mmio.write(0x320, 0x0000)  # DRP 0x48 = Disable internal channels
            self.xadc_wiz_0.mmio.write(0x324, 0x0202)  # DRP 0x49 = Enable Vaux1 & Vaux9

        # 2. Reset DMA 0
        self.axi_dma_0.mmio.write(0x30, 0x04)
        time.sleep(0.005)
        self.axi_dma_0.recvchannel.start()

        # 3. Queue receive buffer
        buf_time = allocate(shape=(self.packet_size,), dtype="u2")
        self.axi_dma_0.recvchannel.transfer(buf_time)

        # 4. Arm Hardware Trigger
        self.trigger.arm()

        # 5. Non-blocking Poll with Timeout
        start = time.time()
        while time.time() - start < timeout:
            if self.axi_dma_0.recvchannel.idle:
                raw_samples = np.array(buf_time)
                
                # De-interleave: Even = A0 (Ch1), Odd = A1 (Ch2)
                raw_ch1 = raw_samples[0::2]
                raw_ch2 = raw_samples[1::2]

                voltages_ch1 = (raw_ch1 >> 4) * (3.3 / 4095.0)
                voltages_ch2 = (raw_ch2 >> 4) * (3.3 / 4095.0)

                # Crop boundary samples
                if crop_startup_samples > 0 and len(voltages_ch1) > (2 * crop_startup_samples):
                    voltages_ch1 = voltages_ch1[crop_startup_samples:-crop_startup_samples]
                    voltages_ch2 = voltages_ch2[crop_startup_samples:-crop_startup_samples]

                buf_time.close()
                return voltages_ch1, voltages_ch2
            time.sleep(0.005)

        buf_time.close()
        raise TimeoutError(f"Capture timed out after {timeout} seconds. Check trigger threshold or signal.")

    def capture(self, crop_startup_samples: int = 8) -> np.ndarray:
        """Captures Channel 1 (A0)."""
        v_ch1, _ = self.capture_stereo(crop_startup_samples=crop_startup_samples)
        return v_ch1

    def capture_fft(self, unit: str = "dBV") -> Tuple[np.ndarray, np.ndarray]:
        """Captures hardware decimated audio FFT spectrum from Channel 1."""
        return self.fft.capture_spectrum(unit=unit)

    def dashboard(self, display_window: int = 1024):
        """Launches the general Oscilloscope Dashboard."""
        dash = OscilloscopeDashboard(
            overlay=self,
            packet_size=self.packet_size,
            fft_points=self.fft_points,
            display_window=display_window
        )
        dash.display()
        return dash

    def audio_dashboard(self):
        """Launches the dedicated Audio & Microphone Dashboard."""
        dash = AudioDashboard(
            overlay=self,
            packet_size=self.packet_size,
            fft_points=self.fft_points
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