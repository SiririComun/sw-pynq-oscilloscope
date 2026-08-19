"""
pynq_oscilloscope.overlay: Unified Custom Overlay for the PYNQ-Z2 Multi-Regime Oscilloscope & Audio Spectrum Analyzer.
Features runtime profile switching (Lab Scope, Audio, Speech, Bass Zoom), non-blocking DMA polling, and Jupyter audio playback.
"""

from pathlib import Path
from typing import Union, Optional, Tuple, Dict
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
    Unified Custom Overlay for the PYNQ-Z2 Multi-Regime Oscilloscope & Audio Spectrum Analyzer.
    """

    PROFILES = {
        "oscilloscope": {"m": 1,  "n": 2048, "fs_per_ch": 500_000.0, "desc": "Wideband Lab Scope (0 - 250 kHz)"},
        "audio":        {"m": 10, "n": 2048, "fs_per_ch": 50_000.0,  "desc": "Full-Band Audio (0 - 25 kHz, Δf=24.4Hz)"},
        "speech":       {"m": 20, "n": 2048, "fs_per_ch": 25_000.0,  "desc": "Speech / Vocal Band (0 - 12.5 kHz, Δf=12.2Hz)"},
        "bass_zoom":    {"m": 50, "n": 2048, "fs_per_ch": 10_000.0,  "desc": "Deep Bass Zoom (0 - 5 kHz, Δf=4.88Hz)"}
    }

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
        self.current_profile = "audio"
        self.fs_per_ch = 50_000.0
        
        self.trigger = HardwareTrigger(self)
        self.xadc = StreamingXADC(self, default_packet_size=packet_size)
        self.fft = StreamingFFT(self, fft_points=fft_points, sample_rate_hz=self.fs_per_ch)
        self.wavegen = AD3SignalGenerator()

        # Apply default Audio profile (M=10, N=2048)
        self.set_profile("audio")

    # =========================================================================
    # 1. Multi-Regime Profile Switcher
    # =========================================================================

    def set_profile(
        self,
        mode: str = "audio",
        decimation: Optional[int] = None,
        fft_size: Optional[int] = None,
        packet_size: Optional[int] = None
    ) -> Dict:
        """
        Dynamically configures Decimator (M), FFT Length (N), and Packetizer boundaries.

        :param mode: Profile name: 'oscilloscope', 'audio', 'speech', or 'bass_zoom'.
        :param decimation: Manual override for decimation factor M (1, 10, 20, 50).
        :param fft_size: Manual override for FFT transform length N (512, 1024, 2048).
        :param packet_size: Manual override for DMA packet size.
        """
        mode_clean = mode.lower().strip()
        base_cfg = self.PROFILES.get(mode_clean, self.PROFILES["audio"])
        
        m_val = decimation if decimation is not None else base_cfg["m"]
        n_val = fft_size if fft_size is not None else base_cfg["n"]
        pkt_val = packet_size if packet_size is not None else n_val

        # 1. Update Hardware Registers
        self.trigger.set_decimation(m_val)
        self.trigger.set_fft_config(n_points=n_val, forward=True)
        self.trigger.set_packet_size(pkt_val)

        # 2. Update Python Driver State
        self.current_profile = mode_clean
        self.packet_size = pkt_val
        self.fft_points = n_val
        self.fs_per_ch = 500_000.0 / m_val

        # 3. Update FFT Driver Grid
        self.fft.fft_points = n_val
        self.fft.num_bins = n_val // 2
        self.fft.sample_rate_hz = self.fs_per_ch
        self.fft.delta_f = self.fs_per_ch / n_val
        self.fft.freq_axis = np.arange(self.fft.num_bins) * self.fft.delta_f

        info = {
            "mode": mode_clean,
            "decimation_M": m_val,
            "sample_rate_hz": self.fs_per_ch,
            "time_window_ms": ((pkt_val // 2) / self.fs_per_ch) * 1000.0,
            "fft_points_N": n_val,
            "delta_f_hz": self.fft.delta_f,
            "max_frequency_hz": self.fs_per_ch / 2.0
        }
        return info

    def get_profile_info(self) -> Dict:
        """Returns the active operating parameters."""
        pkt = self.trigger.get_packet_size()
        return {
            "mode": self.current_profile,
            "decimation_M": self.trigger.get_decimation(),
            "sample_rate_hz": self.fs_per_ch,
            "time_window_ms": ((pkt // 2) / self.fs_per_ch) * 1000.0,
            "fft_points_N": self.trigger.get_fft_length(),
            "delta_f_hz": self.fft.delta_f,
            "max_frequency_hz": self.fs_per_ch / 2.0
        }

    # =========================================================================
    # 2. Synchronized Audio & Stereo Capture (Hardened Non-Blocking Polling)
    # =========================================================================

    def capture_stereo(
        self,
        crop_startup_samples: int = 8,
        timeout: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Captures simultaneous dual-channel decimated time-domain waveforms (Ch1 on A0, Ch2 on A1)
        with non-blocking timeout polling.
        """
        # 1. Initialize XADC continuous sequence
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
            time.sleep(0.01)

        buf_time.close()
        raise TimeoutError(f"Capture timed out after {timeout} seconds. Check trigger threshold or mode.")

    def capture(self, crop_startup_samples: int = 8) -> np.ndarray:
        """Captures Channel 1 (A0)."""
        v_ch1, _ = self.capture_stereo(crop_startup_samples=crop_startup_samples)
        return v_ch1

    def capture_fft(self, unit: str = "dBV") -> Tuple[np.ndarray, np.ndarray]:
        """Captures hardware decimated audio FFT spectrum from Channel 1."""
        return self.fft.capture_spectrum(unit=unit)

    # =========================================================================
    # 3. Jupyter Audio Playback
    # =========================================================================

    def play_audio(self, channel: int = 1, custom_data: Optional[np.ndarray] = None):
        """
        Plays captured microphone audio directly inside the Jupyter Notebook.
        
        :param channel: 1 for Mic 1 (A0), 2 for Mic 2 (A1).
        :param custom_data: Optional numpy array of audio samples to play instead.
        """
        try:
            from IPython.display import Audio, display
        except ImportError:
            raise RuntimeError("IPython is required for audio playback.")

        if custom_data is not None:
            audio_samples = custom_data
        else:
            v_a0, v_a1 = self.capture_stereo()
            audio_samples = v_a0 if channel == 1 else v_a1

        # Remove DC baseline and normalize for audio output
        ac_signal = audio_samples - np.mean(audio_samples)
        max_val = np.max(np.abs(ac_signal))
        if max_val > 1e-4:
            normalized_audio = ac_signal / max_val
        else:
            normalized_audio = ac_signal

        display(Audio(normalized_audio, rate=int(self.fs_per_ch)))

    # =========================================================================
    # 4. Interactive Dashboards
    # =========================================================================

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