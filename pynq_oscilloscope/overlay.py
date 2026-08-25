"""
pynq_oscilloscope.overlay: Unified Custom Overlay for the PYNQ-Z2
Dual-Channel Multi-Regime Oscilloscope, Spectrum Analyzer & Audio Engine.
"""

from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Any
import time
import numpy as np
from pynq import Overlay

from pynq_oscilloscope.loader import HardwareLoader
from pynq_oscilloscope.xadc_dma import StreamingXADC
from pynq_oscilloscope.fft_dma import StreamingFFT
from pynq_oscilloscope.hw_trigger import HardwareTrigger
from pynq_oscilloscope.ad3_wavegen import AD3SignalGenerator
from pynq_oscilloscope.analytics import AcousticAnalytics
from pynq_oscilloscope.audio_utils import (
    process_raw_recording,
    audio_to_wav_bytes,
)


class OscilloscopeOverlay(Overlay):
    """
    Unified Custom Overlay for the PYNQ-Z2 Dual-Channel Multi-Regime Oscilloscope.
    Encapsulates XADC DMA, FFT DMA, Hardware Trigger, AD3 Wavegen, and Acoustic Analytics.
    """

    PROFILES: Dict[str, Dict[str, Any]] = {
        "oscilloscope": {
            "decim_factor": 1,
            "decim_bits": 0,
            "fft_points": 2048,
            "packet_size": 2048,
            "sample_rate_hz": 500_000.0,
            "desc": "Wideband Lab Scope (500 kSPS, 0 - 250 kHz, df = 244.14 Hz)",
        },
        "audio": {
            "decim_factor": 10,
            "decim_bits": 1,
            "fft_points": 2048,
            "packet_size": 2048,
            "sample_rate_hz": 50_000.0,
            "desc": "Full-Band Audio (50 kSPS, 0 - 25 kHz, df = 24.41 Hz)",
        },
        "speech": {
            "decim_factor": 20,
            "decim_bits": 2,
            "fft_points": 2048,
            "packet_size": 2048,
            "sample_rate_hz": 25_000.0,
            "desc": "Speech / Vocal Formants (25 kSPS, 0 - 12.5 kHz, df = 12.21 Hz)",
        },
        "bass_zoom": {
            "decim_factor": 50,
            "decim_bits": 3,
            "fft_points": 2048,
            "packet_size": 2048,
            "sample_rate_hz": 10_000.0,
            "desc": "Deep Bass Zoom (10 kSPS, 0 - 5 kHz, df = 4.88 Hz)",
        },
    }

    def __init__(
        self,
        bitfile_name: Optional[Union[str, Path]] = None,
        version: Optional[str] = None,
        profile: str = "oscilloscope",
        **kwargs
    ):
        if bitfile_name is None:
            resolved_bit = str(HardwareLoader.get_overlay_path(version=version))
        else:
            resolved_bit = str(Path(bitfile_name).resolve())

        super().__init__(resolved_bit, **kwargs)

        self.current_profile = profile
        prof = self.PROFILES.get(profile, self.PROFILES["oscilloscope"])

        self.packet_size = prof["packet_size"]
        self.fft_points = prof["fft_points"]
        self.sample_rate_hz = prof["sample_rate_hz"]

        # Sub-drivers
        self.trigger = HardwareTrigger(self)
        self.xadc = StreamingXADC(self, default_packet_size=self.packet_size)
        self.fft = StreamingFFT(self, fft_points=self.fft_points, sample_rate_hz=self.sample_rate_hz)
        self.wavegen = AD3SignalGenerator()
        self.analytics = AcousticAnalytics(sample_rate_hz=self.sample_rate_hz)

        # Apply initial operating regime profile
        self.set_profile(profile)

    # -------------------------------------------------------------------------
    # Profile & Hardware Routing Configuration
    # -------------------------------------------------------------------------

    def set_profile(self, profile_name: str):
        """
        Dynamically switches operating regimes on the fly.
        Reconfigures hardware decimation (M), transform length (N), packet size,
        and software driver sampling parameters.
        """
        if profile_name not in self.PROFILES:
            valid_keys = list(self.PROFILES.keys())
            raise ValueError(f"Invalid profile '{profile_name}'. Choose from: {valid_keys}")

        prof = self.PROFILES[profile_name]
        self.current_profile = profile_name
        self.packet_size = prof["packet_size"]
        self.fft_points = prof["fft_points"]
        self.sample_rate_hz = prof["sample_rate_hz"]

        # Configure hardware registers in axis_trigger_unit
        self.trigger.set_decimation(prof["decim_factor"])
        self.trigger.set_fft_length(self.fft_points)
        self.trigger.set_packet_size(self.packet_size)

        # Update software drivers
        self.fft.update_configuration(fft_points=self.fft_points, sample_rate_hz=self.sample_rate_hz)
        self.analytics.sample_rate_hz = self.sample_rate_hz

    def set_trigger_source(self, channel: int = 1):
        """
        Selects hardware edge trigger comparator source:
          • channel = 1: Trigger on Channel 1 (A0 / Vaux1)
          • channel = 2: Trigger on Channel 2 (A1 / Vaux9)
        """
        self.trigger.set_trigger_source(channel=channel)

    def set_fft_source(self, channel: int = 1):
        """
        Routes a single clean stream to the hardware xfft_0 core via axis_channel_demux:
          • channel = 1: Route Channel 1 (A0) to FFT
          • channel = 2: Route Channel 2 (A1) to FFT
        """
        self.trigger.set_fft_source(channel=channel)

    # -------------------------------------------------------------------------
    # Single-Frame Capture API
    # -------------------------------------------------------------------------

    def capture_raw(self) -> np.ndarray:
        """Captures a single raw frame (packet_size words) direct from DMA 0."""
        return self.xadc.capture_raw()

    def capture(self, crop_startup_samples: int = 0) -> np.ndarray:
        """Captures physical voltages from Channel 1 (A0)."""
        v_a0, _ = self.capture_stereo()
        if crop_startup_samples > 0 and len(v_a0) > crop_startup_samples:
            v_a0 = v_a0[crop_startup_samples:]
        return v_a0

    def capture_stereo(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Captures simultaneous, phase-locked physical voltages from both channels:
          • v_a0: Channel 1 (A0 / Vaux1)
          • v_a1: Channel 2 (A1 / Vaux9)
        """
        return self.xadc.capture_stereo()

    def capture_fft(self, unit: str = "dBV", window: str = "hann") -> Tuple[np.ndarray, np.ndarray]:
        """Captures the single-sided spectrum (frequencies, magnitudes) from the selected FFT channel."""
        return self.fft.capture_spectrum(unit=unit, window=window)

    def capture_both(
        self,
        unit: str = "dBV",
        window: str = "hann",
        crop_startup_samples: int = 0,
        timeout: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Synchronously captures both Time-Domain channels and Frequency-Domain spectrum:
        Returns: (voltages_a0, voltages_a1, freqs_hz, mags)
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

        # 2. Arm Hardware Trigger in Auto Mode
        self.trigger.arm()

        # 3. Wait for hardware completion
        self.xadc.dma.recvchannel.wait()
        self.fft.dma.recvchannel.wait()

        # 4. Process Time-Domain stereo samples
        raw_interleaved = np.array(self.xadc._buffer)
        raw_ch1 = raw_interleaved[0::2]
        raw_ch2 = raw_interleaved[1::2]
        v_a0 = (raw_ch1 >> 4) * (3.3 / 4095.0)
        v_a1 = (raw_ch2 >> 4) * (3.3 / 4095.0)

        if crop_startup_samples > 0 and len(v_a0) > crop_startup_samples:
            v_a0 = v_a0[crop_startup_samples:]
            v_a1 = v_a1[crop_startup_samples:]

        # 5. Process Frequency-Domain spectrum
        freqs, mags = self.fft.process_buffer(self.fft._buffer, unit=unit, window=window)

        return v_a0, v_a1, freqs, mags

    def capture_stereo_analytic(self) -> Dict[str, Any]:
        """
        Captures a synchronized stereo frame and executes the complete mathematical
        Acoustic Analytics pipeline (Hilbert analytic envelopes, ILD, instantaneous phase difference).
        """
        v_a0, v_a1 = self.capture_stereo()
        analytic_results = self.analytics.analyze_stereo(v_a0, v_a1)
        analytic_results["raw_a0"] = v_a0
        analytic_results["raw_a1"] = v_a1
        return analytic_results

    # -------------------------------------------------------------------------
    # Multi-Second Audio Recording & Jupyter Playback API
    # -------------------------------------------------------------------------

    def record_audio(
        self,
        duration_sec: float = 3.0,
        channel: int = 1,
        profile: Optional[str] = None,
        auto_gain: bool = False
    ) -> np.ndarray:
        """
        Records multi-second continuous audio from physical microphone/AD3 input
        using ping-pong double-buffered DMA streaming.

        :param duration_sec: Length of recording in seconds (e.g. 1.0 to 10.0s).
        :param channel: 1 for Channel 1 (A0), 2 for Channel 2 (A1), or 0 for Stereo tuple.
        :param profile: Operating profile ('audio', 'speech', 'bass_zoom'). Defaults to current active profile.
        :param auto_gain: If True, automatically scales volume to maximize dynamic range without clipping.
        :return: Normalized float64 audio array in range [-1.0, 1.0].
        """
        if profile is not None:
            self.set_profile(profile)

        fs = self.sample_rate_hz
        target_samples = int(duration_sec * fs)

        # In interleaved stereo mode, each 2048-word frame contains 1024 samples per channel
        samples_per_frame_per_ch = self.packet_size // 2
        num_frames = int(np.ceil(target_samples / samples_per_frame_per_ch))

        # Continuous double-buffered hardware DMA streaming
        raw_interleaved = self.xadc.capture_continuous_raw(
            num_frames=num_frames,
            trigger_unit=self.trigger
        )

        # DSP Conditioning (De-interleaving, voltage conversion, DC removal, normalization)
        audio = process_raw_recording(
            raw_interleaved=raw_interleaved,
            channel=channel,
            target_samples=target_samples,
            auto_gain=auto_gain
        )
        return audio

    def play_audio(
        self,
        duration_sec: float = 3.0,
        channel: int = 1,
        custom_data: Optional[np.ndarray] = None,
        sample_rate_hz: Optional[int] = None,
        auto_gain: bool = True
    ):
        """
        Records live audio or takes custom audio data, encodes to in-memory WAV,
        and renders the interactive Jupyter HTML5 audio player widget.

        :param duration_sec: Duration to record if custom_data is None.
        :param channel: Channel to record (1=A0, 2=A1, 0=Stereo).
        :param custom_data: Optional pre-recorded or filtered float array [-1.0, 1.0].
        :param sample_rate_hz: Sample rate (defaults to active overlay sample rate).
        :param auto_gain: Normalize peak audio volume.
        :return: IPython.display.Audio widget.
        """
        from IPython.display import display, Audio

        if custom_data is not None:
            audio_data = custom_data
            sr = int(sample_rate_hz or self.sample_rate_hz)
        else:
            audio_data = self.record_audio(
                duration_sec=duration_sec,
                channel=channel,
                auto_gain=auto_gain
            )
            sr = int(self.sample_rate_hz)

        num_channels = 2 if (audio_data.ndim == 2 and audio_data.shape[1] == 2) else 1
        wav_bytes = audio_to_wav_bytes(audio_data, sample_rate_hz=sr, num_channels=num_channels)

        player = Audio(data=wav_bytes, rate=sr, autoplay=False)
        display(player)
        return player

    # -------------------------------------------------------------------------
    # Interactive Dashboard Launchers
    # -------------------------------------------------------------------------

    def ad3_dashboard(self, display_window: int = 1024):
        """Launches the Academic Dual Laboratory Scope & Aliasing Explorer."""
        from pynq_oscilloscope.dashboard import OscilloscopeDashboard
        dash = OscilloscopeDashboard(overlay=self, display_window=display_window)
        dash.display()
        return dash

    def audio_dashboard(self, display_window: int = 1024):
        """Launches the Dedicated Passive Microphone Instrument."""
        from pynq_oscilloscope.dashboard import AudioDashboard
        dash = AudioDashboard(overlay=self, display_window=display_window)
        dash.display()
        return dash

    def analytic_dashboard(self, display_window: int = 1024):
        """Launches the Multi-Domain Acoustic Analytics & Spectrogram Engine."""
        from pynq_oscilloscope.analytic_dashboard import AcousticAnalyticDashboard
        dash = AcousticAnalyticDashboard(overlay=self, display_window=display_window)
        dash.display()
        return dash

    def dashboard(self, display_window: int = 1024):
        """Default dashboard launcher (aliases to ad3_dashboard for backwards compatibility)."""
        return self.ad3_dashboard(display_window=display_window)

    # -------------------------------------------------------------------------
    # Clean Shutdown & Context Management
    # -------------------------------------------------------------------------

    def close(self):
        """Clean hardware shutdown and DMA/Wavegen handle release."""
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