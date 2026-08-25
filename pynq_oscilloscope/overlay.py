"""
pynq_oscilloscope.overlay: Unified Custom Overlay for the PYNQ-Z2
Dual-Channel Multi-Regime Oscilloscope, Spectrum Analyzer & Audio Filter Engine.
"""

from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Any
import time
import numpy as np
from pynq import Overlay, allocate

from pynq_oscilloscope.loader import HardwareLoader
from pynq_oscilloscope.xadc_dma import StreamingXADC
from pynq_oscilloscope.fft_dma import StreamingFFT
from pynq_oscilloscope.hw_trigger import HardwareTrigger
from pynq_oscilloscope.hw_filter import HardwareFilter
from pynq_oscilloscope.ad3_wavegen import AD3SignalGenerator
from pynq_oscilloscope.analytics import AcousticAnalytics
from pynq_oscilloscope.audio_utils import (
    process_raw_recording,
    audio_to_wav_bytes,
    remove_dc_offset,
    normalize_audio,
)


class OscilloscopeOverlay(Overlay):
    """
    Unified Custom Overlay for the PYNQ-Z2 Dual-Channel Multi-Regime Oscilloscope.
    Encapsulates XADC DMA, FFT DMA, Filtered DMA, Hardware Trigger, Spectral Mask,
    AD3 Wavegen, and Acoustic Analytics.
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
        self.filter = HardwareFilter(self, sample_rate_hz=self.sample_rate_hz, fft_points=self.fft_points)
        self.xadc = StreamingXADC(self, default_packet_size=self.packet_size)
        self.fft = StreamingFFT(self, fft_points=self.fft_points, sample_rate_hz=self.sample_rate_hz)
        self.wavegen = AD3SignalGenerator()
        self.analytics = AcousticAnalytics(sample_rate_hz=self.sample_rate_hz)

        # Bind Filtered Time DMA (axi_dma_2 @ 0x40420000)
        if hasattr(self, "axi_dma_2"):
            self.dma_filtered = self.axi_dma_2
        else:
            dma_blocks = [
                getattr(self, ip) for ip, details in self.ip_dict.items()
                if "dma" in ip.lower() and details.get("phys_addr") == 0x40420000
            ]
            self.dma_filtered = dma_blocks[0] if dma_blocks else None

        # Pre-allocate contiguous CMA buffer for Filtered Time stream
        self._buffer_filtered = allocate(shape=(self.fft_points,), dtype="u2")

        # Apply initial operating regime profile
        self.set_profile(profile)

    # -------------------------------------------------------------------------
    # Profile & Hardware Routing Configuration
    # -------------------------------------------------------------------------

    def set_profile(self, profile_name: str) -> Dict[str, Any]:
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
        self.filter.update_configuration(sample_rate_hz=self.sample_rate_hz, fft_points=self.fft_points)
        self.analytics.sample_rate_hz = self.sample_rate_hz

        return self.get_profile_info(profile_name)

    def get_profile_info(self, profile_name: Optional[str] = None) -> Dict[str, Any]:
        """Returns details and parameters for the active or requested profile."""
        prof_name = profile_name or self.current_profile
        prof = self.PROFILES.get(prof_name, self.PROFILES["oscilloscope"]).copy()
        prof["profile_name"] = prof_name
        prof["time_window_ms"] = (prof["packet_size"] / 2.0 / prof["sample_rate_hz"]) * 1000.0
        prof["nyquist_hz"] = prof["sample_rate_hz"] / 2.0
        return prof

    def set_trigger_source(self, channel: int = 1):
        """Selects hardware edge trigger source: 1 for A0 (Vaux1), 2 for A1 (Vaux9)."""
        self.trigger.set_trigger_source(channel=channel)

    def set_fft_source(self, channel: int = 1):
        """Routes a clean single-channel stream to xfft_0: 1 for A0, 2 for A1."""
        self.trigger.set_fft_source(channel=channel)

    @property
    def fs_per_ch(self) -> float:
        """Sampling frequency per channel in Hertz."""
        return self.sample_rate_hz

    # -------------------------------------------------------------------------
    # Capture APIs (Raw Time, Stereo, FFT & Filtered Time)
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
        """Captures simultaneous synchronized physical voltages (Channel 1 A0, Channel 2 A1)."""
        return self.xadc.capture_stereo()

    def capture_fft(self, unit: str = "dBV", window: str = "hann") -> Tuple[np.ndarray, np.ndarray]:
        """Captures the single-sided spectrum (frequencies, magnitudes) from the selected FFT channel."""
        return self.fft.capture_spectrum(unit=unit, window=window)

    def capture_filtered_time(self, crop_startup_samples: int = 0) -> np.ndarray:
        """
        Captures the reconstructed, filtered time-domain waveform from axi_dma_2 (after PL IFFT).
        Returns physical voltages in range 0.0V - 3.3V.
        """
        if self.dma_filtered is None:
            raise RuntimeError("Filtered Time DMA (axi_dma_2) not found in hardware overlay.")

        self.dma_filtered.recvchannel.transfer(self._buffer_filtered)
        self.dma_filtered.recvchannel.wait()

        raw_samples = np.array(self._buffer_filtered, copy=True)
        voltages = (raw_samples >> 4) * (3.3 / 4095.0)

        if crop_startup_samples > 0 and len(voltages) > crop_startup_samples:
            voltages = voltages[crop_startup_samples:]

        return voltages

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

        # 2. Arm Trigger in Auto Mode
        self.trigger.arm()

        # 3. Wait for hardware completion
        self.xadc.dma.recvchannel.wait()
        self.fft.dma.recvchannel.wait()

        # 4. Process stereo samples
        raw_interleaved = np.array(self.xadc._buffer)
        raw_ch1 = raw_interleaved[0::2]
        raw_ch2 = raw_interleaved[1::2]
        v_a0 = (raw_ch1 >> 4) * (3.3 / 4095.0)
        v_a1 = (raw_ch2 >> 4) * (3.3 / 4095.0)

        if crop_startup_samples > 0 and len(v_a0) > crop_startup_samples:
            v_a0 = v_a0[crop_startup_samples:]
            v_a1 = v_a1[crop_startup_samples:]

        # 5. Process Spectrum
        freqs, mags = self.fft.process_buffer(self.fft._buffer, unit=unit, window=window)

        return v_a0, v_a1, freqs, mags

    def capture_all(
        self,
        unit: str = "dBV",
        window: str = "hann",
        crop_startup_samples: int = 0,
        timeout: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Synchronously captures all 3 hardware DMA streams in parallel:
        Returns: (voltages_a0, voltages_a1, voltages_filtered, freqs_hz, mags)
        """
        if self.dma_filtered is None:
            v_a0, v_a1, freqs, mags = self.capture_both(unit=unit, window=window, crop_startup_samples=crop_startup_samples, timeout=timeout)
            return v_a0, v_a1, v_a0, freqs, mags

        # Ensure all channels are idle before queuing
        t_wait = time.time()
        while not self.xadc.dma.recvchannel.idle or not self.fft.dma.recvchannel.idle or not self.dma_filtered.recvchannel.idle:
            time.sleep(0.001)
            if time.time() - t_wait > timeout:
                try:
                    self.xadc.dma.mmio.write(0x30, 0x04)
                    self.fft.dma.mmio.write(0x30, 0x04)
                    self.dma_filtered.mmio.write(0x30, 0x04)
                    time.sleep(0.01)
                    self.xadc.dma.recvchannel.start()
                    self.fft.dma.recvchannel.start()
                    self.dma_filtered.recvchannel.start()
                except Exception:
                    pass
                break

        # 1. Queue all 3 DMAs concurrently
        self.xadc.dma.recvchannel.transfer(self.xadc._buffer)
        self.fft.dma.recvchannel.transfer(self.fft._buffer)
        self.dma_filtered.recvchannel.transfer(self._buffer_filtered)

        # 2. Arm Hardware Trigger
        self.trigger.arm()

        # 3. Wait for all transfers to complete
        self.xadc.dma.recvchannel.wait()
        self.fft.dma.recvchannel.wait()
        self.dma_filtered.recvchannel.wait()

        # 4. Process Raw Time Stereo
        raw_interleaved = np.array(self.xadc._buffer)
        raw_ch1 = raw_interleaved[0::2]
        raw_ch2 = raw_interleaved[1::2]
        v_a0 = (raw_ch1 >> 4) * (3.3 / 4095.0)
        v_a1 = (raw_ch2 >> 4) * (3.3 / 4095.0)

        # 5. Process Filtered Time
        raw_filt = np.array(self._buffer_filtered)
        v_filt = (raw_filt >> 4) * (3.3 / 4095.0)

        if crop_startup_samples > 0 and len(v_a0) > crop_startup_samples:
            v_a0 = v_a0[crop_startup_samples:]
            v_a1 = v_a1[crop_startup_samples:]
            v_filt = v_filt[crop_startup_samples:]

        # 6. Process Spectrum
        freqs, mags = self.fft.process_buffer(self.fft._buffer, unit=unit, window=window)

        return v_a0, v_a1, v_filt, freqs, mags

    def capture_stereo_analytic(self) -> Dict[str, Any]:
        """Executes the complete mathematical Acoustic Analytics pipeline."""
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
        filtered: bool = False,
        profile: Optional[str] = None,
        auto_gain: bool = False
    ) -> np.ndarray:
        """
        Records multi-second continuous audio from microphone/AD3 input.

        :param duration_sec: Length of recording in seconds.
        :param channel: 1 for A0, 2 for A1, or 0 for Stereo tuple.
        :param filtered: If True, records the real-time FPGA-filtered stream from axi_dma_2.
        :param profile: Operating profile ('audio', 'speech', 'bass_zoom').
        :param auto_gain: If True, scales volume to maximize dynamic range without clipping.
        :return: Normalized float64 audio array in range [-1.0, 1.0].
        """
        if profile is not None:
            self.set_profile(profile)

        fs = self.sample_rate_hz
        target_samples = int(duration_sec * fs)

        if filtered:
            if self.dma_filtered is None:
                raise RuntimeError("Filtered Time DMA (axi_dma_2) not available.")
            
            num_frames = int(np.ceil(target_samples / self.fft_points))
            out_raw = np.empty(num_frames * self.fft_points, dtype=np.uint16)
            buf = allocate(shape=(self.fft_points,), dtype="u2")

            try:
                for frame_idx in range(num_frames):
                    self.dma_filtered.recvchannel.transfer(buf)
                    self.trigger.arm()
                    self.dma_filtered.recvchannel.wait()
                    offset = frame_idx * self.fft_points
                    out_raw[offset : offset + self.fft_points] = np.array(buf, copy=False)
            finally:
                buf.close()

            voltages = (out_raw >> 4) * (3.3 / 4095.0)
            ac_signal = remove_dc_offset(voltages)
            if len(ac_signal) > target_samples:
                ac_signal = ac_signal[:target_samples]
            return normalize_audio(ac_signal, auto_gain=auto_gain)

        # Standard raw stereo recording
        samples_per_frame_per_ch = self.packet_size // 2
        num_frames = int(np.ceil(target_samples / samples_per_frame_per_ch))

        raw_interleaved = self.xadc.capture_continuous_raw(
            num_frames=num_frames,
            trigger_unit=self.trigger
        )

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
        filtered: bool = False,
        custom_data: Optional[np.ndarray] = None,
        sample_rate_hz: Optional[int] = None,
        auto_gain: bool = True
    ):
        """
        Records live audio or takes custom audio data, encodes to in-memory WAV,
        and renders the interactive Jupyter HTML5 audio player widget.
        """
        from IPython.display import display, Audio

        if custom_data is not None:
            audio_data = custom_data
            sr = int(sample_rate_hz or self.sample_rate_hz)
        else:
            audio_data = self.record_audio(
                duration_sec=duration_sec,
                channel=channel,
                filtered=filtered,
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
        from pynq_oscilloscope.audio_dashboard import AudioDashboard
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
        if hasattr(self, "_buffer_filtered") and self._buffer_filtered is not None:
            try:
                self._buffer_filtered.close()
                self._buffer_filtered = None
            except Exception:
                pass
        if hasattr(self, "wavegen") and self.wavegen is not None:
            self.wavegen.stop()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()