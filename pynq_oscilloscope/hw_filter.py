"""
pynq_oscilloscope.hw_filter: High-level driver for the FPGA-based 'axis_spectral_mask' IP.
Provides register-level control over real-time frequency-domain spectral masking and filtering.
"""

from typing import Union, Optional, Tuple, Dict
from pynq import MMIO


class HardwareFilter:
    """
    High-level Python driver for the FPGA-based 'axis_spectral_mask' IP (Base: 0x43C20000).
    
    Controls hardware-accelerated frequency-domain filtering (Lowpass, Highpass, Bandpass, Notch)
    by translating physical frequencies into discrete FFT bin indices.
    """

    # Register Byte Offsets matching axis_spectral_mask.vhd
    REG_CTRL      = 0x00  # [0]: Enable, [2:1]: Mode (00=Lowpass, 01=Highpass, 10=Bandpass, 11=Notch)
    REG_BIN_START = 0x04  # [15:0]: Lower Cutoff Bin index (k_start)
    REG_BIN_STOP  = 0x08  # [15:0]: Upper Cutoff Bin index (k_stop)
    REG_STATUS    = 0x0C  # [0]: Frame Active, [31:16]: Real-time bin index

    # Control Bit Masks
    BIT_FILTER_EN = 1 << 0
    MODE_LOWPASS  = 0 << 1  # Mode "00": Pass k <= k_stop (Lowpass / Bass)
    MODE_HIGHPASS = 1 << 1  # Mode "01": Pass k >= k_start (Highpass / Treble)
    MODE_BANDPASS = 2 << 1  # Mode "10": Pass k_start <= k <= k_stop (Bandpass)
    MODE_NOTCH    = 3 << 1  # Mode "11": Zero k_start <= k <= k_stop (Notch / Bandstop)

    MODE_MAP = {
        "lowpass": MODE_LOWPASS,
        "bass": MODE_LOWPASS,
        "highpass": MODE_HIGHPASS,
        "treble": MODE_HIGHPASS,
        "bandpass": MODE_BANDPASS,
        "notch": MODE_NOTCH,
        "bandstop": MODE_NOTCH,
    }

    REVERSE_MODE_MAP = {
        0: "lowpass",
        1: "highpass",
        2: "bandpass",
        3: "notch",
    }

    def __init__(
        self,
        overlay_or_mmio: Union[object, MMIO],
        sample_rate_hz: float = 50_000.0,
        fft_points: int = 2048
    ):
        self.sample_rate_hz = float(sample_rate_hz)
        self.fft_points = int(fft_points)
        self.num_bins = self.fft_points // 2

        # Bind MMIO from overlay or address
        if isinstance(overlay_or_mmio, MMIO):
            self.mmio = overlay_or_mmio
        elif hasattr(overlay_or_mmio, "axis_spectral_mask_0"):
            self.mmio = overlay_or_mmio.axis_spectral_mask_0.mmio
        elif hasattr(overlay_or_mmio, "ip_dict"):
            mask_ips = [k for k in overlay_or_mmio.ip_dict.keys() if "mask" in k.lower() or "spectral" in k.lower()]
            if mask_ips:
                self.mmio = getattr(overlay_or_mmio, mask_ips[0]).mmio
            else:
                self.mmio = MMIO(0x43C20000, 65536)
        else:
            self.mmio = MMIO(0x43C20000, 65536)

        # Default state: Bypass mode
        self.bypass()

    @property
    def delta_f(self) -> float:
        """Frequency resolution per FFT bin in Hertz."""
        return self.sample_rate_hz / float(self.fft_points)

    def update_configuration(
        self,
        sample_rate_hz: Optional[float] = None,
        fft_points: Optional[int] = None
    ):
        """Updates internal grid parameters when the operating profile changes."""
        if sample_rate_hz is not None:
            self.sample_rate_hz = float(sample_rate_hz)
        if fft_points is not None:
            self.fft_points = int(fft_points)
            self.num_bins = self.fft_points // 2

    # =========================================================================
    # 1. Frequency to Bin Mapping
    # =========================================================================

    def freq_to_bin(self, freq_hz: float) -> int:
        """Converts physical frequency in Hz to nearest valid FFT bin index."""
        clamped_f = max(0.0, min(self.sample_rate_hz / 2.0, float(freq_hz)))
        bin_idx = int(round(clamped_f / self.delta_f))
        return max(0, min(self.num_bins - 1, bin_idx))

    def bin_to_freq(self, bin_idx: int) -> float:
        """Converts FFT bin index to physical center frequency in Hz."""
        return float(bin_idx) * self.delta_f

    # =========================================================================
    # 2. Filter Configuration Methods
    # =========================================================================

    def set_passband(
        self,
        low_hz: float,
        high_hz: float,
        mode: str = "bandpass",
        enable: bool = True
    ):
        """
        Configures the hardware frequency filter cutoffs in physical Hertz.

        :param low_hz: Lower cutoff frequency in Hz.
        :param high_hz: Upper cutoff frequency in Hz.
        :param mode: 'lowpass', 'highpass', 'bandpass', or 'notch'.
        :param enable: If True, engages the filter immediately.
        """
        k_start = self.freq_to_bin(low_hz)
        k_stop = self.freq_to_bin(high_hz)

        # Ensure k_start <= k_stop
        if k_start > k_stop:
            k_start, k_stop = k_stop, k_start

        mode_clean = mode.lower().strip()
        if mode_clean not in self.MODE_MAP:
            raise ValueError(f"Invalid mode '{mode}'. Choose from: {list(self.MODE_MAP.keys())}")

        mode_bits = self.MODE_MAP[mode_clean]

        # Write cutoffs
        self.mmio.write(self.REG_BIN_START, k_start)
        self.mmio.write(self.REG_BIN_STOP, k_stop)

        # Write control register
        ctrl = mode_bits
        if enable:
            ctrl |= self.BIT_FILTER_EN
        self.mmio.write(self.REG_CTRL, ctrl)

    def set_lowpass(self, cutoff_hz: float = 250.0):
        """Configures Lowpass / Bass filter mode (passes 0 Hz up to cutoff_hz)."""
        self.set_passband(low_hz=0.0, high_hz=cutoff_hz, mode="lowpass", enable=True)

    def set_highpass(self, cutoff_hz: float = 1000.0):
        """Configures Highpass / Treble filter mode (passes cutoff_hz up to Nyquist)."""
        self.set_passband(low_hz=cutoff_hz, high_hz=self.sample_rate_hz / 2.0, mode="highpass", enable=True)

    def set_bandpass(self, low_hz: float = 300.0, high_hz: float = 3400.0):
        """Configures Bandpass mode (e.g. Vocal Formants 300 Hz - 3.4 kHz)."""
        self.set_passband(low_hz=low_hz, high_hz=high_hz, mode="bandpass", enable=True)

    def set_notch(self, center_hz: float = 60.0, bandwidth_hz: float = 20.0):
        """Configures Notch / Bandstop mode (e.g. 50/60 Hz mains hum rejection)."""
        half_bw = bandwidth_hz / 2.0
        self.set_passband(low_hz=center_hz - half_bw, high_hz=center_hz + half_bw, mode="notch", enable=True)

    def enable(self):
        """Enables the hardware filter with current settings."""
        ctrl = self.mmio.read(self.REG_CTRL)
        self.mmio.write(self.REG_CTRL, ctrl | self.BIT_FILTER_EN)

    def disable(self):
        """Disables the filter (hardware bypass mode: pass all frequencies)."""
        ctrl = self.mmio.read(self.REG_CTRL)
        self.mmio.write(self.REG_CTRL, ctrl & ~self.BIT_FILTER_EN)

    def bypass(self):
        """Alias for disable."""
        self.disable()

    # =========================================================================
    # 3. Status & Properties
    # =========================================================================

    @property
    def is_enabled(self) -> bool:
        """Returns True if the hardware filter is currently active."""
        return bool(self.mmio.read(self.REG_CTRL) & self.BIT_FILTER_EN)

    @property
    def mode(self) -> str:
        """Returns the active filter mode name."""
        ctrl = self.mmio.read(self.REG_CTRL)
        mode_val = (ctrl >> 1) & 0x03
        return self.REVERSE_MODE_MAP.get(mode_val, "lowpass")

    @property
    def cutoffs(self) -> Tuple[float, float]:
        """Returns the active lower and upper cutoff frequencies in Hz: (low_hz, high_hz)."""
        k_start = self.mmio.read(self.REG_BIN_START) & 0xFFFF
        k_stop = self.mmio.read(self.REG_BIN_STOP) & 0xFFFF
        return self.bin_to_freq(k_start), self.bin_to_freq(k_stop)

    def __repr__(self) -> str:
        status = "ENABLED" if self.is_enabled else "BYPASS"
        mode_str = self.mode.capitalize()
        f_low, f_high = self.cutoffs
        k_start = self.mmio.read(self.REG_BIN_START) & 0xFFFF
        k_stop = self.mmio.read(self.REG_BIN_STOP) & 0xFFFF
        return f"<HardwareFilter: {status}, Mode={mode_str}, Cutoffs=[{f_low:.1f}Hz, {f_high:.1f}Hz] (Bins [{k_start}, {k_stop}])>"