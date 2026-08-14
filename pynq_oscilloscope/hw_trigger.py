from typing import Union, Optional
from pynq import MMIO

class HardwareTrigger:
    """
    High-level Python driver for the FPGA-based 'axis_trigger_unit' IP.
    
    Interfaces via AXI4-Lite registers to configure hardware-level edge detection,
    trigger voltage thresholds, hysteresis, auto-timeouts, and oscilloscope modes.
    """

    # Register Byte Offsets matching axis_trigger_unit.vhd
    REG_CONTROL    = 0x00
    REG_STATUS     = 0x04
    REG_THRESHOLD  = 0x08
    REG_TIMEOUT    = 0x0C
    REG_HYSTERESIS = 0x10

    # Bit masks for REG_CONTROL
    BIT_ARM          = 1 << 0  # Bit 0: Arm trigger unit
    BIT_AUTO_MODE    = 1 << 1  # Bit 1: 1 = Auto Mode, 0 = Normal Mode
    BIT_EDGE_FALLING = 1 << 2  # Bit 2: 0 = Rising Edge, 1 = Falling Edge
    BIT_SINGLE_SHOT  = 1 << 3  # Bit 3: 1 = Single Shot, 0 = Continuous
    BIT_FORCE_TRIG   = 1 << 4  # Bit 4: Software force trigger pulse

    # Bit masks for REG_STATUS
    STATUS_ARMED     = 1 << 0
    STATUS_TRIGGERED = 1 << 1
    STATUS_STREAMING = 1 << 2

    def __init__(self, overlay_or_mmio: Union[object, MMIO], clock_freq_hz: int = 100_000_000):
        """
        Initialize the Hardware Trigger driver from an Overlay object or an MMIO instance.
        """
        self.clock_freq_hz = clock_freq_hz
        self.max_voltage = 3.3

        # Dynamically discover MMIO from overlay or use provided MMIO directly
        if isinstance(overlay_or_mmio, MMIO):
            self.mmio = overlay_or_mmio
        elif hasattr(overlay_or_mmio, "axis_trigger_unit_0"):
            self.mmio = overlay_or_mmio.axis_trigger_unit_0.mmio
        elif hasattr(overlay_or_mmio, "ip_dict"):
            # Search IP dictionary for trigger core
            trigger_ips = [k for k in overlay_or_mmio.ip_dict.keys() if "trigger" in k.lower()]
            if trigger_ips:
                self.mmio = getattr(overlay_or_mmio, trigger_ips[0]).mmio
            else:
                # Fallback to standard base address 0x43C10000
                self.mmio = MMIO(0x43C10000, 65536)
        else:
            self.mmio = MMIO(0x43C10000, 65536)

        # Apply default state: 1.65V Threshold, Auto Mode, Rising Edge, Continuous
        self.configure(
            mode="Auto",
            edge="Rising",
            threshold_volts=1.65,
            timeout_ms=50.0,
            hysteresis_volts=0.02
        )

    # -------------------------------------------------------------------------
    # Core Configuration Methods
    # -------------------------------------------------------------------------

    def configure(
        self,
        mode: str = "Auto",
        edge: str = "Rising",
        threshold_volts: float = 1.65,
        timeout_ms: float = 50.0,
        hysteresis_volts: float = 0.02
    ):
        """
        Configure all hardware trigger settings simultaneously.

        :param mode: Trigger mode: 'Auto', 'Normal', or 'Single'
        :param edge: Trigger edge: 'Rising' or 'Falling'
        :param threshold_volts: Target trigger level in Volts (0.0V - 3.3V)
        :param timeout_ms: Auto-trigger fallback timeout in milliseconds
        :param hysteresis_volts: Digital noise reject band in Volts
        """
        self.set_threshold(threshold_volts)
        self.set_timeout_ms(timeout_ms)
        self.set_hysteresis(hysteresis_volts)
        self.set_edge(edge)
        self.set_mode(mode)

    def set_mode(self, mode: str):
        """Set trigger operating mode: 'Auto', 'Normal', or 'Single'."""
        mode_clean = mode.strip().capitalize()
        ctrl = self.mmio.read(self.REG_CONTROL)

        if mode_clean == "Auto":
            ctrl |= (self.BIT_ARM | self.BIT_AUTO_MODE)
            ctrl &= ~self.BIT_SINGLE_SHOT
        elif mode_clean == "Normal":
            ctrl |= self.BIT_ARM
            ctrl &= ~(self.BIT_AUTO_MODE | self.BIT_SINGLE_SHOT)
        elif mode_clean == "Single":
            ctrl |= (self.BIT_ARM | self.BIT_SINGLE_SHOT)
            ctrl &= ~self.BIT_AUTO_MODE
        else:
            raise ValueError(f"Invalid mode '{mode}'. Choose from: 'Auto', 'Normal', 'Single'.")

        self.mmio.write(self.REG_CONTROL, ctrl)

    def set_edge(self, edge: str):
        """Set trigger slope direction: 'Rising' or 'Falling'."""
        edge_clean = edge.strip().capitalize()
        ctrl = self.mmio.read(self.REG_CONTROL)

        if edge_clean == "Rising":
            ctrl &= ~self.BIT_EDGE_FALLING
        elif edge_clean == "Falling":
            ctrl |= self.BIT_EDGE_FALLING
        else:
            raise ValueError(f"Invalid edge '{edge}'. Choose from: 'Rising' or 'Falling'.")

        self.mmio.write(self.REG_CONTROL, ctrl)

    def set_threshold(self, volts: float):
        """
        Set analog trigger threshold in Volts (0.0V to 3.3V).
        Translates to 12-bit left-aligned ADC code (0x0000 to 0xFFF0).
        """
        clamped_volts = max(0.0, min(self.max_voltage, float(volts)))
        # XADC is 12-bit left-aligned to 16 bits (shifted left by 4)
        raw_12bit = int((clamped_volts / self.max_voltage) * 4095.0)
        raw_code = (raw_12bit & 0xFFF) << 4
        self.mmio.write(self.REG_THRESHOLD, raw_code)

    def get_threshold(self) -> float:
        """Read active threshold voltage from hardware register."""
        raw_code = self.mmio.read(self.REG_THRESHOLD)
        raw_12bit = (raw_code >> 4) & 0xFFF
        return (raw_12bit / 4095.0) * self.max_voltage

    def set_timeout_ms(self, timeout_ms: float):
        """Set timeout in milliseconds for Auto-Trigger mode."""
        cycles = int((float(timeout_ms) / 1000.0) * self.clock_freq_hz)
        self.mmio.write(self.REG_TIMEOUT, max(100, cycles))

    def get_timeout_ms(self) -> float:
        """Read active auto-timeout in milliseconds."""
        cycles = self.mmio.read(self.REG_TIMEOUT)
        return (cycles / self.clock_freq_hz) * 1000.0

    def set_hysteresis(self, volts: float):
        """Set noise rejection band in Volts."""
        clamped_volts = max(0.0, min(0.5, float(volts)))
        raw_12bit = int((clamped_volts / self.max_voltage) * 4095.0)
        raw_code = (raw_12bit & 0xFFF) << 4
        self.mmio.write(self.REG_HYSTERESIS, raw_code)

    # -------------------------------------------------------------------------
    # Arming, Disarming & Manual Triggers
    # -------------------------------------------------------------------------

    def arm(self):
        """Arm the trigger unit."""
        ctrl = self.mmio.read(self.REG_CONTROL)
        self.mmio.write(self.REG_CONTROL, ctrl | self.BIT_ARM)

    def disarm(self):
        """Disarm the trigger unit (halts stream pass-through)."""
        ctrl = self.mmio.read(self.REG_CONTROL)
        self.mmio.write(self.REG_CONTROL, ctrl & ~self.BIT_ARM)

    def force_trigger(self):
        """Manually trigger acquisition via software pulse."""
        ctrl = self.mmio.read(self.REG_CONTROL)
        self.mmio.write(self.REG_CONTROL, ctrl | self.BIT_FORCE_TRIG)

    # -------------------------------------------------------------------------
    # Status Properties
    # -------------------------------------------------------------------------

    @property
    def is_armed(self) -> bool:
        """Return True if hardware trigger is armed and awaiting an edge."""
        return bool(self.mmio.read(self.REG_STATUS) & self.STATUS_ARMED)

    @property
    def is_triggered(self) -> bool:
        """Return True if a valid trigger event occurred and stream is flowing."""
        return bool(self.mmio.read(self.REG_STATUS) & self.STATUS_TRIGGERED)

    def __repr__(self) -> str:
        ctrl = self.mmio.read(self.REG_CONTROL)
        edge = "Falling" if (ctrl & self.BIT_EDGE_FALLING) else "Rising"
        mode = "Auto" if (ctrl & self.BIT_AUTO_MODE) else ("Single" if (ctrl & self.BIT_SINGLE_SHOT) else "Normal")
        armed = "ARMED" if (ctrl & self.BIT_ARM) else "DISARMED"
        thresh = self.get_threshold()
        return f"<HardwareTrigger: {armed}, Mode={mode}, Edge={edge}, Threshold={thresh:.2f}V>"