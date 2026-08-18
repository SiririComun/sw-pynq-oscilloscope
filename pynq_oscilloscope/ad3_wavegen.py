"""
pynq_oscilloscope.ad3_wavegen: Non-blocking multi-threaded wrapper for Digilent AD3 Wavegen.
Features auto-detection to gracefully handle environments where AD3 is not physically connected.
"""

import time
import threading
from typing import Optional, Dict
from pydwf import DwfLibrary, DwfAnalogOutFunction, DwfAnalogOutNode
from pydwf.utilities import openDwfDevice


class AD3SignalGenerator:
    """
    High-level, non-blocking multi-threaded wrapper for Digilent Analog Discovery 3 (AD3) Wavegen.
    Supports concurrent dual-channel analog signal generation:
      - Channel 1 (W1) -> PYNQ-Z2 Arduino Header Pin A0 (Vaux1)
      - Channel 2 (W2) -> PYNQ-Z2 Arduino Header Pin A1 (Vaux9)
    """

    WAVEFORM_MAP = {
        "Sine": DwfAnalogOutFunction.Sine,
        "Square": DwfAnalogOutFunction.Square,
        "Triangle": DwfAnalogOutFunction.Triangle
    }

    def __init__(self):
        self.dwf = DwfLibrary()
        self.is_running = False
        self.is_ready = False
        self._thread: Optional[threading.Thread] = None
        self._device_handle = None
        
        # Channel 1 Parameters (W1 -> Pin A0)
        self.ch1 = {
            "shape": "Sine",
            "frequency": 1000.0,
            "amplitude": 1.0,
            "offset": 1.65,
            "enabled": True
        }
        
        # Channel 2 Parameters (W2 -> Pin A1)
        self.ch2 = {
            "shape": "Square",
            "frequency": 5000.0,
            "amplitude": 1.0,
            "offset": 1.65,
            "enabled": True
        }

    def has_device(self) -> bool:
        """Checks if a Digilent device is physically enumerated on the USB bus."""
        try:
            return self.dwf.enum.count() > 0
        except Exception:
            return False

    def _configure_channel(self, wavegen, ch_index: int, cfg: Dict):
        """Applies hardware settings to a specific wavegen channel."""
        func_enum = self.WAVEFORM_MAP.get(cfg["shape"], DwfAnalogOutFunction.Sine)
        wavegen.nodeEnableSet(ch_index, DwfAnalogOutNode.Carrier, cfg["enabled"])
        if cfg["enabled"]:
            wavegen.nodeFunctionSet(ch_index, DwfAnalogOutNode.Carrier, func_enum)
            wavegen.nodeFrequencySet(ch_index, DwfAnalogOutNode.Carrier, cfg["frequency"])
            wavegen.nodeAmplitudeSet(ch_index, DwfAnalogOutNode.Carrier, cfg["amplitude"])
            wavegen.nodeOffsetSet(ch_index, DwfAnalogOutNode.Carrier, cfg["offset"])
        wavegen.configure(ch_index, cfg["enabled"])

    def _worker(self):
        """Background thread execution loop."""
        try:
            if not self.has_device():
                self.is_ready = False
                self.is_running = False
                return

            self._device_handle = openDwfDevice(self.dwf)
            wavegen = self._device_handle.analogOut
            
            # Initial hardware configuration for both channels
            self._configure_channel(wavegen, 0, self.ch1)
            self._configure_channel(wavegen, 1, self.ch2)
            
            self.is_ready = True
            print("[AD3] Dual Wavegen active: W1 (CH1) -> A0, W2 (CH2) -> A1.")
            
            prev_ch1 = self.ch1.copy()
            prev_ch2 = self.ch2.copy()
            
            while self.is_running:
                # Track Channel 1 updates
                if self.ch1 != prev_ch1:
                    self._configure_channel(wavegen, 0, self.ch1)
                    prev_ch1 = self.ch1.copy()
                    
                # Track Channel 2 updates
                if self.ch2 != prev_ch2:
                    self._configure_channel(wavegen, 1, self.ch2)
                    prev_ch2 = self.ch2.copy()
                    
                time.sleep(0.05)
                
            self.is_ready = False
            wavegen.configure(0, False)
            wavegen.configure(1, False)
            if self._device_handle:
                self._device_handle.close()
                self._device_handle = None
            print("[AD3] Wavegen stopped and hardware handle released.")
            
        except Exception as e:
            if self.is_ready:
                print(f"[AD3] Error in background wavegen thread: {e}")
            self.is_ready = False
            self.is_running = False

    def start(
        self,
        shape: str = "Sine",
        frequency: float = 1000.0,
        amplitude: float = 1.0,
        offset: float = 1.65,
        ch2_shape: str = "Square",
        ch2_frequency: float = 5000.0,
        ch2_amplitude: float = 1.0,
        ch2_offset: float = 1.65,
        enable_ch2: bool = True
    ) -> bool:
        """
        Starts background dual-channel signal generation.
        Returns True if hardware was detected and launched, False if bypassed.
        """
        if self.is_running:
            return True

        if not self.has_device():
            self.is_ready = False
            self.is_running = False
            return False

        self.is_ready = False
        self.ch1 = {
            "shape": shape,
            "frequency": float(frequency),
            "amplitude": float(amplitude),
            "offset": float(offset),
            "enabled": True
        }
        self.ch2 = {
            "shape": ch2_shape,
            "frequency": float(ch2_frequency),
            "amplitude": float(ch2_amplitude),
            "offset": float(ch2_offset),
            "enabled": bool(enable_ch2)
        }
        self.is_running = True
        
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        return True

    def update_ch1(self, shape: str = None, frequency: float = None, amplitude: float = None, offset: float = None):
        if shape is not None and shape in self.WAVEFORM_MAP:
            self.ch1["shape"] = shape
        if frequency is not None:
            self.ch1["frequency"] = float(frequency)
        if amplitude is not None:
            self.ch1["amplitude"] = float(amplitude)
        if offset is not None:
            self.ch1["offset"] = float(offset)

    def update_ch2(self, shape: str = None, frequency: float = None, amplitude: float = None, offset: float = None, enabled: bool = None):
        if shape is not None and shape in self.WAVEFORM_MAP:
            self.ch2["shape"] = shape
        if frequency is not None:
            self.ch2["frequency"] = float(frequency)
        if amplitude is not None:
            self.ch2["amplitude"] = float(amplitude)
        if offset is not None:
            self.ch2["offset"] = float(offset)
        if enabled is not None:
            self.ch2["enabled"] = bool(enabled)

    def update_parameters(self, shape: str = None, frequency: float = None, amplitude: float = None, offset: float = None):
        self.update_ch1(shape=shape, frequency=frequency, amplitude=amplitude, offset=offset)

    def stop(self):
        self.is_ready = False
        if self.is_running:
            self.is_running = False
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=1.0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()