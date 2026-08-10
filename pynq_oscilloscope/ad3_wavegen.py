import time
import threading
from typing import Optional
from pydwf import DwfLibrary, DwfAnalogOutFunction, DwfAnalogOutNode
from pydwf.utilities import openDwfDevice

class AD3SignalGenerator:
    """
    High-level, non-blocking multi-threaded wrapper for Digilent Analog Discovery 3 (AD3) Wavegen.
    Executes physical signal generation in a background thread to keep Jupyter kernels responsive.
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
        
        # Target state parameters
        self.shape_name = "Sine"
        self.frequency = 10000.0  # 10 kHz
        self.amplitude = 1.5      # 1.5V
        self.offset = 1.65        # 1.65V
        self.channel = 0          # Channel 1 (W1)

    def _worker(self):
        """Background thread execution loop."""
        try:
            self._device_handle = openDwfDevice(self.dwf)
            wavegen = self._device_handle.analogOut
            
            # Initial configuration
            func_enum = self.WAVEFORM_MAP.get(self.shape_name, DwfAnalogOutFunction.Sine)
            wavegen.nodeEnableSet(self.channel, DwfAnalogOutNode.Carrier, True)
            wavegen.nodeFunctionSet(self.channel, DwfAnalogOutNode.Carrier, func_enum)
            wavegen.nodeFrequencySet(self.channel, DwfAnalogOutNode.Carrier, self.frequency)
            wavegen.nodeAmplitudeSet(self.channel, DwfAnalogOutNode.Carrier, self.amplitude)
            wavegen.nodeOffsetSet(self.channel, DwfAnalogOutNode.Carrier, self.offset)
            
            wavegen.configure(self.channel, True)
            self.is_ready = True
            print("[AD3] Wavegen initialized and signal active.")
            
            prev_shape = self.shape_name
            prev_freq = self.frequency
            prev_amp = self.amplitude
            prev_offset = self.offset
            
            while self.is_running:
                # Track shape changes
                if self.shape_name != prev_shape:
                    prev_shape = self.shape_name
                    func_enum = self.WAVEFORM_MAP.get(prev_shape, DwfAnalogOutFunction.Sine)
                    wavegen.nodeFunctionSet(self.channel, DwfAnalogOutNode.Carrier, func_enum)
                    wavegen.configure(self.channel, True)
                    
                # Track frequency changes
                if self.frequency != prev_freq:
                    prev_freq = self.frequency
                    wavegen.nodeFrequencySet(self.channel, DwfAnalogOutNode.Carrier, prev_freq)
                    wavegen.configure(self.channel, True)
                    
                # Track amplitude changes
                if self.amplitude != prev_amp:
                    prev_amp = self.amplitude
                    wavegen.nodeAmplitudeSet(self.channel, DwfAnalogOutNode.Carrier, prev_amp)
                    wavegen.configure(self.channel, True)

                # Track offset changes (FIXED!)
                if self.offset != prev_offset:
                    prev_offset = self.offset
                    wavegen.nodeOffsetSet(self.channel, DwfAnalogOutNode.Carrier, prev_offset)
                    wavegen.configure(self.channel, True)
                    
                time.sleep(0.05)
                
            self.is_ready = False
            wavegen.configure(self.channel, False)
            if self._device_handle:
                self._device_handle.close()
                self._device_handle = None
            print("[AD3] Wavegen stopped and hardware handle released.")
            
        except Exception as e:
            print(f"[AD3] Error in background wavegen thread: {e}")
            self.is_ready = False
            self.is_running = False

    def start(self, shape: str = "Sine", frequency: float = 10000.0, amplitude: float = 1.5, offset: float = 1.65):
        if self.is_running:
            print("[AD3] Wavegen is already running.")
            return

        self.is_ready = False
        self.shape_name = shape
        self.frequency = float(frequency)
        self.amplitude = float(amplitude)
        self.offset = float(offset)
        self.is_running = True
        
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def update_parameters(self, shape: str = None, frequency: float = None, amplitude: float = None, offset: float = None):
        """Dynamically update parameters (including DC offset) while active."""
        if shape is not None and shape in self.WAVEFORM_MAP:
            self.shape_name = shape
        if frequency is not None:
            self.frequency = float(frequency)
        if amplitude is not None:
            self.amplitude = float(amplitude)
        if offset is not None:
            self.offset = float(offset)

    def stop(self):
        self.is_ready = False
        if self.is_running:
            self.is_running = False
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=2.0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()