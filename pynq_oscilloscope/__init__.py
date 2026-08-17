from pynq_oscilloscope.overlay import OscilloscopeOverlay
from pynq_oscilloscope.hw_trigger import HardwareTrigger
from pynq_oscilloscope.loader import HardwareLoader
from pynq_oscilloscope.env_checker import install_ad3_drivers, check_usb_permissions
from pynq_oscilloscope.xadc_dma import StreamingXADC
from pynq_oscilloscope.fft_dma import StreamingFFT
from pynq_oscilloscope.ad3_wavegen import AD3SignalGenerator
from pynq_oscilloscope.dashboard import OscilloscopeDashboard
from pynq_oscilloscope.notebooks import copy_notebooks

__version__ = "1.3.0"
__all__ = [
    "OscilloscopeOverlay",
    "HardwareTrigger",
    "HardwareLoader",
    "install_ad3_drivers",
    "check_usb_permissions",
    "StreamingXADC",
    "StreamingFFT",
    "AD3SignalGenerator",
    "OscilloscopeDashboard",
    "copy_notebooks",
]
